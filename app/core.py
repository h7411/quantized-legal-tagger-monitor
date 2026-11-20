"""Quantized Legal Tagger Monitor: Core ML and Quantization Engine.

Builds a quantized GPT-2 legal clause tagger and monitors model size,
inference latency, and tagging accuracy for legal aid workers.
"""
import copy
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Curated LEDGAR Legal Clause Dataset
# Representative SEC contract provisions across common legal categories.
# -----------------------------------------------------------------------------
LEDGAR_CATEGORIES = [
    "Governing Law",
    "Indemnification",
    "Termination",
    "Confidentiality",
    "Amendments",
    "Notices",
    "Severability",
    "Warranties",
]

RAW_LEDGAR_CLAUSES = [
    # Governing Law
    {"text": "This Agreement shall be governed by, and construed and enforced in accordance with, the laws of the State of Delaware, without giving effect to any conflicts of law principles.", "label": "Governing Law"},
    {"text": "The validity, interpretation, and performance of this Contract shall be controlled by and construed under the laws of the State of New York.", "label": "Governing Law"},
    {"text": "This Deed and any dispute or claim arising out of or in connection with it or its subject matter shall be governed by English law.", "label": "Governing Law"},
    
    # Indemnification
    {"text": "The Company agrees to defend, indemnify, and hold harmless the Executive and their heirs against any damages, liabilities, or losses incurred in connection with legal proceedings.", "label": "Indemnification"},
    {"text": "Vendor shall indemnify, protect, and hold harmless Client from any claims, suits, or damages arising out of Vendor's gross negligence or willful misconduct.", "label": "Indemnification"},
    {"text": "Borrower covenants to indemnify and defend Lender against all losses, expenses, and reasonable attorneys fees resulting from breach of warranty.", "label": "Indemnification"},

    # Termination
    {"text": "Either party may terminate this Agreement immediately upon written notice if the other party breaches any material covenant and fails to cure within thirty days.", "label": "Termination"},
    {"text": "This contract shall automatically terminate upon bankruptcy, insolvency, or dissolution of the Service Provider.", "label": "Termination"},
    {"text": "The Client reserves the right to terminate this Statement of Work for convenience upon giving forty-five days prior written notice.", "label": "Termination"},

    # Confidentiality
    {"text": "Each party shall hold in strict confidence and not disclose to any third party all Proprietary and Confidential Information received under this Agreement.", "label": "Confidentiality"},
    {"text": "The Receiving Party agrees to use reasonable care to prevent unauthorized dissemination or publication of the Disclosing Party's trade secrets.", "label": "Confidentiality"},
    {"text": "All technical data, source code, and customer records disclosed during negotiations shall remain confidential for five years following termination.", "label": "Confidentiality"},

    # Amendments
    {"text": "No amendment, modification, or supplement of this Agreement shall be binding unless executed in writing and signed by authorized representatives of both parties.", "label": "Amendments"},
    {"text": "This Agreement may not be amended, altered, or modified except by an instrument in writing signed by each of the parties hereto.", "label": "Amendments"},
    {"text": "Any waiver or modification of any term of this Contract must be expressed in writing and duly signed by the Executive Committee.", "label": "Amendments"},

    # Notices
    {"text": "All notices and other communications hereunder shall be in writing and delivered personally, or sent by certified registered mail, return receipt requested.", "label": "Notices"},
    {"text": "Any notice given by either party under this Contract shall be sent to the address set forth on the signature page or electronically to confirmed email.", "label": "Notices"},
    {"text": "Notices shall be deemed effectively given upon receipt when delivered by overnight courier service with tracking confirmation.", "label": "Notices"},

    # Severability
    {"text": "If any provision of this Agreement is held to be invalid, illegal, or unenforceable, the validity of the remaining provisions shall not in any way be impaired.", "label": "Severability"},
    {"text": "In the event that any court of competent jurisdiction finds any clause unenforceable, the parties request the court reform such provision to the minimum extent necessary.", "label": "Severability"},
    {"text": "The invalidity or unenforceability of any section of this Contract shall not affect the enforceability of any other section herein.", "label": "Severability"},

    # Warranties
    {"text": "The Service Provider warrants that all deliverables will conform to the specifications and be performed in a professional and workmanlike manner.", "label": "Warranties"},
    {"text": "Supplier warrants that the goods delivered are free from defects in design, materials, and workmanship for twelve months from delivery.", "label": "Warranties"},
    {"text": "Except as expressly stated herein, neither party makes any express or implied representations or warranties of merchantability or fitness for a particular purpose.", "label": "Warranties"},
]


def clean_clause(text: str) -> str:
    """Preprocess and normalize legal clause text."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text.strip()


def format_classification_prompt(clause: str, tag: Optional[str] = None) -> str:
    """Format a clause into classification-as-generation prompt format."""
    cleaned = clean_clause(clause)
    if tag:
        return f"Clause: {cleaned}\nTag: {tag}"
    return f"Clause: {cleaned}\nTag:"


class LegalClauseTokenizer:
    """Lightweight BPE/Word tokenizer tailored for legal clause classification-as-generation."""
    
    PAD_TOKEN = "<|pad|>"
    EOS_TOKEN = "<|endoftext|>"
    UNK_TOKEN = "<|unk|>"

    def __init__(self, clauses: Optional[List[Dict[str, str]]] = None):
        self.special_tokens = [self.PAD_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_vocab(clauses or RAW_LEDGAR_CLAUSES)

    def _tokenize_text(self, text: str) -> List[str]:
        # Split into alphanumeric tokens, punctuation, and newlines
        return re.findall(r"\w+|[^\w\s]|\n", text)

    def _build_vocab(self, data: List[Dict[str, str]]) -> None:
        vocab = list(self.special_tokens)
        
        # Add labels directly as single tokens
        for cat in LEDGAR_CATEGORIES:
            if cat not in vocab:
                vocab.append(cat)
                
        # Add prompt structural markers
        for marker in ["Clause", ":", "\n", "Tag"]:
            if marker not in vocab:
                vocab.append(marker)

        # Collect corpus words
        word_counts: Dict[str, int] = {}
        for item in data:
            prompt = format_classification_prompt(item["text"], item.get("label"))
            tokens = self._tokenize_text(prompt)
            for tok in tokens:
                word_counts[tok] = word_counts.get(tok, 0) + 1

        for word in sorted(word_counts.keys()):
            if word not in vocab:
                vocab.append(word)

        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    def encode(self, text: str, max_length: Optional[int] = None, padding: bool = False) -> List[int]:
        tokens = self._tokenize_text(text)
        ids = [self.token_to_id.get(tok, self.token_to_id[self.UNK_TOKEN]) for tok in tokens]
        if max_length is not None:
            ids = ids[:max_length]
            if padding and len(ids) < max_length:
                ids += [self.pad_token_id] * (max_length - len(ids))
        return ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.id_to_token.get(idx, self.UNK_TOKEN) for idx in token_ids if idx != self.pad_token_id]
        # Reconstruct string, respecting spaces and newlines
        result = []
        for i, tok in enumerate(tokens):
            if tok == "\n":
                result.append(tok)
            elif i > 0 and tokens[i - 1] not in ["\n"] and tok not in [",", ".", ":", ";", ")", "]", "'", '"']:
                result.append(" " + tok)
            else:
                result.append(tok)
        return "".join(result)


def prepare_ledgar_subset(
    raw_data: Optional[List[Dict[str, str]]] = None,
    test_ratio: float = 0.25,
    max_length: int = 64
) -> Dict[str, Any]:
    """Prepare and tokenize train and test splits for the LEDGAR subset."""
    data = raw_data or RAW_LEDGAR_CLAUSES
    tokenizer = LegalClauseTokenizer(data)

    # Stratified split by category
    category_buckets: Dict[str, List[Dict[str, str]]] = {cat: [] for cat in LEDGAR_CATEGORIES}
    for item in data:
        cat = item["label"]
        category_buckets.setdefault(cat, []).append(item)

    train_data: List[Dict[str, Any]] = []
    test_data: List[Dict[str, Any]] = []

    for cat, items in category_buckets.items():
        n_test = max(1, int(len(items) * test_ratio))
        test_items = items[:n_test]
        train_items = items[n_test:]
        
        for item in train_items:
            prompt = format_classification_prompt(item["text"], item["label"])
            input_ids = tokenizer.encode(prompt, max_length=max_length, padding=True)
            train_data.append({
                "text": item["text"],
                "label": item["label"],
                "prompt": prompt,
                "input_ids": input_ids,
            })
        for item in test_items:
            eval_prompt = format_classification_prompt(item["text"])
            test_data.append({
                "text": item["text"],
                "label": item["label"],
                "prompt": eval_prompt,
                "target_prompt": format_classification_prompt(item["text"], item["label"]),
                "input_ids": tokenizer.encode(eval_prompt, max_length=max_length, padding=False),
            })

    return {
        "tokenizer": tokenizer,
        "train_data": train_data,
        "test_data": test_data,
        "categories": LEDGAR_CATEGORIES,
    }


# --- Milestone 1: Prepare a subset of the LEDGAR dataset for legal clause tagging.
def milestone_1() -> Dict[str, Any]:
    """Prepare, clean, and tokenize a curated subset of the LEDGAR dataset."""
    prepared = prepare_ledgar_subset()
    tokenizer = prepared["tokenizer"]
    train_data = prepared["train_data"]
    test_data = prepared["test_data"]
    
    sample = train_data[0]
    sample_encoded = sample["input_ids"]
    sample_decoded = tokenizer.decode(sample_encoded)

    return {
        "status": "completed",
        "dataset_name": "LEDGAR-subset",
        "categories": prepared["categories"],
        "num_categories": len(prepared["categories"]),
        "train_count": len(train_data),
        "test_count": len(test_data),
        "total_samples": len(train_data) + len(test_data),
        "vocab_size": tokenizer.vocab_size,
        "sample_prompt": sample["prompt"],
        "sample_decoded": sample_decoded,
        "tokenizer": tokenizer,
        "train_data": train_data,
        "test_data": test_data,
    }


# --- Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Apply dynamic and static post-training quantization to the trained model.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Present a dashboard-like output of model efficiency metrics.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
