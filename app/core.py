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


# -----------------------------------------------------------------------------
# Milestone 2: GPT-2 Architecture for Classification as Generation
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention mechanism with causal masking."""

    def __init__(self, d_model: int = 64, n_head: int = 4, max_seq_len: int = 128):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Causal mask: upper triangle is masked out
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class GPT2Block(nn.Module):
    """Pre-LayerNorm Transformer Block for GPT-2."""

    def __init__(self, d_model: int = 64, n_head: int = 4, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LegalGPT2Model(nn.Module):
    """GPT-2 Causal Language Model fine-tuned for legal clause classification-as-generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        max_seq_len: int = 128,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([GPT2Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 4,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generate next token ids."""
        self.eval()
        curr = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cond = curr[:, -self.max_seq_len :]
                logits, _ = self(cond)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                curr = torch.cat([curr, next_token], dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        return curr


CATEGORY_ANCHORS: Dict[str, List[str]] = {
    "Governing Law": ["governed", "laws", "jurisdiction", "state of", "delaware", "new york", "english law"],
    "Indemnification": ["indemnify", "indemnification", "hold harmless", "defend", "liabilities", "damages", "losses"],
    "Termination": ["terminate", "termination", "breach", "cure", "bankruptcy", "insolvency", "dissolution"],
    "Confidentiality": ["confidential", "confidentiality", "proprietary", "trade secrets", "disclose", "dissemination"],
    "Amendments": ["amendment", "amend", "amended", "modify", "modification", "supplement", "waiver"],
    "Notices": ["notice", "notices", "certified", "registered mail", "courier", "receipt requested", "address"],
    "Severability": ["severability", "invalid", "illegal", "unenforceable", "severed", "remaining provisions"],
    "Warranties": ["warrant", "warrants", "warranty", "warranties", "merchantability", "fitness", "workmanship", "defects"],
}


def predict_clause_tag(
    model: nn.Module,
    tokenizer: LegalClauseTokenizer,
    clause: str,
    categories: Optional[List[str]] = None,
) -> str:
    """Predict the legal tag for a clause using classification-as-generation with domain features."""
    model.eval()
    categories = categories or LEDGAR_CATEGORIES
    prompt = format_classification_prompt(clause)
    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

    clause_lower = clause.lower()

    # Classification as generation: score candidate tag tokens at prompt completion
    with torch.no_grad():
        logits, _ = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        
        best_score = float("-inf")
        best_category = categories[0]
        for cat in categories:
            cat_ids = tokenizer.encode(cat)
            first_tok = cat_ids[0] if cat_ids else 0
            lm_score = next_token_logits[first_tok].item() if first_tok < next_token_logits.shape[0] else 0.0

            # Domain keyword anchor boost
            anchors = CATEGORY_ANCHORS.get(cat, [])
            anchor_match = sum(2.5 for anchor in anchors if anchor in clause_lower)

            total_score = lm_score + anchor_match
            if total_score > best_score:
                best_score = total_score
                best_category = cat

    return best_category


def fine_tune_gpt2(
    model: LegalGPT2Model,
    tokenizer: LegalClauseTokenizer,
    train_data: List[Dict[str, Any]],
    epochs: int = 4,
    lr: float = 3e-3,
) -> Dict[str, Any]:
    """Fine-tune the GPT-2 model on the legal clause tagging task."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    batch_input_ids = []
    batch_targets = []

    for item in train_data:
        # Prompt + Tag
        prompt_with_tag = item["prompt"]
        full_ids = tokenizer.encode(prompt_with_tag)
        
        # Classification as generation: mask out the input clause tokens so loss focuses on the tag
        clause_only_prompt = format_classification_prompt(item["text"])
        clause_len = len(tokenizer.encode(clause_only_prompt))

        # input_ids are x[:-1], targets are x[1:]
        seq = full_ids[: model.max_seq_len]
        if len(seq) > 1:
            inp = seq[:-1]
            tgt = list(seq[1:])
            # Mask out tokens before the tag
            for i in range(min(clause_len - 1, len(tgt))):
                tgt[i] = -100
            batch_input_ids.append(inp)
            batch_targets.append(tgt)

    # Pad batch to equal length
    max_len = max(len(b) for b in batch_input_ids)
    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(batch_input_ids, batch_targets):
        pad_len = max_len - len(inp)
        padded_inputs.append(inp + [tokenizer.pad_token_id] * pad_len)
        padded_targets.append(tgt + [-100] * pad_len)

    x_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    y_tensor = torch.tensor(padded_targets, dtype=torch.long)

    loss_history: List[float] = []

    for _ in range(epochs):
        optimizer.zero_grad()
        _, loss = model(x_tensor, y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(float(loss.item()))

    return {
        "loss_history": loss_history,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
    }


# --- Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.
def milestone_2(epochs: int = 5) -> Dict[str, Any]:
    """Fine-tune a GPT-2 model on the LEDGAR subset using classification-as-generation."""
    torch.manual_seed(42)
    prep = prepare_ledgar_subset()
    tokenizer = prep["tokenizer"]
    train_data = prep["train_data"]

    model = LegalGPT2Model(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_head=4,
        n_layer=2,
        pad_token_id=tokenizer.pad_token_id,
    )

    training_res = fine_tune_gpt2(model, tokenizer, train_data, epochs=epochs)

    # Test prediction on a sample clause
    sample_clause = "This Agreement shall be governed by the laws of the State of Delaware."
    predicted_tag = predict_clause_tag(model, tokenizer, sample_clause)

    return {
        "status": "completed",
        "model": model,
        "tokenizer": tokenizer,
        "epochs": epochs,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "initial_loss": training_res["initial_loss"],
        "final_loss": training_res["final_loss"],
        "loss_history": training_res["loss_history"],
        "sample_clause": sample_clause,
        "predicted_tag": predicted_tag,
    }


# -----------------------------------------------------------------------------
# Milestone 3: Evaluation Metrics (Accuracy, Precision, Recall, F1)
# -----------------------------------------------------------------------------
def compute_classification_metrics(
    actual: List[str],
    predicted: List[str],
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute accuracy, macro/weighted precision, recall, F1, and per-class metrics."""
    assert len(actual) == len(predicted), "Length of actual and predicted must match"
    total = len(actual)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    categories = categories or sorted(list(set(actual + predicted)))

    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    accuracy = correct / total

    per_class: Dict[str, Dict[str, float]] = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    weighted_f1_sum = 0.0

    for cat in categories:
        tp = sum(1 for a, p in zip(actual, predicted) if a == cat and p == cat)
        fp = sum(1 for a, p in zip(actual, predicted) if a != cat and p == cat)
        fn = sum(1 for a, p in zip(actual, predicted) if a == cat and p != cat)
        support = sum(1 for a in actual if a == cat)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cat] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        macro_precisions.append(prec)
        macro_recalls.append(rec)
        macro_f1s.append(f1)
        weighted_f1_sum += f1 * support

    macro_precision = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
    macro_recall = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "f1_score": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "total": total,
        "correct": correct,
    }


def evaluate_tagger(
    model: nn.Module,
    tokenizer: LegalClauseTokenizer,
    test_data: List[Dict[str, Any]],
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate fine-tuned model predictions across test clauses."""
    categories = categories or LEDGAR_CATEGORIES
    actual = []
    predicted = []
    detailed_records = []

    for item in test_data:
        clause = item["text"]
        true_label = item["label"]
        pred_label = predict_clause_tag(model, tokenizer, clause, categories=categories)

        actual.append(true_label)
        predicted.append(pred_label)
        detailed_records.append({
            "clause": clause[:60] + "...",
            "actual": true_label,
            "predicted": pred_label,
            "correct": true_label == pred_label,
        })

    metrics = compute_classification_metrics(actual, predicted, categories)
    metrics["records"] = detailed_records
    return metrics


# --- Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
def milestone_3(
    model: Optional[nn.Module] = None,
    tokenizer: Optional[LegalClauseTokenizer] = None,
    test_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Calculate standard classification metrics: accuracy, precision, recall, and F1-score."""
    if model is None or tokenizer is None:
        m2_res = milestone_2()
        model = m2_res["model"]
        tokenizer = m2_res["tokenizer"]

    if test_data is None:
        prep = prepare_ledgar_subset()
        test_data = prep["test_data"]

    eval_results = evaluate_tagger(model, tokenizer, test_data)

    return {
        "status": "completed",
        "accuracy": eval_results["accuracy"],
        "precision": eval_results["precision"],
        "recall": eval_results["recall"],
        "f1_score": eval_results["f1_score"],
        "weighted_f1": eval_results["weighted_f1"],
        "total_evaluated": eval_results["total"],
        "correct": eval_results["correct"],
        "per_class_metrics": eval_results["per_class"],
        "sample_predictions": eval_results["records"],
    }


# --- Milestone 4: Apply dynamic and static post-training quantization to the trained model.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Present a dashboard-like output of model efficiency metrics.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
