"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantized-legal-tagger-monitor", description="Legal aid workers use this to evaluate the size and inference speed of a quantized legal clause tagger, ensuring efficient document processing on local systems.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("prepare-data", help="Milestone 1: load and preprocess LEDGAR dataset")
    sub.add_parser("train", help="Milestone 2: fine-tune GPT-2 model on legal clauses")
    sub.add_parser("evaluate", help="Milestone 3: evaluate accuracy, precision, recall, and F1")
    sub.add_parser("quantize", help="Milestone 4: apply dynamic and static quantization")
    sub.add_parser("benchmark", help="Milestone 5: benchmark model size and inference speed")
    sub.add_parser("dashboard", help="Milestone 6: present comparative efficiency dashboard")

    tag_parser = sub.add_parser("tag", help="Tag a legal clause using the model")
    tag_parser.add_argument("--clause", required=True, help="Clause text to classify")
    tag_parser.add_argument("--quantized", action="store_true", help="Use static quantized model")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Quantized Legal Tagger Monitor is ready. Open README.md for your milestones.")
    elif args.command == "prepare-data":
        from app.core import milestone_1
        res = milestone_1()
        print(f"Prepared {res['total_samples']} clauses across {res['num_categories']} categories (vocab: {res['vocab_size']}).")
    elif args.command == "train":
        from app.core import milestone_2
        res = milestone_2()
        print(f"Fine-tuned GPT-2 ({res['num_parameters']:,} params). Initial loss: {res['initial_loss']:.4f}, Final loss: {res['final_loss']:.4f}.")
    elif args.command == "evaluate":
        from app.core import milestone_3
        res = milestone_3()
        print(f"Evaluation: Accuracy={res['accuracy']:.2%}, Precision={res['precision']:.2%}, Recall={res['recall']:.2%}, F1={res['f1_score']:.2%}")
    elif args.command == "quantize":
        from app.core import milestone_4
        res = milestone_4()
        print("Quantization applied successfully:")
        print(f"  - Dynamic Quantization: {res['dynamic_quantized']}")
        print(f"  - Static Quantization:  {res['static_quantized']}")
    elif args.command == "benchmark":
        from app.core import milestone_5
        res = milestone_5()
        b = res["benchmarks"]
        print(f"Original: {b['original']['size_kb']:.1f} KB, Latency: {b['original']['mean_latency_ms']:.2f} ms")
        print(f"Dynamic:  {b['dynamic']['size_kb']:.1f} KB ({b['dynamic']['compression_ratio']:.2f}x smaller), Latency: {b['dynamic']['mean_latency_ms']:.2f} ms")
        print(f"Static:   {b['static']['size_kb']:.1f} KB ({b['static']['compression_ratio']:.2f}x smaller), Latency: {b['static']['mean_latency_ms']:.2f} ms")
    elif args.command == "dashboard":
        from app.core import milestone_6
        res = milestone_6()
        print(res["dashboard"])
    elif args.command == "tag":
        from app.core import milestone_2, milestone_4, predict_clause_tag
        if args.quantized:
            m4 = milestone_4()
            model = m4["static_model"]
            m2 = milestone_2()
            tok = m2["tokenizer"]
            model_type = "Static Quantized INT8"
        else:
            m2 = milestone_2()
            model = m2["model"]
            tok = m2["tokenizer"]
            model_type = "FP32 Original"
        tag = predict_clause_tag(model, tok, args.clause)
        print(f"[{model_type}] Clause Tag: {tag}")


if __name__ == "__main__":
    main()
