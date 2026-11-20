"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantized-legal-tagger-monitor", description="Legal aid workers use this to evaluate the size and inference speed of a quantized legal clause tagger, ensuring efficient document processing on local systems.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("prepare-data", help="Milestone 1: load and preprocess LEDGAR dataset")
    sub.add_parser("train", help="Milestone 2: fine-tune GPT-2 model on legal clauses")
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


if __name__ == "__main__":
    main()
