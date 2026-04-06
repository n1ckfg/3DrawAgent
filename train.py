import argparse
from pathlib import Path

from dotenv import load_dotenv

from core import (
    best_candidate,
    contrast_pair,
    extract_new_experiences,
    generate_candidates,
    load_experiences,
    load_prompts,
    merge_experiences,
    save_json,
    score_candidates,
    valid_candidates,
    CLIPReward,
)
from llm import LLMClient


def parse_args():
    parser = argparse.ArgumentParser(description="Training-free experience learning for 3DrawAgent.")
    parser.add_argument("--prompts", default="examples/prompts.json", help="JSON file with training prompts.")
    parser.add_argument(
        "--experiences",
        default=None,
        help="Optional JSON file with an initial experience list.",
    )
    parser.add_argument("--output-dir", default="outputs/train", help="Directory for training outputs.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of learning epochs.")
    parser.add_argument("--samples", type=int, default=5, help="Number of candidates per prompt.")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature during training.")
    parser.add_argument("--max-tokens", type=int, default=6000, help="LLM max tokens for generation and critique.")
    parser.add_argument(
        "--max-new-experiences",
        type=int,
        default=2,
        help="Maximum number of new experiences extracted from one prompt.",
    )
    parser.add_argument(
        "--max-experiences",
        type=int,
        default=100,
        help="Maximum size of the experience library.",
    )
    parser.add_argument("--cameras", default="assets/cameras.json", help="Camera definition file.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda:0.")
    return parser.parse_args()


def slim_candidate(item):
    if item is None:
        return None
    return {
        "sample_id": item["sample_id"],
        "valid": item["valid"],
        "reward": item["reward"],
        "error": item.get("error"),
        "response": item["response"],
        "curves": item.get("curves", []),
    }


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    args = parse_args()

    prompts = load_prompts(args.prompts)
    experiences = load_experiences(args.experiences)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    scorer = CLIPReward(cameras_path=args.cameras, device=args.device)

    print(f"Loaded {len(prompts)} prompts.")
    print(f"Loaded {len(experiences)} existing experiences.")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_records = []

        for prompt_index, theme in enumerate(prompts, start=1):
            print(f"[{prompt_index}/{len(prompts)}] {theme}")

            candidates = generate_candidates(
                llm=llm,
                theme=theme,
                experiences=experiences,
                num_samples=args.samples,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            score_candidates(candidates, scorer, theme)

            best, worst = contrast_pair(candidates)
            new_items = []
            if best is not None and worst is not None:
                try:
                    new_items = extract_new_experiences(
                        llm=llm,
                        theme=theme,
                        best=best,
                        worst=worst,
                        experiences=experiences,
                        max_new_experiences=args.max_new_experiences,
                        max_tokens=args.max_tokens,
                    )
                except Exception as exc:
                    print(f"  experience extraction failed: {exc}")

            experiences = merge_experiences(
                experiences,
                new_items,
                limit=args.max_experiences,
            )

            scores = [item["reward"] for item in valid_candidates(candidates)]
            best_item = best_candidate(candidates)
            best_score = best_item["reward"] if best_item else 0.0

            print(
                f"  valid={len(valid_candidates(candidates))}/{len(candidates)} "
                f"best={best_score:.4f} new_exp={len(new_items)} total_exp={len(experiences)}"
            )

            epoch_records.append(
                {
                    "prompt": theme,
                    "scores": scores,
                    "best": slim_candidate(best),
                    "worst": slim_candidate(worst),
                    "new_experiences": new_items,
                    "experience_count": len(experiences),
                }
            )

        save_json(output_dir / f"epoch_{epoch + 1:02d}.json", epoch_records)
        save_json(output_dir / "experiences.json", experiences)

    print(f"\nSaved experiences to {output_dir / 'experiences.json'}")


if __name__ == "__main__":
    main()
