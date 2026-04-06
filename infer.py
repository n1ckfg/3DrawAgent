import argparse
from pathlib import Path

from dotenv import load_dotenv

from core import (
    BatchRenderer,
    generate_candidates,
    load_experiences,
    load_prompts,
    save_rendered_images,
    safe_filename,
)
from llm import LLMClient


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for 3DrawAgent.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Single prompt to generate.")
    group.add_argument("--prompts", help="JSON file with prompts.")
    parser.add_argument(
        "--experiences",
        default=None,
        help="Optional JSON file with the experience library.",
    )
    parser.add_argument("--output-dir", default="outputs/infer", help="Output directory.")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=6000, help="LLM max tokens.")
    parser.add_argument("--cameras", default="assets/cameras.json", help="Camera definition file.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda:0.")
    return parser.parse_args()


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    args = parse_args()

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = load_prompts(args.prompts)

    experiences = load_experiences(args.experiences)
    llm = LLMClient()
    renderer = BatchRenderer(cameras_path=args.cameras, device=args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_prompt = len(prompts) == 1
    for index, theme in enumerate(prompts):
        item = generate_candidates(
            llm=llm,
            theme=theme,
            experiences=experiences,
            num_samples=1,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )[0]

        if single_prompt:
            item_dir = output_dir
        else:
            item_dir = output_dir / f"{index:03d}_{safe_filename(theme)}"
            item_dir.mkdir(parents=True, exist_ok=True)

        if not item["valid"]:
            print(f"Failed: {theme}")
            print(f"  {item.get('error')}")
            continue

        images = renderer.render(item["curves"])
        save_rendered_images(images, item_dir)
        print(f"Saved renders for: {theme} -> {item_dir}")


if __name__ == "__main__":
    main()
