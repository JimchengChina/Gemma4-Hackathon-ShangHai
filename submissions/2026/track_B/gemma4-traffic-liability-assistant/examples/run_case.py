from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma4_liability_assistant import TrafficLiabilityPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Gemma 4 traffic liability demo case.")
    parser.add_argument("--video", required=True, help="Local accident video path.")
    parser.add_argument("--image", action="append", default=[], help="Local accident image path. Repeatable.")
    parser.add_argument(
        "--report",
        default="甲车沿主路直行，乙车疑似跟车过近。现场有一处人行横道，视频中可见前车突然减速。",
        help="Text report or claim narrative.",
    )
    parser.add_argument("--party", action="append", default=None, help="Party name. Repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TrafficLiabilityPipeline()
    parties = args.party or ["甲车", "乙车"]
    result = pipeline.run(
        video_path=args.video,
        text_report=args.report,
        image_paths=args.image,
        party_names=parties,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
