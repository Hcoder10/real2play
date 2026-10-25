#!/usr/bin/env python3
"""Write a US Letter, 300 dpi ArUco board; print at 100 percent actual size."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2p.perception.aruco import BoardConfig, make_board


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("scripts/perception/board_letter.png"))
    parser.add_argument("--marker-mm", type=float, default=None)
    parser.add_argument("--gap-mm", type=float, default=None)
    args = parser.parse_args(argv)
    cfg = BoardConfig.from_dict(json.loads(args.config.read_text())) if args.config else BoardConfig()
    if args.marker_mm is not None:
        cfg.marker_length_m = args.marker_mm/1000
    if args.gap_mm is not None:
        cfg.separation_m = args.gap_mm/1000
    make_board(args.out, cfg)
    print(f"Wrote {args.out}: US Letter, 300 dpi. Print 100%, then measure a marker with a ruler.")


if __name__ == "__main__":
    main()
