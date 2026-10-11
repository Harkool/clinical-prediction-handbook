"""Python subprocess entry point used by run.py; paths are repository-relative."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("quick", "book"), default="quick")
    args = parser.parse_args()
    from cph.course import run_course
    run_course(args.chapter, Path(args.output), args.mode)


if __name__ == "__main__":
    main()
