import argparse
from tools.eval import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation for mouse Y_meiji experiment."
    )
    parser.add_argument(
        "file",
        type=str,
        help="video file path (ex: ./data/01504.MTS)",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=["csv", "txt"],
        default="csv",
        help="format of output file",
    )
    parser.add_argument(
        "-s",
        "--shift",
        type=float,
        default=0,
        help="video shift",
    )
    parser.add_argument("--show", action="store_true", help="show visualization")
    args = parser.parse_args()
    main(args)
