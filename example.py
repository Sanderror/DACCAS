"""Minimal end-to-end DACCAS example.

    python example.py path/to/captcha.png

Runs the two-step pipeline: Classify (dispatcher) -> Solve (routed solver).
"""
import sys
import json

from daccas import DACCAS


def main():
    if len(sys.argv) < 2:
        print("usage: python example.py <image_path> [more_images...]")
        sys.exit(1)

    # device="cuda" if you have a GPU; "cpu" otherwise.
    daccas = DACCAS(
        models_dir="models",
        charsets_dir="charsets",
        device="cpu",
        # convnext_weights_path="models/convnext_large.pth",  # if offline
        # moving_window_charset="charsets/moving_window_charset.txt",
    )

    for path in sys.argv[1:]:
        print("=" * 70)
        print("IMAGE:", path)

        # Step 1 - classify (dispatch)
        cls = daccas.Classify(path)
        print(f"  class      : {cls['class']}  (conf {cls['confidence']:.3f})")

        # Step 2 - solve (route to the matching solver)
        result = daccas.Solve(path, cls["class"])
        print("  result     :")
        print(json.dumps(result, indent=4, default=str))

    print("=" * 70)


if __name__ == "__main__":
    main()
