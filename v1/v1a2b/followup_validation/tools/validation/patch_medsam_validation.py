from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    args = parser.parse_args()

    path = Path(args.script)
    text = path.read_text(encoding="utf-8")
    original = text
    text = text.replace("details.mode", 'details["mode"]')
    text = text.replace("details.expansion_scale", 'details["expansion_scale"]')
    path.write_text(text, encoding="utf-8")
    print("Patched pandas column access:" if text != original else "Pandas column access already patched:", path)


if __name__ == "__main__":
    main()
