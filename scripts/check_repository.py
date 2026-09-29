"""Check documentation links, paper assets, and aggregate table structure offline."""

import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [ROOT / "README.md"]
    for folder in ("docs", "data", "figures", "examples"):
        files.extend((ROOT / folder).rglob("*.md"))
    errors = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "\\operatorname" in text:
            errors.append(f"{path.relative_to(ROOT)}: unsupported math macro")
        urls = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'(?:src|href)="([^"]+)"', text)
        for value in urls:
            value = value.strip("<>")
            parts = urlsplit(value)
            if parts.scheme or not parts.path:
                continue
            if not (path.parent / unquote(parts.path)).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link {value}")
    manifest = json.loads((ROOT / "figures/manifest.json").read_text(encoding="utf-8"))
    if len(manifest["figures"]) != 9:
        errors.append("Expected all nine manuscript figures")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for item in manifest["figures"]:
        path = ROOT / "figures" / item["file"]
        data = path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(data).hexdigest() != item["sha256"]:
            errors.append(f"Figure mismatch: {item['file']}")
        if f"](figures/{item['file']})" not in readme:
            errors.append(f"Figure missing from overview: {item['file']}")
    counts = {"initial_validation.csv": 7, "external_interventions.csv": 12,
              "internal_interventions.csv": 9, "reliability_accuracy.csv": 12, "reliability_effects.csv": 6}
    for name, count in counts.items():
        with (ROOT / "data/paper" / name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != count or any(None in row or None in row.values() for row in rows):
            errors.append(f"Malformed aggregate table: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(files)} Markdown documents, nine paper figures, and five aggregate tables.")


if __name__ == "__main__":
    main()
