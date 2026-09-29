"""Regenerate the compact, explicitly simulated example snapshot."""

from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mai.pipeline import run_pipeline


def main():
    destination = ROOT / "examples/output"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "run"
        run_pipeline(ROOT / "configs/demo.json", output)
        for name in ("summary.json", "config.json", "contextual.csv", "reliability.csv", "report.md", "report.html"):
            shutil.copyfile(output / name, destination / name)
    report = destination / "report.html"
    markup = report.read_text(encoding="utf-8")
    links = " · ".join(f"<a href='{name}'>{name}</a>" for name in ("summary.json", "contextual.csv", "reliability.csv", "README.md"))
    markup = re.sub(r"<h2>Artifacts</h2><p>.*?</p>", "<h2>Snapshot files</h2><p>" + links + "</p>", markup)
    report.write_text(markup, encoding="utf-8")
    md = destination / "report.md"
    text = md.read_text(encoding="utf-8").replace(
        "See `report.html` and `summary.json` for paired effects, confidence intervals, calibration, and provenance.",
        "See [summary.json](summary.json) for paired effects, confidence intervals, calibration, and provenance. "
        "Download and open [report.html](report.html) in a browser for the formatted report. "
        "See the [snapshot notes](README.md) to regenerate the complete audit artifacts.")
    md.write_text(text, encoding="utf-8")
    print("Refreshed examples/output with a compact simulated snapshot.")


if __name__ == "__main__":
    main()
