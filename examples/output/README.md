# Simulated workflow snapshot

These files were generated with [configs/demo.json](../../configs/demo.json). All responses and hidden features are simulated. These are **software demonstration outputs, not empirical paper results**.

- [report.md](report.md): readable summary on GitHub.
- [report.html](report.html): download and open locally for the formatted report.
- [summary.json](summary.json): detailed counts, calibration, drift, effects, uncertainty, and inference units.
- [contextual.csv](contextual.csv) and [reliability.csv](reliability.csv): tabular exports.
- [config.json](config.json): saved configuration; its materials path resolves from the original `configs/` directory.

This compact snapshot omits full raw prompts, samples, and intermediate files. Generate those artifacts with:

```bash
python -m mai pipeline --config configs/demo.json --output runs/demo
```

To update this tracked snapshot from the current implementation:

```bash
python scripts/refresh_example.py
```

The full local report links to all of its audit files and the hash manifest. The snapshot HTML links only to files actually included here.
