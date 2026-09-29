# Architecture and artifacts

The command line delegates to `pipeline.run_pipeline`. Backends share `generate(prompt, seed)`; hidden-capable backends also implement `hidden(prompt)` and `generate_steered(prompt, seed, delta)`. `AuditBackend` records each successful generation immediately. Core task generation, scoring, inference, and reports use only Python's standard library.

```text
materials.json → construction → calibration/evaluation split
                                 ├─ calibration → prompt strength + hidden controller
                                 └─ evaluation  → shared initial → social / prompts / correction / hidden
verified task generator → shared initial → A / B / C / D
                                      ↓
                          metrics → paired inference → reports + manifest
```

## Output directory

| Artifact | Contents |
|:--|:--|
| config.json | Effective configuration, without API credentials |
| status.json | `running`, `complete`, or `failed`, with an error type/message on failure |
| construction.json | Accepted/flagged/rejected tasks and source-to-output stage logs |
| contextual_tasks.json | Explicit disjoint calibration and evaluation task lists |
| contextual_calibration.jsonl | Per-task/repetition shared baseline and all strength candidates |
| prompt_calibration.json | Round scores, eligibility, common-valid tasks, selected strength |
| hidden_calibration.jsonl | Paired hidden features and risk labels from calibration observations |
| hidden_controller.json | Direction, standardization parameters, fitted weights, or unavailability reason |
| contextual_records.jsonl | Baseline raw answers, frozen references, interventions, scores, trigger audits |
| reliability_tasks.json | Verified task parameters, answer keys, wrong panel options, provenance |
| reliability_records.jsonl | Shared initial answers, independent A–D responses, branch order, settings |
| samples.jsonl | Each successful generation's prompt, seed, raw response, optional steering delta |
| summary.json | Construction counts, calibration, drift, paired effects, CIs/tests, audit counts |
| contextual.csv / reliability.csv | Small tables for subsequent plotting or inspection |
| report.html / report.md | Browser report and compact textual overview |
| manifest.json | SHA-256 for every output except the manifest itself; source-material hash, version, Python, timestamp |

## Record conventions

Task IDs identify one scenario across repetitions. Agent IDs are stable within a run. Repetitions are one-based; hidden layer indices are zero-based. Raw answers remain strings and invalid answers remain present. Undefined metrics use JSON `null`, never NaN. Seeds describe requested reproducibility; an external model service may not honor deterministic sampling.

Contextual records carry `initial`, `social`, `p0`, `priors`, `majority`, and branch analyses. A failed contextual repetition records `valid: false` and a reason. Reliability records carry `task_id`, `task_type`, `repetition`, `answer`, `incorrect_majority`, `initial`, `conditions`, `condition_order`, and `generation`. Answer keys are used for evaluation and for constructing the unlabeled external vote panel, never sent as labeled solutions.

The saved [example output](../examples/output/report.md) is a compact snapshot. Full prompts and thousands of raw samples are generated locally by the pipeline and excluded from Git by default. This keeps the repository inspectable while retaining the full audit trail for actual experiments.

## Extending the project

- Add a backend by implementing the generation contract, then select it in `backends.make_backend`.
- Add a contextual decision axis by supplying ordered options and validators in `construction.py`.
- Add a synthetic task family with both a rational solver and an independent oracle; include it in balanced generation and stratified inference.
- Keep calibration and evaluation tasks disjoint, and preserve initial responses across every comparison.
- Model-specific constrained decoding, original experimental prompts, and template-block resampling should be introduced explicitly with provenance, not silently substituted into generic runs.
