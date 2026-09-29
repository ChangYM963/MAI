# Running the project

[Overview](../README.md) · [Method](methodology.md) · [Artifacts](architecture.md)

## Environment

Python 3.9+ is sufficient for the offline workflow and chat endpoint adapter. Run commands from the repository root. Optional installation exposes the `mai` console command:

```bash
python -m venv .venv
# Activate the environment using your shell's standard command
python -m pip install -e .
```

## Offline workflow

```bash
python -m mai pipeline --config configs/demo.json --output runs/demo
```

Open `runs/demo/report.html` directly in a browser. On PowerShell, `Invoke-Item runs/demo/report.html` opens the report with the associated application. The entire run uses synthetic responses and hidden features. A completed or failed output directory is preserved; choose `runs/demo-2` for another run.

The smaller algebra example is also available:

```bash
python -m demo.mai_demo
python -m demo.mai_demo --save-record records.jsonl
python -m mai analyze records.jsonl
```

## Chat-completions endpoint

The adapter sends one independent request per sample. Supply a trusted OpenAI-compatible endpoint ending in its API prefix, usually `/v1`; the adapter appends `/chat/completions`.

PowerShell:

```powershell
$env:MAI_BASE_URL = "https://your-endpoint.example/v1"
$env:MAI_MODEL = "your-model-id"
# Set MAI_API_KEY in your environment if authentication is required.
python -m mai pipeline --config configs/chat.json --dry-run
python -m mai pipeline --config configs/chat.json --output runs/chat
```

Bash:

```bash
export MAI_BASE_URL="https://your-endpoint.example/v1"
export MAI_MODEL="your-model-id"
# Export MAI_API_KEY if authentication is required.
python -m mai pipeline --config configs/chat.json --dry-run
python -m mai pipeline --config configs/chat.json --output runs/chat
```

`--model` and `--base-url` override those two environment variables. Credentials are read only from `MAI_API_KEY`; keep them out of configuration files. Remote endpoints use HTTPS; HTTP is accepted for localhost/127.0.0.1/::1. If a provider rejects the optional `seed` field, add `"send_seed": false` to the chat configuration. That also limits run-to-run reproducibility.

Defaults are temperature 0.7, top-p 1, maximum 16 output tokens, unconstrained generation, and strict single-letter parsing. Endpoints must implement this chat contract; reasoning-only endpoints may require their own adapter. Network and HTTP errors stop the run and write `status.json`; partial samples remain available. There is no automatic resume or paid request retry.

The dry run estimates the maximum number of generation requests. It makes no model calls. Hidden-feature extraction, when enabled locally, adds forward passes beyond that count.

## Local models and hidden intervention

```bash
python -m pip install -e ".[local]"
python -m mai pipeline --config configs/local.json --model /path/to/model --output runs/local
```

Use a local model directory or a Hugging Face model ID you can access. The optional adapter targets Transformers 4.45–4.x and PyTorch 2.2+, a tokenizer with a chat template, and a decoder with `network.model.layers`. Weights are loaded without remote custom code. CUDA uses float16; CPU uses float32. The adapter loads the whole model on one device; hardware capacity must match the chosen model.

Set `hidden.layer` to a valid zero-based block index. `hidden.alpha` controls update strength and `hidden.threshold` gates predicted risk. No automatic search over layers or thresholds is claimed. The same last-position update is applied on every generation forward pass. Chat endpoints do not expose hidden states, so `configs/chat.json` disables this branch.

## Task construction from your material

Start with the schema in [materials.json](../examples/materials.json):

```bash
python -m mai construct examples/materials.json --output construction.json
```

For model-based construction, set the chat environment variables above and add `--semantic`. This calls the model for applicable screening, normalization, rewriting, rendering, and quality roles, using a larger output budget of 1200 tokens:

```bash
python -m mai construct examples/materials.json --semantic --output semantic-construction.json
```

Inspect accepted tasks, added assumptions, rejected cases, and all raw stage outputs. To evaluate the reviewed accepted tasks, save the `accepted` array as a separate materials JSON file and point the pipeline's `materials` field at it. The main pipeline runs deterministic schema validation on this reviewed input; it does not automatically treat unreviewed semantic outputs as the original research dataset.

## Configuration

Copy a configuration before adapting it. `materials` is resolved relative to the **configuration file**, not your shell's current directory.

| Field | Meaning |
|:--|:--|
| backend | `simulation`, `chat`, or `local` |
| seed | Root seed for splitting, sampling, and inference |
| agents / samples / repetitions | Contextual agent count, responses per agent/stage, repeated evaluations |
| prior_beta | Fixed-prior mixing coefficient |
| calibration_tasks / calibration_rounds | Disjoint calibration subset size and repeated calibration rounds |
| prompt_strengths | Distinct candidate strengths in [0,1] |
| postprocess_lambda | Probability-shrinkage coefficient in [0,1] |
| hidden | `enabled`, zero-based `layer`, `alpha`, and risk `threshold` |
| reliability_variants | Five tasks per family per variant, 15 total per variant |
| reliability_samples | Responses per agent/stage in A–D reliability |
| bootstrap_replicates / sign_flip_replicates | Resampling effort; use 10000 for paper-scale inference |
| send_seed | Optional chat setting; defaults to true |

At least one accepted task must remain after calibration. A shared configuration uses the same agent count and evaluation repetitions for both paths; use separate runs when matching the paper's different sample/agent counts.

## Standalone reliability evaluation

```bash
python -m mai generate --variants 4 --output tasks.json
python -m mai run --tasks tasks.json --output records.jsonl --agents 5 --samples 3 --repetitions 2
python -m mai analyze records.jsonl --inference --bootstrap 10000 --output analysis.json
```

These commands use the configured chat endpoint. Four variants produce 60 new verified tasks. `analyze` without `--inference` prints descriptive estimates. Inference rejects duplicate task/repetition records and incomplete repetition groups. For complete prompt/seed-level audit logs, use the full pipeline.

## Checks and common issues

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
```

| Symptom | Action |
|:--|:--|
| Output directory exists | Select a fresh directory; keep the original audit artifacts |
| No selected prompt strength | Inspect undefined calibration round means and the recorded status; add representative calibration material |
| Hidden controller unavailable | Inspect `hidden_controller.json` for insufficient positive examples or a zero direction |
| Many abstentions | Inspect raw responses and endpoint generation settings; do not repair answers post hoc |
| Invalid target block | Select a supported architecture and valid zero-based layer |
| API error | Check endpoint contract, authentication, model ID, connectivity, and optional seed support |
| GitHub math/image failure | Use the repository-relative image links; equations use supported `math` fences and basic math commands |

GitHub's math syntax is documented in its [official guide](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions). Every paper figure also has a direct full-resolution link.
