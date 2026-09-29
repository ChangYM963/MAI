<p align="center"><img src="figures/cover.svg" width="1100" alt="MAI — majority-alignment drift and decision reliability"></p>

<h1 align="center">Majority-Alignment Drift and Reliability in Financial Multi-Agent LLM Systems</h1>
<p align="center"><strong>A Context-Adaptive Evaluation Framework</strong></p>

<p align="center"><a href="#research-question">Research question</a> · <a href="#method">Method</a> · <a href="#selected-results">Results</a> · <a href="#reference-code">Code</a> · <a href="#scope-and-reproducibility">Scope</a></p>

## Research question

Multi-agent LLM systems often allow decision agents to see one another's answers and revise their own. This interaction can suppress an initially useful independent judgment, especially when the group information is wrong. A final answer alone cannot show every change: the agent may keep the same option while its response distribution moves toward the majority.

This project studies two connected questions:

1. **Behavior:** How far does a decision agent's response distribution move toward the *initial* majority after seeing group information?
2. **Reliability:** When the majority is incorrect, do agents retain correct initial judgments, and does the system still choose the correct answer?

The paper uses financial decisions as its main setting. It keeps **contextual tasks** (for measuring and controlling drift) separate from **exactly solvable synthetic tasks** (for assessing correctness). A five-function construction pipeline screens, normalizes, rewrites, renders, and quality-checks contextual tasks; these functions are distinct from the decision agents under evaluation.

## Method

### Two-stage majority-alignment measurement

Each task offers five ordered actions, **A–E**, along one decision axis. Agents answer independently first. The option with the highest *mean initial response probability* across agents becomes the reference majority. It is held fixed during the second stage, when agents see group information and answer again. The initial minority group is also held fixed.

For option distribution $p$ and majority option $m$, the **Majority Alignment Index** is

$$
\operatorname{MAI}(p;m)=\sum_{k\in\{A,B,C,D,E\}}4\left(1-\frac{|r(k)-r(m)|}{\max_j|r(j)-r(m)|}\right)p(k),
$$

where $r(A)=4,\ldots,r(E)=0$. MAI lies in **[0, 4]**. Its difference between stages, $\Delta\operatorname{MAI}$, is positive when a distribution moves toward the fixed majority reference. We report mean drift among agents in the **initial minority** and the proportion of those agents who actually switch their selected option to that majority. MAI measures ordinal proximity; it is neither a correctness probability nor calibrated confidence.

The contextual analysis estimates distributions from repeated valid responses. In the paper's main pipeline, an agent-specific prior is mixed with empirical frequencies using $\beta=0.35$, with the same prior on both stages. The reference code supports this operation in [`mai/contextual.py`](mai/contextual.py). It also implements the paper's post-processing correction

$$
\widetilde p^{(1)}=p^{(0)}+(1-\lambda)(p^{(1)}-p^{(0)}),
$$

which shrinks second-stage drift by construction. The paper also investigates structured control prompts and a lightweight hidden-layer intervention; those model-specific experimental systems are described in the paper but are **not** implemented in this public reference package.

### Four-condition reliability evaluation

For tasks with an explicit correct option, the paper uses the *same initial agent responses* in four subsequent branches. Every branch requires a second answer:

| Condition | External group information | Fixed verification prompt |
|:--|:--|:--|
| A | None | No |
| B | Preset incorrect majority | No |
| C | None | Yes |
| D | Preset incorrect majority | Yes |

The external five-member panel contains **four votes for a preset wrong option and one for the correct option**. It is independent of the evaluated agents' initial answers. The verification prompt asks agents to recheck the stated facts, objective, and constraints; it does not reveal the answer.

An agent's decision is the unique mode of its valid samples. An invalid response or tied mode means abstention. The system requires a **strict majority of all scheduled agents**, including abstainers in the denominator. Incorrect consensus requires at least **80% of all agents** to select the *same* wrong option. Initial correct-judgment retention measures how many initially correct agents remain correct after the second stage.

The main contrasts are **accuracy loss** $\mathrm{Acc}_A-\mathrm{Acc}_B$, **direct recovery** $\mathrm{Acc}_D-\mathrm{Acc}_B$, and **specific protection** $(\mathrm{Acc}_D-\mathrm{Acc}_B)-(\mathrm{Acc}_C-\mathrm{Acc}_A)$. Specific protection subtracts the verification prompt's effect without group information.

## Selected results

In the initial two-stage validation, the **conditional mean positive drift** and the **Switch Rate** capture different behavior. For example, Qwen Plus has positive drift in 31 of 50 tasks, while only 1.15% of initial-minority agents switch to the majority option:

| Model | Positive-drift tasks / 50 | Mean drift in positive tasks | Switch Rate |
|:--|--:|--:|--:|
| Qwen Plus | 31 | 0.1327 | 1.15% |
| DeepSeek-v3 | 31 | 0.0922 | 1.29% |
| GLM-4.6 | 50 | 0.2069 | 3.18% |
| Doubao | 5 | 0.0312 | 0.20% |
| Qwen2.5-7B | 35 | 0.1568 | 2.60% |
| Mistral-7B | 33 | 0.1531 | 6.05% |
| Llama-3.1-8B | 24 | 0.2141 | 6.21% |

Positive drift uses the paper's strict $\varepsilon=0$ criterion in this table; extremely small floating-point changes can affect the counts. Doubao has aggregate point estimates only. The paper separately reports scenario and template resampling to evaluate uncertainty.

The paper's reliability experiment used 60 exactly solvable tasks per model (20 each of quadratic utility, loss budget, and liquidity budget), two repetitions per task, five agents, and three samples per agent per stage. Models and decoding protocols are analyzed separately. The table reports **system accuracy**:

| Model | A: no group | B: wrong majority | D: wrong majority + verification |
|:--|--:|--:|--:|
| Qwen2.5-7B | 21.67% | 0.00% | 2.50% |
| Mistral-7B | 19.17% | 10.83% | 13.33% |
| Llama-3.1-8B | 17.50% | 8.33% | 10.00% |

Qwen's 21.7-percentage-point A–B loss remains supported after the paper's joint Holm correction. The other two losses do not pass that joint correction. Verification improves accuracy under the incorrect majority by only about 1.7–2.5 percentage points; no model has a statistically supported **specific-protection** effect after the specified correction. The paper uses task-level paired inference; the small reference code below reports point estimates only.

The paper compares three ways to control drift: structured prompts, post-processing probability correction, and lightweight hidden-layer intervention. With $\lambda=0.6$, post-processing makes drift 0.4 times the uncorrected value by construction; it should not be interpreted as an independently learned improvement. Stronger prompts do not produce consistent gains across models. Hidden-layer responses also vary by model, layer, and trigger setting.

<p align="center"><a href="figures/reliability_main.png"><img src="figures/reliability_main.png" width="1000" alt="System accuracy, accuracy loss, and verification effects with uncertainty intervals"></a></p>

<details>
<summary><strong>Distributional drift across models</strong></summary>

<p align="center"><a href="figures/herd_validation.png"><img src="figures/herd_validation.png" width="1000" alt="Signed and positive majority-alignment drift across models"></a></p>

Drift magnitude and prevalence differ across models. Doubao has aggregate point estimates only. A smaller drift measure alone does not establish improved decision accuracy.
</details>

<details>
<summary><strong>Incorrect consensus, retention, and abstention</strong></summary>

<p align="center"><a href="figures/reliability_auxiliary.png"><img src="figures/reliability_auxiliary.png" width="1000" alt="Incorrect consensus, correct-judgment retention, and abstention by condition"></a></p>

For Llama, verification reduces incorrect consensus but also lowers retention of initially correct judgments and increases abstention; accuracy recovers only slightly. Outcome measures must accompany behavioral drift.
</details>

## Reference code

This repository includes a **working, standard-library Python reference implementation**, not just an equation example:

| File | Function |
|:--|:--|
| [`mai/tasks.py`](mai/tasks.py) | Generate balanced, exactly solvable teaching tasks and verify their answer keys with rational arithmetic. |
| [`mai/runner.py`](mai/runner.py) | Build initial and A–D prompts, call a chat-completions endpoint, reuse initial responses across branches, and save raw responses. |
| [`mai/contextual.py`](mai/contextual.py) | Estimate response distributions, derive the initial majority, and evaluate fixed-reference two-stage drift. |
| [`mai/metrics.py`](mai/metrics.py) | Apply the paper's response, abstention, system vote, MAI, consensus, retention, and point-estimate rules. |
| [`mai/__main__.py`](mai/__main__.py) | Command-line entry point for task generation, model runs, and record analysis. |
| [`demo/mai_demo.py`](demo/mai_demo.py) | Run an offline synthetic trace through both measurement paths. |

Python **3.8+** is sufficient. No third-party package is required.

### Run the offline example

```bash
python -m demo.mai_demo
```

The trace illustrates a useful distinction: one agent's selected option stays **C**, while its MAI relative to **D** rises from **2.67 to 3.11**. The example then evaluates a five-agent A–D record. Its responses and outcomes are hand-written for teaching; they are **not observations from the study**.

To inspect the saved-record analysis path without a model endpoint:

```bash
python -m demo.mai_demo --save-record records.jsonl
python -m mai analyze records.jsonl
```

### Generate tasks and run a model

```bash
python -m mai generate --output tasks.json --variants 1
python -m mai run --tasks tasks.json --output records.jsonl \
  --base-url https://YOUR-TRUSTED-ENDPOINT/v1 --model YOUR-MODEL
python -m mai analyze records.jsonl
```

`--variants 1` writes **15 tasks**: one unique answer at each option position for each of the three task types. These are transparent teaching examples, not the paper's 60 evaluation tasks. `run` defaults to five agents, three samples per stage, two repetitions, temperature 0.7, and distinct seeds; options are configurable through `--help`. The endpoint must support an OpenAI-compatible `/chat/completions` request. Set `MAI_API_KEY` in the environment if the endpoint requires a bearer token. `MAI_BASE_URL` and `MAI_MODEL` can replace the corresponding command-line arguments. A local HTTP endpoint is supported; remote endpoints must use HTTPS.

Each JSONL line contains the task ID, answer key, preset wrong majority, raw initial and A–D responses, condition order, and generation settings. The analysis treats malformed replies as abstentions instead of silently repairing them. Keep raw records if you need to audit parsing or rerun the metrics. The analysis command pools completed task repetitions and reports **descriptive point estimates**, not confidence intervals or significance tests.

## Scope and reproducibility

The code makes the core protocol inspectable and lets users run their own models. It does **not** recreate the paper's original contextual task-construction agents, model-specific prior assignments, prompt-strength calibration, hidden-layer intervention, complete experimental task sets, or paired bootstrap/sign-flip inference. Consequently, running these teaching tasks will **not reproduce the published numbers**. Model endpoint behavior, decoding constraints, prompts, and task material can all change results. In the paper, Qwen used unconstrained generation; Mistral and Llama used constrained single-option decoding. This generic adapter uses unconstrained chat completions and strict parsing for every model.

The study's conclusions apply to the models, tasks, and protocols evaluated. In particular, the reliability experiment uses a preset incorrect majority, and baseline accuracy is low. It does not establish a quantitative or causal relationship between MAI drift and accuracy loss. The central practical lesson is to evaluate distributional movement **alongside** retention of initially correct judgments, incorrect consensus, abstention, and final system accuracy.
