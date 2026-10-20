# EvoHarnessBench EOG tool experiments

Local GPT-5 ReAct agent for the EOG tool track. The agent calls EvoHarnessBench MCP tools through the authenticated SDK, then saves a structured trace and verifier score. It accepts any published EOG domain; use calendar for the first smoke run.

## Setup

```sh
cd evoharnessbench-experiments/tool-track
uv sync
export EVAL_SERVICE_API_KEY=...  # set in your own terminal; never commit it
export OPENAI_API_KEY=...
uv run --no-sync eog-tools bootstrap-sdk
```

The SDK wheel is served by EvoHarnessBench behind authentication, so it is installed after `uv sync` rather than stored in `uv.lock`. The project requires SDK 0.19.2 or newer for independent cohort and harness selection. Re-run `bootstrap-sdk` after any `uv sync` that removes it. The command records the wheel version and SHA-256 in the ignored `.cache/sdk/sdk-info.json`. `EVAL_SERVICE_URL` can override the service URL used by the notebooks.

## Run one task

Find a calendar training task ID with the SDK:

```sh
uv run --no-sync python -c 'from simple_agentic_evals import EvalClient; print(EvalClient().task_ids("evovling_tools", "eog", 1, "train", "calendar", limit=1))'
uv run --no-sync eog-tools run-task --domain calendar --version 1 --split train --task-id TASK_ID
```

`run-task` saves `runs/<run-id>/result.json` with agent steps, tool calls and results, token usage, Score, strict Pass, and per-verifier feedback. The runner distinguishes agent or infrastructure errors from a valid zero score. It always requests the cumulative tool pool for the chosen stage.

For a limited held-out evaluation:

```sh
uv run --no-sync eog-tools run-deployment --domain calendar --limit 1
```

This uses the SDK's `run_benchmark` in deployment mode and saves `report.json` plus per-task traces. The default limit is one task per cohort, so its score is a smoke result. A full domain run requires `--full` and can be costly. `--max-steps`, `--max-completion-tokens`, and `--tool-result-chars` override the recorded defaults of 50, 16,384, and 4,000. The 16,384-token setting comes from the experiment configuration; it is not established by the public notebook example.

## Offline checks

```sh
uv run --no-sync python -m unittest discover -s tests -v
```

Run artifacts, cached wheels, and `.env` files are ignored by Git. The reference notebooks are outside this standalone repository in the parent directory.

## RELAI custom adapter

The adapter is in `.relai/custom-adaptor/eog-tools/`. From this repository,
use the RELAI CLI build containing custom adapter support, initialize RELAI,
and snapshot the SDK's EOG **training** task catalog:

```sh
relai init --bare
python3 .relai/custom-adaptor/eog-tools/generate_catalog.py
git add .
git commit -m "Configure RELAI EOG adapter and training catalog"
```

The generator reads every advertised tool domain and stage, checks the SDK's
training task counts, and records each task as
`<domain>/v<stage>/<sdk-task-id>`. It never adds test-split tasks. Each row also
has a shared domain tag and a unique task tag derived from its original cohort.
Later, `adapt(stage, tasks)` can map an SDK
task handle to this catalog using its stage and task ID; include the domain
whenever the same ID appears in more than one domain.
The generator joins those IDs to the public `ZixuanKe/evovling_tools` training
JSONL files at the pinned revision in `generate_catalog.py`. It includes the
full system and user prompts and one declared evaluator per published SQL
verifier, including the exact predicate. It excludes `gym_servers_config`,
which contains service tokens. Each grade returns one result per verifier;
their mean is the EOG task score. If the service key is unavailable but the
existing catalog contains an SDK ID snapshot, regenerate with
`python3 .relai/custom-adaptor/eog-tools/generate_catalog.py --from-existing-catalog`.
This mode checks the published rows against the saved IDs rather than a live
SDK listing.

Run one training task, a domain, or all cataloged training tasks:

```sh
relai simulate --custom-tests 'eog-tools::calendar/v1/TASK_ID' --result-json runs/relai-simulate.json
relai simulate --custom-tags 'eog-tools::domain.calendar'
relai optimize --custom-tags 'eog-tools::task.calendar.v1.TASK_ID' --no-pr --total-rollouts 6
```

The adapter's preparation command creates a `uv` environment in the current
checkout or optimizer worktree and installs or upgrades the authenticated SDK
when version 0.19.2 is absent. By default, an adapter task uses its cohort stage
as the tool-pool stage. For a direct RELAI run using a later tool pool, pass
`--env EOG_RELAI_HARNESS_STAGE=2`; the task ID still selects its original cohort.
Both API
keys must be available to the command, either in the environment or through
RELAI's runtime environment options. RELAI's optimizer also needs its own
credentials and a saved optimizer scope.

## Self-evolving CommandAgent run

Install the RELAI CLI build with `--skip-final-eval` and `--result-json`, then
run one EOG domain through the SDK's self-evolving protocol:

```sh
uv run --no-sync eog-tools run-self-evolving --domain calendar --limit 1 --total-rollouts 6
```

To give RELAI all training tasks seen through the current stage, add
`--training-pool cumulative`. The default, `current`, uses only that stage's
training tasks. RELAI evaluates every selected task with the current stage's
tool pool while preserving its original cohort and verifier. The exact tasks supplied by the SDK at each completed stage are
saved as catalog selectors in `runs/<run-id>/state/training-tasks.json`; a limited
run saves only its selected tasks. The rollout budget is shared across the
selected tasks. Each stage's `status.json` records selected IDs, counts by
cohort stage, and the tool-pool stage.

Use `--full` for the complete domain and `--matrix` to collect the continual
learning matrix. The SDK snapshots each stage's training tasks; the adaptation
command maps those exact IDs to the catalog and runs `relai optimize` on them.
The solve command runs in a fresh process for each scored task and writes a
trace, including the active Git commit, under the run's `command-agent/tasks/`
directory. The persistent detached checkout and stage handoff record live
under `runs/<run-id>/state/`. A zero-update stage keeps the prior version.
An already-running checkout keeps its original catalog and tags; new runs use
the updated catalog.
The checkout copies the local RELAI config and saved optimizer scope once;
optimizer memory starts empty for each benchmark run and persists across its
stages.

Optimization uses all requested rollouts for candidate search and acceptance;
the SDK's held-out evaluation supplies the stage score. The initial default is
six rollouts per stage and a one-task training/test limit for a small smoke run.
Timeouts are independent: `--optimize-timeout-minutes 720` limits RELAI within
each stage, `--adapt-timeout-minutes 750` limits the SDK adaptation command,
and `--task-timeout-minutes 30` limits each scored task. The adaptation timeout
must exceed the optimization timeout.

The SDK prints stage-level progress in the terminal and captures adaptation
stdout until completion. For live RELAI output, inspect
`runs/<run-id>/command-agent/adaptation/stage-<n>/optimizer.log`; for example,
`tail -f` that file. The same directory has `status.json`, refreshed every
minute while RELAI runs, and `optimizer-artifacts/<relai-run-id>/events.jsonl`.
`status.json` records the selected training tasks, optimizer PID, heartbeat,
latest event time, result path, outcome, and active agent commit. A heartbeat
shows the bridge is waiting for the optimizer; check the log and events to see
whether RELAI is making progress.
