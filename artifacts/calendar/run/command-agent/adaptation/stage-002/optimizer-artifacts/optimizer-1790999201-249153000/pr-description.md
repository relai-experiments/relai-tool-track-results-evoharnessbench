Accepted optimizer changes improved the agent on the evaluated environments.

## Change 1 of 2: Add fallback for yearless date searches

**Summary**

Updated `EXECUTION_GUIDANCE` and `FINALIZATION_CHECKPOINT` to require one bounded alternate search for empty, yearless-date retrievals while preserving explicit years and mutation safeguards. The custom adapter creates a fresh `ReactAgent`, whose `run()` injects both prompts, so the replay consumes this change directly.

Updated prompt assertions in `tests/test_agent.py`. Self-review matched the approved scope; tests were not run per instruction.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251201_191924_271_e81f6083_21fbee85` |  | 45% | 55% | (+9%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20260109_043327_590_806f1a83_d690097e` |  | 100% | 100% | (+0%) |

## Change 2 of 2: Reconcile ambiguous writes before retrying

**Summary**

- Added generic ambiguous-write reconciliation and lifecycle-field copy guidance in `ReactAgent.run`.
- The runner invokes this loop directly, so MCP retries are gated until a relevant read confirms persisted state.
- Updated focused agent-loop tests. Self-review matched the approved plan; tests were not run as instructed.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251201_191924_271_e81f6083_21fbee85` |  | 73% | 100% | (+27%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20260106_132019_309_bca89e7d_5ae689d9` |  | 91% | 91% | (+0%) |
