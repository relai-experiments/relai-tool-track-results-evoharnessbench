Accepted optimizer changes improved the agent on the evaluated environments.

## Change 1 of 2: Strengthen discovery and objective verification

**Summary**

- Strengthened `ReactAgent` prompts for state-independent discovery, exact identifier validation, prerequisite transitions, and full objective verification.
- Updated prompt-wiring tests accordingly.
- This is consumed by `runner.py → ReactAgent.run`, where prompts enter every replay. Self-review matched the approved plan; tests/replay were not run per instruction.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251202_041242_095_99ba2325_a63db5e8` |  | 50% | 100% | (+50%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251212_081413_439_66392a82_7eb34f57` |  | 25% | 25% | (+0%) |

## Change 2 of 2: Prioritize explicit objectives and constraints

**Summary**

Updated `src/eog_tools/agent.py` to prioritize explicit objectives, authorized prerequisites, authoritative prohibitions, and exact targets during planning and verification. This prompt is consumed directly by `ReactAgent.run` on both runner and command-bridge paths.

Updated focused prompt-assembly tests. Self-review matched the approved scope; tests were not run per instruction.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251202_041242_095_99ba2325_a63db5e8` |  | 100% | 100% | (+0%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251212_081413_439_66392a82_7eb34f57` |  | 25% | 25% | (+0%) |
