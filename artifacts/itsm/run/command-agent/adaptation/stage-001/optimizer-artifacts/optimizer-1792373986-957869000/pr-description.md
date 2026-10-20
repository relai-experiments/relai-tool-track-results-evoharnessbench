Accepted optimizer changes improved the agent on the evaluated environments.

## Change 1 of 3: Add persistent mutation-fidelity guidance

**Summary**

Added generic mutation-fidelity guidance to ReactAgent’s persistent system context before the user message. The runner calls ReactAgent.run, and every ReAct iteration reuses this context.

Updated prompt-assembly assertions to preserve prompt ordering and verify guidance delivery. Self-review matched the approved scope; tests were not run per instruction.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251218_143555_095_97d7df7d_ece487bd` |  | 100% | 100% | (+0%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251218_055646_898_97d7df7d_be7218cc` |  | 50% | 50% | (+0%) |

## Change 2 of 3: Require mutation postcondition verification

**Summary**

Strengthened mutation guidance in `ReactAgent.run` to derive, apply, and verify policy-required postconditions before completion. The runner constructs/uses `ReactAgent`, which appends this guidance to the environment policy before the first model call.

Updated `tests/test_agent.py` to assert the full protocol and prompt placement. Self-review matched the approved scope; tests were not run per instruction.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251218_193035_592_391f1eff_5293f595` |  | 0% | 100% | (+100%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20260115_201951_323_391f1eff_659b9dae` |  | 0% | 0% | (+0%) |

## Change 3 of 3: Clarify lookup and requester-role handling

**Summary**

- Added lookup-first ambiguity and requester-role guidance in `ReactAgent.run`’s appended system prompt.
- This prompt is assembled on every simulator/runtime call, including cached-agent paths, so the model consumes it before acting.
- Updated prompt-composition coverage. Self-review matched the approved scope; tests were not run as instructed.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251221_222834_884_97d7df7d_c088062a` |  | 67% | 67% | (+0%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251219_081545_939_99ba2325_887f5580` |  | 100% | 100% | (+0%) |
