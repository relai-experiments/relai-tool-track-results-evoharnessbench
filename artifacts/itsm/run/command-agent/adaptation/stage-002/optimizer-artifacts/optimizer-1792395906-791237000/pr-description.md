Accepted optimizer changes improved the agent on the evaluated environments.

## Change 1 of 2: Recover from empty discovery searches

**Summary**

- Added schema-aware empty-search recovery inside `ReactAgent.run`, immediately after MCP tool results and before the next model completion.
- Raw metadata identifies read-only discovery tools; mutation tools are excluded. Repeated empty searches trigger progressively stronger broadening guidance.
- Updated focused agent tests for recovery, evidence preservation, escalation, and exclusions. The custom adapter constructs this agent per run, so the path consumes the change.
- Self-review matched the approved plan. Tests were not run, per instruction.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20260115_064627_351_66392a82_cd4009dd` |  | 0% | 100% | (+100%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251222_084449_326_66392a82_cadddb04` |  | 0% | 0% | (+0%) |

## Change 2 of 2: Enforce conservative mutation scope

**Summary**

- Strengthened `MUTATION_FIDELITY_GUIDANCE` with a closed-world scope ledger and conservative mutation rules.
- `ReactAgent.run` injects this guidance into every fresh system message; the grounded transcript confirms this exercised path.
- Updated system-guidance assertions in `tests/test_agent.py`. Self-review matched the approved plan; tests were not run per instruction.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20260114_085016_654_e6427839_ca9119ec` |  | 67% | 100% | (+33%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251221_191123_397_97d7df7d_7196ef19` |  | 33% | 33% | (+0%) |
