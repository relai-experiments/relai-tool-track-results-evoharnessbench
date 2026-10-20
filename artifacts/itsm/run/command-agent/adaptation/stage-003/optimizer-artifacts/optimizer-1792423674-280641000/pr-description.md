Accepted optimizer changes improved the agent on the evaluated environments.

## Change 1 of 1: Improve truncated tool-result recovery

**Summary**

- Added explicit truncation markers and read-only discovery recovery in `ReactAgent.run`, guiding pagination or non-overlapping partitions and preventing equivalent retries.
- Updated focused agent tests for marker visibility, ordering, escalation, mutation safety, and untruncated continuation.
- The adapter reaches this exact tool-result loop, so the behavior is consumed per run. Self-review matched the approved plan; tests were not run as instructed.

**Training environments**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20260115_034527_380_66392a82_a172d90d` |  | 0% | 0% | (+0%) |

**Regression controls**

| Environment | Description | Score Before | Score After | Diff |
| --- | --- | ---: | ---: | ---: |
| `task_20251221_231733_401_66392a82_514cea3c` |  | 75% | 100% | (+25%) |
