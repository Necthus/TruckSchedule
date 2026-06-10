# Test Set2 November Metrics

- Date range: `2023-11-01 ~ 2023-11-15`
- Dispatch method: `Fastest`
- Mode: evaluation only, no training, no checkpoint saving
- RL checkpoints: existing `best_checkpoint.json` in each method directory

| method | loaded checkpoint | cost RMB | vs Urgent | fuel | overtime | discontinuity | station OT | project OT | go km | return km | delivered m3 | cost/revenue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ScratchPerceptionOnlyRL` | 19 | 569,223.05 | -38,130.21 | 459,893.88 | 24,875.50 | 10,000.00 | 51,889.00 | 22,564.67 | 63,210.57 | 86,176.07 | 55,511.50 | 2.93% |
| `ScratchDispatchAwareNoShapingRL` | 199 | 576,630.26 | -30,723.00 | 439,698.59 | 21,533.00 | 10,000.00 | 87,926.00 | 17,472.67 | 63,307.45 | 77,591.96 | 55,511.50 | 2.97% |
| `ScratchRL` | 99 | 578,443.40 | -28,909.86 | 448,434.90 | 22,285.50 | 10,000.00 | 75,918.33 | 21,804.67 | 63,105.77 | 81,575.07 | 55,511.50 | 2.98% |
| `ScratchDispatchAwareRL` | 99 | 591,277.15 | -16,076.11 | 464,896.48 | 21,603.00 | 15,000.00 | 71,241.00 | 18,536.67 | 63,277.99 | 88,136.83 | 55,511.50 | 3.04% |
| `ScratchCostOnlyRL` | 19 | 606,904.28 | -448.98 | 450,388.78 | 20,472.50 | 30,000.00 | 88,018.67 | 18,024.33 | 63,302.64 | 82,052.00 | 55,511.50 | 3.12% |
| `Urgent` | None | 607,353.26 | +0.00 | 443,505.09 | 45,127.50 | 50,000.00 | 42,245.00 | 26,475.67 | 63,306.95 | 79,184.86 | 55,511.50 | 3.13% |
| `ScratchDispatchAwareNoCostRL` | 99 | 684,308.34 | +76,955.09 | 477,332.51 | 83,251.50 | 45,000.00 | 59,711.67 | 19,012.67 | 63,399.82 | 93,122.93 | 55,511.50 | 3.52% |

Best method on test set2: `ScratchPerceptionOnlyRL`.
Best cost: `569,223.05 RMB`.

Artifacts:

- `run/test_set2_november/metrics.json`
- `run/test_set2_november/metrics.csv`
