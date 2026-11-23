# Robustness results

Test set: 44 handwritten requests. Each variant is generated with a fixed seed.

| variant | base acc | base auto rate | base auto acc | robust acc | robust auto rate | robust auto acc |
|---|---|---|---|---|---|---|
| clean | 93% | 73% | 100% | 95% | 68% | 100% |
| reworded | 86% | 80% | 94% | 91% | 73% | 97% |
| informal | 91% | 68% | 100% | 95% | 66% | 100% |
| typos | 89% | 57% | 100% | 89% | 59% | 96% |
| multi_team | 100% | 68% | 100% | 100% | 65% | 100% |

Thresholds: base = 0.65, robust = 0.85

For multi_team, accuracy counts a prediction of either team as correct.
