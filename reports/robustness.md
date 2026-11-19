# Robustness results

Test set: 44 handwritten requests. Each variant is generated with a fixed seed.

| variant | base acc | base auto rate | base auto acc | robust acc | robust auto rate | robust auto acc |
|---|---|---|---|---|---|---|
| clean | 84% | 55% | 100% | 84% | 55% | 100% |
| reworded | 84% | 48% | 100% | 84% | 57% | 100% |
| informal | 86% | 52% | 96% | 86% | 55% | 100% |
| typos | 77% | 43% | 100% | 84% | 50% | 100% |
| multi_team | 98% | 32% | 100% | 98% | 32% | 100% |

Thresholds: base = 0.75, robust = 0.9

For multi_team, accuracy counts a prediction of either team as correct.
