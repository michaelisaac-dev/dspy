# Aggregated results


## HeartDisease (n_test=152)

Paper matrix: 21 cells; best = 70.4 (Predict + BootstrapFewShotRandomSearch); baseline Predict = 57.24

| arm | seeds | accuracy | test LM calls | test cost/1k |
|---|---|---|---|---|
| Predict (unoptimized) | 1 | 59.2 | 152 | $0.10 |
| CoT (unoptimized) | 1 | 47.4 | 152 | $0.21 |
| Flex identity (unoptimized) | 1 | 59.2 | 152 | $0.10 |
| Prompt space: GEPA (CoT) | 3 | 75.2 ± 4.9 | 152 | $0.26 |
| Flex: GEPA-compiled | 3 | 78.1 ± 8.4 | 0 | $0.00 |

## Iris (official split) (n_test=75)

Paper matrix: 38 cells; best = 92.0 (Predict + BootstrapFewShot); baseline Predict = 58.67

| arm | seeds | accuracy | test LM calls | test cost/1k |
|---|---|---|---|---|
| Predict (unoptimized) | 1 | 60.0 | 75 | $0.05 |
| CoT (unoptimized) | 1 | 56.0 | 75 | $0.12 |
| Flex identity (unoptimized) | 1 | 60.0 | 75 | $0.05 |
| Prompt space: GEPA (CoT) | 1 | 56.0 | 75 | $0.12 |
| Flex: GEPA-compiled | 1 | 60.0 | 75 | $0.05 |

## Iris (fixed split) (n_test=75)

| arm | seeds | accuracy | test LM calls | test cost/1k |
|---|---|---|---|---|
| Predict (unoptimized) | 1 | 80.0 | 75 | $0.05 |
| CoT (unoptimized) | 1 | 80.0 | 75 | $0.11 |
| Flex identity (unoptimized) | 1 | 80.0 | 75 | $0.05 |
| Prompt space: GEPA (CoT) | 1 | 96.0 | 75 | $0.17 |
| Flex: GEPA-compiled | 1 | 96.0 | 0 | $0.00 |

## Scone (n_test=500)

Paper matrix: 28 cells; best = 93.0 (GeneratorCriticRanker + MIPROv2); baseline Predict = 79.0

| arm | seeds | accuracy | test LM calls | test cost/1k |
|---|---|---|---|---|
| Predict (unoptimized) | 1 | 79.6 | 500 | $0.04 |
| CoT (unoptimized) | 1 | 89.0 | 500 | $0.09 |
| Flex identity (unoptimized) | 1 | 79.6 | 500 | $0.04 |
| Prompt space: GEPA (CoT) | 1 | 86.8 | 500 | $0.17 |
| Flex: GEPA-compiled | 2 | 87.1 ± 3.0 | 500 | $0.17 |
