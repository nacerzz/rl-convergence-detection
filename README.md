# RL Convergence Detection

Detecting Reinforcement Learning Training Convergence Using Intrinsic Learning Signals.

## Project Idea

In reinforcement learning, training is often stopped after a fixed number of steps. However, this can waste computation if the agent has already converged earlier. This project investigates whether lightweight intrinsic signals can help detect when an RL agent has stopped learning.

We use PPO and log different training signals:

- evaluation return
- RND prediction error
- policy entropy
- visited states
- training steps

The main experiment compares fixed-budget training with an automatic stopping rule on `CartPole-v1`. We also include a small `Acrobot-v1` stress test to check whether the same signals behave similarly in a harder environment.

## Research Question

Can intrinsic learning signals provide a useful indication of whether an RL agent is still learning, plateauing, or has converged?

## Method

We compare two training settings:

1. **Fixed budget**: PPO is trained for the full training budget of 30,000 steps.
2. **Automatic stopping**: PPO is stopped early when the convergence detector indicates that learning has plateaued.

The automatic stopping rule uses three conditions:

- mean evaluation return is high enough
- mean RND prediction error is low
- policy entropy is stable over recent evaluations

For the CartPole-v1 experiment, the stopping rule uses:

- patience = 3 evaluations
- mean evaluation return >= 380
- mean RND prediction error <= 0.0002
- policy entropy change <= 0.04

## Main Experiment: CartPole-v1

In the first pilot experiment, the automatic stopping rule was too strict and did not trigger. Both fixed-budget and automatic-stopping runs used the full 30,000 training steps.

After inspecting the plots, we revised the stopping rule to combine evaluation return, RND prediction error, and policy entropy.

With the revised rule, automatic stopping triggered after about 8,000–14,000 training steps across three seeds, while the fixed-budget baseline always used 30,000 steps.

The automatic stopping rule reduced the average training budget from 30,000 steps to approximately 10,333 steps. This corresponds to a saving of about 19,667 training steps, or about 65.6%.

## Ablation Analysis

To better understand which signal drives the stopping behavior, we performed an offline ablation using the fixed-budget CartPole-v1 runs. We compared four stopping rules:

- return-only
- RND-only
- entropy-only
- combined rule

The ablation showed that the intrinsic signals alone stop much earlier than the combined rule. RND-only stopped after about 4,667 steps on average, and entropy-only stopped after about 3,667 steps on average. This suggests that RND prediction error and policy entropy should be treated as supporting proxy signals, not as direct proof of convergence.

Return-only stopping triggered after about 9,667 steps on average, close to the combined rule. The combined rule stopped after about 10,333 steps on average and is more conservative because it requires both sufficient evaluation performance and stabilization of intrinsic signals.

## Acrobot-v1 Stress Test

In addition to the main CartPole-v1 experiment, we ran a small Acrobot-v1 stress test. Acrobot-v1 is a harder control task with a continuous observation space.

In this environment, the automatic stopping rule did not trigger within 30,000 steps. The evaluation return remained noisy and unstable, while RND prediction error and policy entropy did not provide a clear convergence point.

This supports the interpretation that RND prediction error and policy entropy are proxy signals, not direct convergence measures. While the combined rule worked well on CartPole-v1, it does not directly generalize to Acrobot-v1 without environment-specific tuning.

## Interpretation

The CartPole-v1 results show that the evaluation return increases quickly during early training and reaches a high level around the stopping point.

The RND prediction error drops strongly at the beginning and then becomes almost flat. This suggests that the agent first encounters novel states and later visits a more stable state distribution.

The policy entropy decreases during training, indicating that the policy becomes more deterministic and explores less.

However, the Acrobot-v1 stress test shows that these signals do not always provide a reliable convergence point in harder environments. Therefore, intrinsic signals should not be interpreted as direct convergence measures by themselves. They are useful supporting signals, especially when combined with evaluation performance.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the CartPole-v1 experiment:

```bash
python experiments/run_experiment.py
```

Generate CartPole-v1 plots:

```bash
python analysis/plots.py
```

Run convergence summary:

```bash
python analysis/convergence_analysis.py
```

Run ablation analysis:

```bash
python analysis/ablation_analysis.py
```

Run the Acrobot-v1 stress test:

```bash
python experiments/run_acrobot.py
```

Generate Acrobot-v1 plots:

```bash
python analysis/plots_acrobot.py
```

Run tests:

```bash
python -m pytest tests
```

## Output Files

CartPole-v1 experiment results are saved in:

```text
results/cartpole/
```

CartPole-v1 plots and analysis summaries are saved in:

```text
figures/cartpole/
```

Acrobot-v1 experiment results are saved in:

```text
results/acrobot/
```

Acrobot-v1 plots are saved in:

```text
figures/acrobot/
```

Important CartPole-v1 output files:

- `evaluation_return.png`
- `rnd_prediction_error.png`
- `policy_entropy.png`
- `training_steps_comparison.png`
- `convergence_summary.csv`
- `ablation_summary.csv`
- `ablation_stopping_steps.csv`

Important Acrobot-v1 output files:

- `evaluation_return.png`
- `rnd_prediction_error.png`
- `policy_entropy.png`
- `visited_states.png`
- `combined_acrobot_results.csv`

## Project Structure

```text
experiments/run_experiment.py
```

Main CartPole-v1 experiment script. Runs PPO with fixed-budget training and automatic stopping.

```text
experiments/run_acrobot.py
```

Small Acrobot-v1 stress test.

```text
src/convergence_rl/
```

Contains the main project code, including environment creation, intrinsic signals, stopping rules, evaluation, and logging.

```text
analysis/
```

Contains scripts for generating plots, convergence summaries, and ablation analysis.

```text
results/
```

Contains the raw CSV result files.

```text
figures/
```

Contains generated plots and summary CSV files.

```text
tests/
```

Contains simple tests for the RND signal and convergence stopping rule.

## Limitation

The main successful result is shown on CartPole-v1, which is a relatively simple and dense-reward environment. In such an environment, evaluation return, RND prediction error, and policy entropy may move together because the task is easy.

The Acrobot-v1 stress test shows that this relation is not guaranteed in a harder environment. The stopping rule did not trigger there within 30,000 steps, and the signals were less clearly aligned with performance. More environments and additional tuning would be needed to evaluate whether the stopping signal generalizes more broadly.