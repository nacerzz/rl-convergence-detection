# RL Convergence Detection

Detecting Reinforcement Learning Training Convergence Using Intrinsic Learning Signals.

## Project Idea

In reinforcement learning, training is often stopped after a fixed number of steps. However, this can waste computation if the agent has already converged earlier. This project investigates whether lightweight intrinsic signals can help detect when an RL agent has stopped learning.

We use PPO on `CartPole-v1` and log different training signals:

- evaluation return
- RND prediction error
- policy entropy
- visited states
- training steps

The goal is to compare fixed-budget training with an automatic stopping rule.

## Research Question

Can intrinsic learning signals provide a useful indication of whether an RL agent is still learning, plateauing, or has converged?

## Method

We compare two training settings:

1. **Fixed budget**: PPO is trained for the full training budget of 30,000 steps.
2. **Automatic stopping**: PPO is stopped early when the convergence detector indicates that learning has plateaued.

The automatic stopping rule uses three conditions:

- mean evaluation return is at least 380
- mean RND prediction error is below 0.0002
- policy entropy is stable over recent evaluations

The stopping rule uses a patience of 3 evaluations.

## Results

In the first pilot experiment, the automatic stopping rule was too strict and did not trigger. Both fixed-budget and automatic-stopping runs used the full 30,000 training steps.

After inspecting the plots, we revised the stopping rule to combine evaluation return, RND prediction error, and policy entropy.

With the revised rule, automatic stopping triggered after about 8,000–14,000 training steps across three seeds, while the fixed-budget baseline always used 30,000 steps.

The automatic stopping rule reduced the average training budget from 30,000 steps to approximately 10,333 steps. This corresponds to a saving of about 19,667 training steps, or about 65.6%.

## Ablation Analysis

To better understand which signal drives the stopping behavior, we also performed an offline ablation using the fixed-budget runs. We compared four stopping rules:

- return-only
- RND-only
- entropy-only
- combined rule

The ablation showed that the intrinsic signals alone stop much earlier than the combined rule. RND-only stopped after about 4,667 steps on average, and entropy-only stopped after about 3,667 steps on average. This suggests that RND prediction error and policy entropy should be treated as supporting proxy signals, not as direct proof of convergence.

Return-only stopping triggered after about 9,667 steps on average, close to the combined rule. The combined rule stopped after about 10,333 steps on average and is more conservative because it requires both sufficient evaluation performance and stabilization of intrinsic signals.

## Interpretation

The evaluation return increases quickly during early training and reaches a high level around the stopping point.

The RND prediction error drops strongly at the beginning and then becomes almost flat. This suggests that the agent first encounters novel states and later visits a more stable state distribution.

The policy entropy decreases during training, indicating that the policy becomes more deterministic and explores less.

Together, these results suggest that RND prediction error and policy entropy can provide useful supporting information about training progress. However, they should not be interpreted as direct convergence measures by themselves. The combined stopping rule is safer because it also requires sufficient evaluation performance.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python experiments/run_experiment.py
```

Generate plots:

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

Run tests:

```bash
python -m pytest tests
```

## Output Files

Experiment results are saved in:

```text
results/cartpole/
```

Plots and analysis summaries are saved in:

```text
figures/cartpole/
```

Important output files:

- `evaluation_return.png`
- `rnd_prediction_error.png`
- `policy_entropy.png`
- `training_steps_comparison.png`
- `convergence_summary.csv`
- `ablation_summary.csv`
- `ablation_stopping_steps.csv`

## Project Structure

```text
experiments/run_experiment.py
```

Main experiment script. Runs PPO with fixed-budget training and automatic stopping.

```text
src/convergence_rl/
```

Contains the main project code, including environment creation, intrinsic signals, stopping rules, evaluation, and logging.

```text
analysis/
```

Contains scripts for generating plots, convergence summaries, and ablation analysis.

```text
results/cartpole/
```

Contains the raw CSV result files.

```text
figures/cartpole/
```

Contains generated plots and summary CSV files.

```text
tests/
```

Contains simple tests for the RND signal and convergence stopping rule.

## Limitation

So far, the final experiment focuses on `CartPole-v1`. This environment is useful as a controlled proof of concept, but it is also relatively simple and dense-reward. Therefore, the observed relationship between evaluation return, RND prediction error, and policy entropy may not directly generalize to more difficult environments.

More environments, such as `Acrobot-v1`, `LunarLander-v3`, or MiniGrid tasks, would be needed to test whether the stopping signal generalizes beyond CartPole-v1.