# RL Convergence Detection

Detecting Reinforcement Learning Training Convergence Using Intrinsic Learning Signals.

## Project Idea

In reinforcement learning, training is often stopped after a fixed number of steps. However, this can waste computation if the agent has already converged earlier. This project investigates whether lightweight intrinsic signals can help detect when an RL agent has stopped learning.

We use PPO on `CartPole-v1` and log different training signals:

- evaluation return
- RND prediction error
- policy entropy
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

With the revised rule, automatic stopping triggered after about 8,000–9,000 training steps, while the fixed-budget baseline still used 30,000 steps. This saves roughly 70% of training steps on CartPole-v1 without an obvious loss in performance.

## Interpretation

The evaluation return increases quickly during early training and reaches a high level around the stopping point.

The RND prediction error drops strongly at the beginning and then becomes almost flat. This suggests that the agent first encounters novel states and later visits a more stable state distribution.

The policy entropy decreases during training, indicating that the policy becomes more deterministic and explores less.

Together, these signals suggest that intrinsic learning signals can help detect training convergence.

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

## Output Files

Experiment results are saved in:

```text
results/cartpole/
```

Plots are saved in:

```text
figures/cartpole/
```

Important plots:

- `evaluation_return.png`
- `rnd_prediction_error.png`
- `policy_entropy.png`
- `training_steps_comparison.png`

## Limitation

So far, the experiment was tested on CartPole-v1 only. More environments, such as LunarLander or MiniGrid, would be needed to evaluate whether the stopping signal generalizes.