#python scripts/train_agent.py \
#  --experiment-name final_comparison_dqn \
#  --no-dueling-dqn --no-double-dqn --no-prioritize-replay \
#  --number-of-episodes 4000
#
#python scripts/train_agent.py \
#  --experiment-name final_comparison_dueling_dqn \
#  --dueling-dqn --no-double-dqn --no-prioritize-replay \
#  --number-of-episodes 4000

python scripts/train_agent.py \
  --experiment-name final_comparison_dueling_double_dqn \
  --dueling-dqn --double-dqn --no-prioritize-replay \
  --number-of-episodes 4000

python scripts/train_agent.py \
  --experiment-name final_comparison_dueling_double_per_dqn \
  --dueling-dqn --double-dqn --prioritize-replay \
  --number-of-episodes 4000
