ls -d1 experiments/dqn_per* | xargs -I {} python scripts/evaluate_agent.py {}
