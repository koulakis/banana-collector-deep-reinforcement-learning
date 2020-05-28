from pathlib import Path
import logging

import torch
import typer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from scripts.definitions import ROOT_DIR
from banana_collector.agent import DqnAgent

DEFAULT_ENVIRONMENT_EXECUTABLE_PATH = str(ROOT_DIR / 'unity_banana_environment/Banana.x86_64')
DEVICE = torch.device('cpu')


def evaluate(
    agent_dir: Path,
    number_of_episodes: int = 1000,
    maximum_timestaps: int = 1000,
    environment_path: str = DEFAULT_ENVIRONMENT_EXECUTABLE_PATH
):
    agent_path = agent_dir / 'checkpoint.pth'
    if not agent_path.exists():
        logging.warning(f'No saved parameters found for agent in {agent_dir}.')
        return
    hist_path = agent_dir / 'evaluation_histogram.png'
    scores_path = agent_dir / 'scores_evaluation.csv'

    env = UnityEnvironment(file_name=environment_path, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])

    agent = DqnAgent(state_size=state_size, action_size=action_size, device=DEVICE)
    agent.load(agent_path)

    scores = []

    for _ in tqdm(list(range(1, number_of_episodes + 1))):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(maximum_timestaps):
            action = agent.act(state, epsilon=0.0)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            state = next_state
            score += reward
            if done:
                break
        scores.append(score)

    scores_ts = pd.Series(scores)

    plt.hist(scores, bins=100, color='steelblue')
    xlim = plt.ylim()
    med = scores_ts.median()
    plt.vlines(med, *xlim, linewidth=2, linestyle='--', color='orange', label=f'median: {med}')
    plt.legend()

    plt.savefig(hist_path)
    scores_ts.to_csv(scores_path, index=False)


if __name__ == '__main__':
    typer.run(evaluate)
