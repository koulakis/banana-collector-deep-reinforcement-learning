from typing import Optional

import typer
import torch
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import pandas as pd

from scripts.definitions import ROOT_DIR
from banana_collector.agent import DqnAgent
from banana_collector.train import train_agent

ENVIRONMENT_EXECUTABLE_PATH = str(ROOT_DIR / 'unity_banana_environment/Banana.x86_64')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
EXPERIMENTS_DIR = ROOT_DIR / 'experiments'


def train(
    experiment_name: str = 'dqn',
    device: str = DEVICE,
    environment_path: str = ENVIRONMENT_EXECUTABLE_PATH,
    number_of_episodes: int = 2000,
    maximum_timestaps: int = 1000,
    initial_epsilon: float = 1.0,
    final_epsilon: float = 0.01,
    epsilon_decay: float = 0.995,
    hidden_layers: Optional[str] = None
):
    """Train an agent and save its parameters along with training artifacts."""
    device = torch.device(device)
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    agent_path = experiment_dir / 'checkpoint.pth'
    performance_path = experiment_dir / 'performance.png'
    score_path = experiment_dir / 'scores.csv'

    env = UnityEnvironment(file_name=environment_path, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])

    if hidden_layers:
        hidden_layers = list(map(int, hidden_layers.split(',')))
    agent = DqnAgent(state_size=state_size, action_size=action_size, device=device, hidden_layers=hidden_layers)

    scores = train_agent(
        agent,
        env,
        brain_name,
        agent_path,
        number_of_episodes=number_of_episodes,
        maximum_timestaps=maximum_timestaps,
        initial_epsilon=initial_epsilon,
        final_epsilon=final_epsilon,
        epsilon_decay=epsilon_decay
    )

    (pd.DataFrame({'score': scores})
     .reset_index()
     .rename(columns={'index': 'episode'})
     .to_csv(score_path, index=False))

    plt.plot(range(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(performance_path)


if __name__ == '__main__':
    typer.run(train)
