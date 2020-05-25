from time import sleep
from enum import Enum
import logging
from typing import Optional
from pathlib import Path

from unityagents import UnityEnvironment
import typer
import torch

from scripts.definitions import ROOT_DIR
from banana_collector.agent import RandomAgent, DqnAgent

DEFAULT_ENVIRONMENT_EXECUTABLE_PATH = str(ROOT_DIR / 'unity_banana_environment/Banana.x86_64')
DEVICE = torch.device('cpu')


class AgentType(str, Enum):
    random = 'random'
    dqn = 'dqn'


def run_environment(
        environment_path: str = DEFAULT_ENVIRONMENT_EXECUTABLE_PATH,
        agent_type: AgentType = AgentType.dqn,
        agent_parameters_path: Optional[Path] = None
):
    """Run the banana environment and visualize the actions an agent.

    Args:
        environment_path: path of the executable which runs the banana environment
        agent_type: the type of the agent. Can either be a dummy rangdom agent or an agent using a DQN
        agent_parameters_path: an optional path to load the agent parameters from
    """
    env = UnityEnvironment(file_name=environment_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])

    if agent_type is AgentType.random:
        agent = RandomAgent(state_size, action_size)
        logging.info(f'Loaded {agent} agent.')
    else:
        agent = DqnAgent(state_size=state_size, action_size=action_size, device=DEVICE)
        if agent_parameters_path:
            agent.load(agent_parameters_path)
        logging.info(f'Loaded {agent} agent.')

    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        sleep(0.02)
        if done:
            break

    print("Score: {}".format(score))

    env.close()


if __name__ == '__main__':
    typer.run(run_environment)
