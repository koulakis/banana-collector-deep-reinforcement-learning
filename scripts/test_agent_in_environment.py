from time import sleep

from unityagents import UnityEnvironment
import typer

from scripts.definitions import ROOT_DIR
from banana_collector.agent import RandomAgent

ENVIRONMENT_EXECUTABLE_PATH = str(ROOT_DIR / 'unity_banana_environment/Banana.x86_64')


def run_environment():
    print(ENVIRONMENT_EXECUTABLE_PATH)
    env = UnityEnvironment(file_name=ENVIRONMENT_EXECUTABLE_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])

    agent = RandomAgent(state_size, action_size)

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
