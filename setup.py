import setuptools
from setuptools.command.install import install
import subprocess

UNITY_AGENTS_PATH = 'udacity_custom_unity_agents/'


class InstallUdacityCustomUnityAgents(install):
    """Install the agents defined by udacity before setting up the package."""
    def run(self):
        subprocess.run(f'pip install {UNITY_AGENTS_PATH}'.split(' '))
        install.run(self)


setuptools.setup(
    name="banana-collector",
    version="0.0.1",
    author="Marios Koulakis",
    description="This is a solution for the first project of the Udacity deep reinforcement learning course.",
    packages=['banana_collector', 'scripts'],
    install_requires=[
        'tensorflow==1.7.1',
        'mlagents',
        'numpy',
        'typer'
    ],
    cmdclass={
        'install': InstallUdacityCustomUnityAgents
    },
    extras_require={
        'dev': [
            'jupyterlab',
            'pytest-timeout',
            'pytest-benchmark',
            'flake8'
        ]
    },
    python_requires='~=3.6'
)
