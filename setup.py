import setuptools
from setuptools.command.install import install
from subprocess import check_call

UNITY_AGENTS_PATH = 'udacity_custom_unity_agents/'


class InstallUdacityCustomUnityAgents(install):
    """Install the agents defined by udacity before setting up the package."""
    def run(self):
        check_call(f'pip install {UNITY_AGENTS_PATH}'.split(' '))
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
            'jupyterlab'
        ]
    },
    python_requires='~=3.6'
)
