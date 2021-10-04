from setuptools import setup, find_packages

setup(name='dynamic_grid',
      version='0.0.1',
      install_requires=['gym', 'numpy'],
      packages=find_packages(),
      author="Ese Ben-Iwhiwhu, Jeff Dick, Anand Samra, Andrea Soltoggio",
      author_email="e.ben-iwhiwhu@lboro.ac.uk",
      description="This is an implementation of a configurable square grid world environment (tasks can be defined based on changes in the reward structure, input space or transition dynamics).",
      license="Copyright (c) 2021 Ese Ben-Iwhiwhu. MIT License",
      keywords="Deep reinforcement learning, dynamic goals/rewards, meta learning, adaptation, continual learning, multi-task learning",
      url="",
      project_urls={
        "Bug Tracker": "",
        "Documentation": "",
        "Source Code": "",
    }
)
