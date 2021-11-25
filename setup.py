from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="asynch_rl",
    version="0.0.1",
    author="Enrico Regolin",
    author_email="enrico.regolin@gmail.com",
    description="Asynchronous Reinforcement Learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EnricoReg/asynch-rl",
    packages= find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
