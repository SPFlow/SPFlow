from os import path
import setuptools

with open("../requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="spflow",
    version="0.0.0",
    description="Sum Product Flow: An Easy and Extensible Library for Sum-Product Networks",
    url="https://github.com/SPFlow/SPFlow",
    install_requires=requirements
)
