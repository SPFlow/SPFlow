"""
Created on October 08, 2018

@author: Alejandro Molina
"""

from os import path
import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

with open("../requirements.txt", "r") as fh:
    requirements = fh.readlines()

# Get __version__ from _meta.py. This is a trick to avoid importing the package
#   from ``setup.py``, which can have unintended side-effects.
with open(path.join("spn", "_meta.py")) as f:
    exec(f.read())

setuptools.setup(
    name="spflow",
    version=__version__,
    author="Alejandro Molina et al.",
    author_email="molina@cs.tu-darmstadt.de",
    description="Sum Product Flow: An Easy and Extensible Library for Sum-Product Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejandromolinaml/SPFlow",
    packages=setuptools.find_packages(),
    package_data={"spn.algorithms.splitting": ["*.R"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
)
