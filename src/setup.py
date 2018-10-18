'''
Created on October 08, 2018

@author: Alejandro Molina
'''

import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spflow",
    version="0.0.7",
    author="Alejandro Molina et al.",
    author_email="alejomc@gmail.com",
    description="Sum Product Flow: An Easy and Extensible Library for Sum-Product Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejandromolinaml/SPFlow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'sklearn', 'statsmodels', 'networkx', 'joblib', 'matplotlib', 'pydot',
                      'lark-parser', 'tqdm'],
)
