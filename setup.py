#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='fastNLP',
    version='0.0.1',
    description='fastNLP: Deep Learning Toolkit for NLP, developed by Fudan FastNLP Team',
    long_description=readme,
    license=license,
    author='fudanNLP',
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=reqs.strip().split('\n'),
)
