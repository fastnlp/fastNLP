#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='fastNLP',
    version='0.1.1',
    description='fastNLP: Deep Learning Toolkit for NLP, developed by Fudan FastNLP Team',
    long_description=readme,
    license=license,
    author='fudanNLP',
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=reqs.strip().split('\n'),
)
