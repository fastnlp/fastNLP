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
    name='FastNLP',
    version='dev0.5.0',
    description='fastNLP: Deep Learning Toolkit for NLP, developed by Fudan FastNLP Team',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache License',
    author='FudanNLP',
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=reqs.strip().split('\n'),
)
