#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()

pkgs = [p for p in find_packages() if p.startswith('fastNLP')]
print(pkgs)

setup(
    name='FastNLP',
    version='0.7.1',
    url='https://gitee.com/fastnlp/fastNLP',
    description='fastNLP: Deep Learning Toolkit for NLP, developed by Fudan FastNLP Team',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache License',
    author='Fudan FastNLP Team',
    python_requires='>=3.6',
    packages=pkgs,
    install_requires=reqs.strip().split('\n'),
)
