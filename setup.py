#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 01:05:17 2024

@author: tungchentsai
"""

from setuptools import setup, find_packages


with open(r'./README.md') as file:
    long_description = file.read()

with open(r'./LICENSE') as file:
    license = file.read()


setup(
    name='tkinter-extensions',
    version='0.1.0',
    author='Tung-Chen Tsai',
    author_email='tungchentsai1753@gmail.com',
    description='Some tkinter extensions that allow you to build GUI apps with '
                'modern UI/UX design concepts.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license=license,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    url='https://github.com/TsaiTung-Chen/tkinter-extensions',
    packages=find_packages(include=[
        'tkinter_extensions',
        'tkinter_extensions.*'
    ]),
    install_requires=['packaging', 'ttkbootstrap<1.11', 'numpy>=2.0'],
    python_requires='>=3.12',
    package_data={'': ['*.json']}
)

