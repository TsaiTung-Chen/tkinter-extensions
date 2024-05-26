#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 01:05:17 2024

@author: tungchentsai
"""

import json
from setuptools import setup, find_packages


with open(r"./tkinter_extensions/metadata.json") as json_file:
    metadata = json.load(json_file)

with open(r"./README.md") as file:
    long_description = file.read()

with open(r"./LICENSE") as file:
    license = file.read()


if __name__ == "__main__":
    setup(name=metadata["__name__"],
          version=metadata["__version__"],
          author=metadata["__author__"],
          author_email=metadata["__author_email__"],
          description=metadata["__description__"],
          long_description=long_description,
          long_description_content_type='text/markdown',
          classifiers=metadata["__classifiers__"],
          url=metadata["__url__"],
          packages=find_packages(include=['tkinter_extensions',
                                          'tkinter_extensions.*']),
          install_requires=metadata["__install_requires__"],
          python_requires=metadata["__python_requires__"],
          package_data={"": [r"*.json"]},
          license=license
    )

