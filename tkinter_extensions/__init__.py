#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
@repository: https://github.com/TsaiTung-Chen/tkinter-extensions
"""

import sys
from importlib.metadata import version, PackageNotFoundError

from tkinter_extensions import constants, dialogs, widgets

if sys.version_info < (3, 10):
    raise ImportError(
        "tkinter_extensions requires Python 3.10 or higher. "
        "Please upgrade your Python version."
    )


__version__: str
try:
    __version__ = version('tkinter_extensions')
except PackageNotFoundError:
    __version__ = ''

__all__: list[str] = ['__version__', 'constants', 'dialogs', 'widgets']
del sys, version, PackageNotFoundError

