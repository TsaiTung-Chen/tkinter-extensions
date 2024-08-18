#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
@repository: https://github.com/TsaiTung-Chen/tkinter-extensions
"""

from importlib.metadata import version, PackageNotFoundError

from . import constants, dialogs, widgets


try:
    __version__ = version('tkinter_extensions')
except PackageNotFoundError:
    __version__ = ''

__all__ = ['__version__', 'constants', 'dialogs', 'widgets']
del version, PackageNotFoundError

