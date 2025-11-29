#!/usr/bin/env python3
"""
Created on Mon May 22 22:35:24 2023
@author: tungchentsai
@repository: https://github.com/TsaiTung-Chen/tkinter-extensions
"""
import sys
from tkinter import TkVersion
from importlib.metadata import version, PackageNotFoundError

from tkinter_extensions import constants, dialogs, widgets

__all__: list[str] = ['__version__', 'constants', 'dialogs', 'widgets']
# =============================================================================
# MARK: Version Check
# =============================================================================
if sys.version_info < (3, 10):
    raise ImportError(
        'tkinter_extensions requires Python 3.10 or higher. '
        'Please upgrade your Python version.'
    )

# Tk 9.0 with Python 3.13.5 does not work well, so we restrict the version to
# lower than 9.0.
#TODO: Check if Tk 9 with Python 3.14 works correctly
if TkVersion >= 9.0:
    raise ImportError(
        'tkinter_extensions requires Tk version lower than 9.0, but your version '
        f'is {TkVersion}. '
        'Please downgrade your Tk version. '
        'You can run `python -m tkinter` to check your Tk version.'
    )

try:
    __version__ = version('tkinter_extensions')
except PackageNotFoundError:
    __version__ = ''

del sys, TkVersion, version, PackageNotFoundError

