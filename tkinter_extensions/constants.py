#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:18:31 2022

@author: tungchentsai
"""

import sys
import numpy as np


PLATFORM = sys.platform
LEFTCLICK = '<ButtonPress-1>'
RIGHTCLICK = '<ButtonPress-2>' if PLATFORM == 'darwin' else '<ButtonPress-3>'
MOUSESCROLL = ['<ButtonPress-4>', '<ButtonPress-5>'] if PLATFORM == 'linux' else [
    '<MouseWheel>']

if PLATFORM == 'darwin':
    COMMAND = 'Mod1'
    OPTION = 'Mod2'
else:
    COMMAND = 'Control'
    OPTION = 'Alt'
CONTROL = 'Control'
SHIFT = 'Shift'
LOCK = 'Lock'
MODIFIERS = {COMMAND, OPTION, CONTROL, SHIFT, LOCK}
MODIFIER_MASKS = {
    "Shift": int('0b1', base=2),
    "Lock": int('0b10', base=2),
    "Control": int('0b100', base=2),
    "Mod1": int('0b1_000', base=2),  # command (Mac)
    "Mod2": int('0b10_000', base=2),   # option (Mac)
    "Mod3": int('0b100_000', base=2),
    "Mod4": int('0b1000_000', base=2),
    "Mod5": int('0b10000_000', base=2),
    "Button1": int('0b100_000_000', base=2),
    "Button2": int('0b1_000_000_000', base=2),
    "Button3": int('0b10_000_000_000', base=2),
    "Button4": int('0b100_000_000_000', base=2),
    "Button5": int('0b1000_000_000_000', base=2),
    "Alt": int('0b100_000_000_000_000_000', base=2)
}


BUILTIN_WIDGETS = [
    'TButton', 'TCheckbutton', 'TCombobox', 'TEntry', 'TFrame', 'TLabel',
    'TLabelFrame', 'TMenubutton', 'TNotebook', 'TPandedwindow',
    'TProgressbar', 'TRadiobutton', 'TScale', 'TScrollbar', 'TSeparator',
    'TSizegrip', 'Treeview'
]


if PLATFORM == 'darwin':  # macOS has a default DPI value of 72
    DEFAULT_SCALE = 72./72.  # this is about 1.00
else:  # Windows and Linux have a default DPI value of 96
    DEFAULT_SCALE = 96./72.  # this is about 1.33


Int = int | np.integer
Float = float | np.floating
Complex = complex | np.complexfloating
Number = Int | Float | Complex
IntFloat = Int | Float


del sys

