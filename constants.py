#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:07:34 2023

@author: tungchentsai
"""

import sys

PLATFORM = sys.platform

RIGHTCLICK = '<ButtonPress-2>' if PLATFORM == 'darwin' else '<ButtonPress-3>'
MOUSESCROLL = ['<ButtonPress-4>', '<ButtonPress-5>'] if PLATFORM == 'linux' \
    else ['<MouseWheel>']

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

