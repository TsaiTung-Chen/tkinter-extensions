"""
Created on Sun Dec 11 19:18:31 2022
@author: tungchentsai

This module defines library-wide initial constants.
"""
import platform
import tkinter as tk
from typing import TypeAlias, Any, Literal
from collections.abc import Callable, Sequence

import numpy as np
# =============================================================================
# MARK: Screen resolution
# =============================================================================
OS: str = platform.system()
TK_DPI: float = 72.  # Tk default points (physical dots) per inch
SYSTEM_PPI: float  # system's pixels per inch (PPI)  
if OS == 'Darwin':
    SYSTEM_PPI = 72.  # macOS has a default PPI value of 72
else:
    SYSTEM_PPI = 96.  # Windows and Linux have a default PPI value of 96
DEFAULT_PPD: float = SYSTEM_PPI / TK_DPI  # default pixels per point (PPD)
 # => this is about 1.00 for macOS and 1.33 for Windows/Linux


# =============================================================================
# MARK: Event Sequences
# =============================================================================
MLEFTPRESS: str = '<ButtonPress-1>'
MLEFTRELEASE: str = '<ButtonRelease-1>'
MDLEFTPRESS: str = '<Double-ButtonPress-1>'
MLEFTMOTION: str = '<B1-Motion>'
MRIGHTPRESS: str
MRIGHTRELEASE: str
MDRIGHTPRESS: str
MRIGHTMOTION: str
if OS == 'Darwin':
    MRIGHTPRESS = '<ButtonPress-2>'
    MRIGHTRELEASE = '<ButtonRelease-2>'
    MDRIGHTPRESS = '<Double-ButtonPress-2>'
    MRIGHTMOTION = '<B2-Motion>'
else:
    MRIGHTPRESS = '<ButtonPress-3>'
    MRIGHTRELEASE = '<ButtonRelease-3>'
    MDRIGHTPRESS = '<Double-ButtonPress-3>'
    MRIGHTMOTION = '<B3-Motion>'
MMOTION: str = '<Motion>'
MSCROLL: list[str] = ['<ButtonPress-4>', '<ButtonPress-5>'] if OS == 'Linux' \
    else ['<MouseWheel>']

DRAWSTARTED: str = '<<DrawStarted>>'
DRAWSUCCEEDED: str = '<<DrawSucceeded>>'
DRAWFAILED: str = '<<DrawFailed>>'
DRAWENDED: str = '<<DrawEnded>>'


# =============================================================================
# MARK: Keys
# =============================================================================
COMMAND: str
OPTION: str
if OS == 'Darwin':
    # On macOS, the command key is Mod1 and the option key is Mod2.
    COMMAND = 'Mod1'
    OPTION = 'Mod2'
else:
    # On Windows and Linux, the Ctrl key is used as the command key and the
    # Alt key is used as the option key.
    COMMAND = 'Control'
    OPTION = 'Alt'
CONTROL: str = 'Control'
SHIFT: str = 'Shift'
LOCK: str = 'Lock'
MODIFIERS: set[str] = {COMMAND, OPTION, CONTROL, SHIFT, LOCK}
MODIFIER_MASKS: dict[str, int] = {
    "Shift": int('0b1', base=2),
    "Lock": int('0b10', base=2),
    "Control": int('0b100', base=2),
    "Mod1": int('0b1_000', base=2),  # command (macOS)
    "Mod2": int('0b10_000', base=2),   # option (macOS)
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


# =============================================================================
# MARK: Built-in Widgets
# =============================================================================
BUILTIN_WIDGETS: tuple[str, ...] = (
    'TButton', 'TCheckbutton', 'TCombobox', 'TEntry', 'TFrame', 'TLabel',
    'TLabelFrame', 'TMenubutton', 'TNotebook', 'TPandedwindow',
    'TProgressbar', 'TRadiobutton', 'TScale', 'TScrollbar', 'TSeparator',
    'TSizegrip', 'Treeview'
)


# =============================================================================
# MARK: Type Aliases
# =============================================================================
_Int: TypeAlias = int | np.integer
_Float: TypeAlias = float | np.floating
_Complex: TypeAlias = complex | np.complexfloating
_Number: TypeAlias = _Int | _Float | _Complex
_IntFloat: TypeAlias = _Int | _Float

_NpInt: TypeAlias = np.int64
_NpFloat: TypeAlias = np.float64
_NpIntFloat: TypeAlias = _NpInt | _NpFloat

_Anchor: TypeAlias = Literal['center', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
_Compound: TypeAlias = Literal['top', 'left', 'center', 'right', 'bottom', 'none']
_Cursor: TypeAlias = (
    str | tuple[str] | tuple[str, str]
    | tuple[str, str, str] | tuple[str, str, str, str]
)
_Image: TypeAlias = tk.Image | str
_Relief: TypeAlias = Literal['raised', 'sunken', 'flat', 'ridge', 'solid', 'groove']
_EventCommand: TypeAlias = Callable[[tk.Event], Any]
_ButtonCommand: TypeAlias = Callable[[], Any]
_EntryValidateCommand: TypeAlias = str | Sequence[str] | Callable[[], bool]
_XYScrollCommand: TypeAlias = Callable[[float, float], Any]
_TakeFocus: TypeAlias = bool | Literal[0, 1, ''] | Callable[[str], bool | None]

_ScreenUnits: TypeAlias = _IntFloat | str
_Padding: TypeAlias = (
    _ScreenUnits
    | tuple[_ScreenUnits]
    | tuple[_ScreenUnits, _ScreenUnits]
    | tuple[_ScreenUnits, _ScreenUnits, _ScreenUnits]
    | tuple[_ScreenUnits, _ScreenUnits, _ScreenUnits, _ScreenUnits]
)

type Int = _Int
type Float = _Float
type Complex = _Complex
type Number = _Number
type IntFloat = _IntFloat

type NpInt = _NpInt
type NpFloat = _NpFloat
type NpIntFloat = _NpIntFloat

type Anchor = _Anchor
type Compound = _Compound
type Cursor = _Cursor
type Image = _Image
type Relief = _Relief
type EventCommand = _EventCommand
type ButtonCommand = _ButtonCommand
type EntryValidateCommand = _EntryValidateCommand
type XYScrollCommand = _XYScrollCommand
type TakeFocus = _TakeFocus

type ScreenUnits = _ScreenUnits
type Padding = _Padding


# =============================================================================
# MARK: Numpy constants
# =============================================================================
NPFINFO = np.finfo(_NpFloat)

