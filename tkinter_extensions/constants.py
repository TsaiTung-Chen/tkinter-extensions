"""
Created on Sun Dec 11 19:18:31 2022
@author: tungchentsai

This module defines library-wide constants. It includes both initial constants
imported from the `tkinter_extensions._constants` module and derived constants
that are defined after their object dependencies have been established.
"""
from tkinter_extensions._constants import (
    OS, TK_DPI, SYSTEM_PPI, DEFAULT_PPD,
    
    MLEFTPRESS, MLEFTRELEASE, MDLEFTPRESS, MLEFTMOTION, MRIGHTPRESS,
    MRIGHTRELEASE, MDRIGHTPRESS, MRIGHTMOTION, MMOTION, MSCROLL,
    DRAWSTARTED, DRAWSUCCEEDED, DRAWFAILED, DRAWENDED,
    
    COMMAND, OPTION, CONTROL, SHIFT, LOCK, MODIFIERS, MODIFIER_MASKS,
    
    BUILTIN_WIDGETS,
    
    Int, _Int, Float, _Float, Complex, _Complex, Number, _Number,
    IntFloat, _IntFloat,
    NpInt, _NpInt, NpFloat, _NpFloat, NpIntFloat, _NpIntFloat,
    Anchor, _Anchor, Compound, _Compound, Cursor, _Cursor, Image, _Image,
    Relief, _Relief, EventCommand, _EventCommand, ButtonCommand, _ButtonCommand,
    EntryValidateCommand, _EntryValidateCommand,
    XYScrollCommand, _XYScrollCommand, TakeFocus, _TakeFocus,
    ScreenUnits, _ScreenUnits, Padding, _Padding,
    
    NPFINFO
)
from tkinter_extensions.variables import (
    _VariableTraceCommand, VariableTraceCommand, _Variable, Variable
)

