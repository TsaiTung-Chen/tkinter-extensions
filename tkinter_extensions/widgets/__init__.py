"""
Created on Mon May 22 22:35:24 2023
@author: tungchentsai
"""
from tkinter_extensions.widgets.dnd import (
    DnDItem, OrderlyDnDItem, DnDContainer, RearrangedDnDContainer
)
from tkinter_extensions.widgets.collapsed import CollapsedFrame
from tkinter_extensions.widgets.scrolled import ScrolledWidget
from tkinter_extensions.widgets.scrolled import (
    ScrolledTkFrame, ScrolledFrame, ScrolledLabelframe
)
from tkinter_extensions.widgets.scrolled import (
    ScrolledText, ScrolledTreeview, ScrolledCanvas
)
matplotlib = version = Version = mpl_version = None
try:
    from importlib.metadata import version
    
    import matplotlib
    from packaging.version import Version
    
    mpl_version = Version(version('matplotlib'))
    if mpl_version < Version('3.8.4') or mpl_version >= Version('3.9.0'):
        raise ModuleNotFoundError
except ModuleNotFoundError:
    Plotter = None
else:
    from tkinter_extensions.widgets.plotter import Plotter
del matplotlib, version, Version, mpl_version

from tkinter_extensions.widgets.figure import Figure
from tkinter_extensions.widgets.spreadsheets import Sheet, Book
from tkinter_extensions.widgets._others import (
    Window, UndockedFrame, OptionMenu, Combobox, ColorButton, WrapLabel,
    ToolTip
)


__all__ = [
    'DnDItem', 'OrderlyDnDItem', 'DnDContainer', 'RearrangedDnDContainer',
    'CollapsedFrame', 'ScrolledWidget', 'ScrolledTkFrame', 'ScrolledFrame',
    'ScrolledLabelframe', 'ScrolledText', 'ScrolledTreeview', 'ScrolledCanvas',
    'UndockedFrame', 'Plotter', 'Figure', 'Sheet', 'Book', 'Window', 'OptionMenu',
    'Combobox', 'ColorButton', 'WrapLabel', 'ToolTip'
]

