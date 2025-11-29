"""
Created on Mon May 22 22:35:24 2023
@author: tungchentsai
"""
import tkinter as tk
from types import NoneType
from typing import Any, Literal
from collections.abc import Callable

import ttkbootstrap as tb

from tkinter_extensions import variables as vrb
from tkinter_extensions.utils import redirect_layout_managers
# =============================================================================
# MARK: Widgets
# =============================================================================
class CollapsedFrame(tb.Frame):
    _root: Callable[[], tb.Window]
    
    def __init__(
        self,
        master: tk.Misc,
        text: str | None = None,
        orient: Literal['horizontal', 'vertical'] = 'vertical',
        labelwidget: tk.Misc | None = None,
        variable: vrb.Variable | None = None,
        onvalue: Any = '__on__',
        offvalue: Any = '__off__',
        button_style: str = '',
        button_bootstyle: str | tuple[str, ...] | None = None,
        **kw
    ):
        assert orient in ('vertical', 'horizontal'), orient
        assert text or labelwidget, (text, labelwidget)
        assert isinstance(variable, (tk.Variable, NoneType)), variable
        
        self._orient: Literal['horizontal', 'vertical'] = orient
        self._onvalue: Any = onvalue
        self._offvalue: Any = offvalue
        self._variable: vrb.Variable = variable or vrb.StringVar(
            master, value=self._onvalue
        )
        self._labelwidget: tk.Misc = labelwidget or tb.Checkbutton(
            master,
            text=text if text is not None else '',
            onvalue=self._onvalue,
            offvalue=self._offvalue,
            variable=self._variable,
            style=button_style,
            bootstyle=button_bootstyle
        )
        self._variable.trace_add('write', self._on_variable_update, weak=True)
        
        self._container = container = tb.Labelframe(
            master,
            labelwidget=self._labelwidget,
            **kw
        )
        
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        super().__init__(container, relief='flat', borderwidth=0)
        self.grid(sticky='nesw')
        
        redirect_layout_managers(self, container, orig_prefix='content_')
        self.content_pack_configure: Callable
        self.content_pack: Callable
        self.content_pack_forget: Callable
        self.content_pack_info: Callable
        self.content_pack_propagate: Callable
        self.content_pack_slaves: Callable
        self.content_place_configure: Callable
        self.content_place: Callable
        self.content_place_forget: Callable
        self.content_place_info: Callable
        self.content_place_slaves: Callable
        self.content_grid_configure: Callable
        self.content_grid: Callable
        self.content_grid_forget: Callable
        self.content_grid_remove: Callable
        self.content_grid_info: Callable
        self.content_grid_propagate: Callable
        self.content_grid_slaves: Callable
        self.content_grid_bbox: Callable
        self.content_grid_location: Callable
        self.content_grid_size: Callable
        self.content_grid_anchor: Callable
        
        self._collapsed: bool = False
        self._on_variable_update()
    
    @property
    def container(self):
        return self._container
    
    @property
    def labelwidget(self):
        return self._labelwidget
    
    @property
    def collapsed(self):
        return self._collapsed
    
    @property
    def variable(self):
        return self._variable
    
    def collapse(self) -> None:
        self.variable.set(self._offvalue)
    
    def expand(self) -> None:
        self.variable.set(self._onvalue)
    
    def toggle(self) -> None:
        if self.variable.get() == self._onvalue:
            self.collapse()
        else:
            self.expand()
    
    def _collapse(self) -> None:
        if self.collapsed:
            return
        
        self._collapsed = True
        self.update_idletasks()
        if self._orient == 'vertical':
            width = self.container.winfo_reqwidth()
            height = self.labelwidget.winfo_reqheight() + 4
        else:  # horizontal
            width = self.labelwidget.winfo_reqwidth() + 4
            height = self.container.winfo_reqheight()
        self.container.grid_propagate(False)
        self.content_grid_remove()
        self.container.configure(width=width, height=height)
    
    def _expand(self) -> None:
        if not self.collapsed:
            return
        
        self._collapsed = False
        self.container.grid_propagate(True)
        self.content_grid()
    
    def _on_variable_update(self, *_) -> None:
        if self.variable.get() == self._onvalue:  # show
            self._expand()
        else:  # hide
            self._collapse()

