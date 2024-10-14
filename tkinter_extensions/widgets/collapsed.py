#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import tkinter as tk
from typing import Optional

import ttkbootstrap as ttk

from .. import variables as vrb
from ..utils import redirect_layout_managers
# =============================================================================
# ---- Widgets
# =============================================================================
class CollapsedFrame(ttk.Frame):
    def __init__(self,
                 master=None,
                 text='',
                 orient: str = 'vertical',
                 labelwidget: tk.BaseWidget = None,
                 variable: Optional[tk.Variable] = None,
                 onvalue='__on__',
                 offvalue='__off__',
                 button_style: Optional[str] = None,
                 button_bootstyle: Optional[str] = None,
                 **kw):
        assert orient in ('vertical', 'horizontal'), orient
        assert text or labelwidget, (text, labelwidget)
        assert isinstance(variable, (tk.Variable, type(None))), variable
        
        self._orient = orient
        self._onvalue = onvalue
        self._offvalue = offvalue
        self._variable = variable or vrb.StringVar(master, value=self._onvalue)
        self._labelwidget = labelwidget or ttk.Checkbutton(
            master,
            text=text,
            onvalue=self._onvalue,
            offvalue=self._offvalue,
            variable=self._variable,
            style=button_style,
            bootstyle=button_bootstyle
        )
        self._variable.trace_add('write', self._on_variable_update, weak=True)
        
        self._container = container = ttk.Labelframe(
            master,
            labelwidget=self._labelwidget,
            **kw
        )
        
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        super().__init__(container, relief='flat', borderwidth=0)
        self.grid(sticky='nesw')
        
        redirect_layout_managers(self, container, orig_prefix='content_')
        self._collapsed = False
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
    
    def collapse(self):
        self.variable.set(self._offvalue)
    
    def expand(self):
        self.variable.set(self._onvalue)
    
    def toggle(self):
        if self.variable.get() == self._onvalue:
            self.collapse()
        else:
            self.expand()
    
    def _collapse(self):
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
    
    def _expand(self):
        if not self.collapsed:
            return
        
        self._collapsed = False
        self.container.grid_propagate(True)
        self.content_grid()
    
    def _on_variable_update(self, *_):
        if self.variable.get() == self._onvalue:  # show
            self._expand()
        else:  # hide
            self._collapse()


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    root = ttk.Window(themename='cyborg')
    
    for orient, labelanchor, sep in [('vertical', 'n', ' '),
                                     ('horizontal', 'w', '\n')]:
        text = sep.join(['Click', 'me', 'to', 'collapse', 'or', 'expand'])
        collapsed = CollapsedFrame(
            root,
            text=text,
            orient=orient,
            labelanchor=labelanchor,
            button_bootstyle='success-round-toggle'
        )
        collapsed.grid(sticky='w', padx=6, pady=6)
        ttk.Label(collapsed, text='Label 1').pack()
        ttk.Label(collapsed, text='Label 2').pack()
    
    root.mainloop()

