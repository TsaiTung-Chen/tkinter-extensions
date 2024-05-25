#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
@source: https://github.com/TsaiTung-Chen/tk-utils
"""

import tkinter as tk
from typing import Optional

import ttkbootstrap as ttk

from ..utils import redirect_layout_managers
# =============================================================================
# ---- Widgets
# =============================================================================
class CollapsedFrame(ttk.Frame):
    def __init__(self,
                 master=None,
                 text='',
                 labelwidget:tk.BaseWidget=None,
                 variable:Optional[tk.Variable]=None,
                 onvalue=None,
                 orient:str='vertical',
                 style:Optional[str]=None,
                 bootstyle:Optional[str]=None,
                 **kw):
        _style = None if style is None else style.lower()
        assert orient in ('vertical', 'horizontal'), orient
        assert text or labelwidget, (text, labelwidget)
        assert isinstance(variable, (tk.Variable, type(None))), variable
        
        self._orient = orient
        bootstyle_button = bootstyle
        if bootstyle:
            bootstyle = ttk.Bootstyle.ttkstyle_widget_color(bootstyle)
        
        if style and ('frame' in _style):
            style_button = None
        else:
            style_button = style
            style = None
        
        self._onvalue = onvalue = onvalue or '__on__'
        self._variable = variable = variable or tk.StringVar(
            master, value=onvalue)
        self._labelwidget = labelwidget = labelwidget or ttk.Checkbutton(
            master,
            text=text,
            onvalue=onvalue,
            offvalue=f'!{onvalue}',
            variable=variable,
            style=style_button,
            bootstyle=bootstyle_button
        )
        variable.trace_add('write', self.toggle)
        
        self._container = container = ttk.Labelframe(master,
                                                     labelwidget=labelwidget,
                                                     style=style,
                                                     bootstyle=bootstyle,
                                                     **kw)
        
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        super().__init__(container, relief='flat', borderwidth=0)
        self.grid(sticky='nesw')
        
        redirect_layout_managers(self, container, orig_prefix='content_')
        self._collapsed = False
    
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
        self.container.grid_propagate(0)
        self.content_grid_remove()
        self.container.configure(width=width, height=height)
    
    def resume(self):
        if not self.collapsed:
            return
        
        self._collapsed = False
        self.container.grid_propagate(1)
        self.content_grid()
    
    def toggle(self, name=None, index=None, mode=None):
        if self.variable.get() == self._onvalue:  # show
            self.resume()
        else:  # hide
            self.collapse()


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    root = ttk.Window(themename='cyborg')
    
    for orient, labelanchor, sep in [('vertical', 'n', ' '),
                                     ('horizontal', 'w', '\n')]:
        text = sep.join(['Click', 'me', 'to', 'collapse'])
        collapsed = CollapsedFrame(root,
                                   text=text,
                                   orient=orient,
                                   labelanchor=labelanchor,
                                   bootstyle='success-round-toggle')
        collapsed.grid(sticky='w', padx=6, pady=6)
        ttk.Label(collapsed, text='Label 1').pack()
        ttk.Label(collapsed, text='Label 2').pack()
    
    root.mainloop()

