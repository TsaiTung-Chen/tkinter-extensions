#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import tkinter as tk
from typing import Tuple, Callable

import ttkbootstrap as ttk
# =============================================================================
# ---- Classes
# =============================================================================
class UndockedFrame(tk.Frame):  # ttk can't be undocked so use tk instead
    def __init__(self,
                 master,
                 *args,
                 window_title:str='',
                 dock_callbacks:Tuple[Callable, Callable]=(None, None),
                 undock_callbacks:Tuple[Callable, Callable]=(None, None),
                 place_button:bool=True,
                 **kwargs):
        super().__init__(master, *args, **kwargs)
        self._window_title = window_title
        self._layout_manager = None
        self._layout_info = None
        self.set_dock_callbacks(dock_callbacks)
        self.set_undock_callbacks(undock_callbacks)
        
        self.undock_bt = bt = ttk.Button(
            self,
            text='Undock',
            takefocus=0,
            bootstyle='link-primary',
            command=self.undock
        )
        if place_button:
            self.place_undock_button()
        else:
            bt._place_info = None
        self.bind('<<MapChild>>', lambda e: bt.lift())
    
    def place_undock_button(
            self, *, anchor='se', relx=1., rely=1., x=-2, y=-2, **kw):
        self.undock_bt.place(anchor=anchor, relx=relx, rely=rely, x=x, y=y, **kw)
        self.undock_bt._place_info = self.undock_bt.place_info()
        self.undock_bt.lift()
    
    def set_dock_callbacks(self, callbacks=(None, None)):
        if callbacks is None:
            callbacks = (None, None)
        assert len(callbacks) == 2, callbacks
        assert all( c is None or callable(c) for c in callbacks ), callbacks
        self._dock_callbacks = callbacks
    
    def set_undock_callbacks(self, callbacks=(None, None)):
        if callbacks is None:
            callbacks = (None, None)
        assert len(callbacks) == 2, callbacks
        assert all( c is None or callable(c) for c in callbacks ), callbacks
        self._undock_callbacks = callbacks
    
    def undock(self):
        callback_begin, callback_final = self._undock_callbacks
        
        if callback_begin:
            callback_begin()
        
        tk.Wm.wm_manage(self, self)  # make self frame become a toplevel
        tk.Wm.wm_withdraw(self)
        tk.Wm.wm_title(self, self._window_title)
        tk.Wm.wm_protocol(self, 'WM_DELETE_WINDOW', self.dock)
        
        if self.undock_bt._place_info:
            self.undock_bt.place_forget()
        
        if callback_final:
            callback_final()
        
        tk.Wm.wm_deiconify(self)
        self._root().focus_set()
        self.focus_set()
        self.lift()
    
    def dock(self):
        callback_begin, callback_final = self._dock_callbacks
        
        if callback_begin:
            callback_begin()
        
        tk.Wm.wm_forget(self, self)
        getattr(self, self._layout_manager)(**self._layout_info)
        
        if self.undock_bt._place_info:
            self.undock_bt.place(**self.undock_bt._place_info)
        
        if callback_final:
            callback_final()
    
    def pack_configure(self, *args, **kwargs):
        super().pack_configure(*args, **kwargs)
        self._layout_manager = 'pack'
        self._layout_info = self.pack_info()
    
    pack = pack_configure
    
    def grid_configure(self, *args, **kwargs):
        super().grid_configure(*args, **kwargs)
        self._layout_manager = 'grid'
        self._layout_info = self.grid_info()
    
    grid = grid_configure
    
    def place_configure(self, *args, **kwargs):
        super().place_configure(*args, **kwargs)
        self._layout_manager = 'place'
        self._layout_info = self.place_info()
    
    place = place_configure

