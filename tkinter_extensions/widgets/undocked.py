#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import tkinter as tk
from typing import Callable

import ttkbootstrap as ttk
# =============================================================================
# ---- Classes
# =============================================================================
class UndockedFrame(tk.Frame):  # ttk can't be undocked so use tk instead
    def __init__(
            self,
            master,
            *args,
            window_title: str = '',
            dock_callbacks: tuple[Callable | None,
                                  Callable | None] = (None, None),
            undock_callbacks: tuple[Callable | None,
                                    Callable | None] = (None, None),
            undock_button: bool = True,
            place_button: bool = True,
            **kwargs
    ):
        super().__init__(master, *args, **kwargs)
        self._window_title = window_title
        self._layout_manager: Callable | None = None
        self._layout_info: dict | None = None
        self.set_dock_callbacks(dock_callbacks)
        self.set_undock_callbacks(undock_callbacks)
        
        if undock_button:
            self._undock_button = bt = ttk.Button(
                self,
                text='Undock',
                takefocus=False,
                bootstyle='link-primary',
                command=self.undock
            )
            if place_button:
                self.place_undock_button()
            else:
                bt._place_info = None
            self.bind('<<MapChild>>', lambda e: bt.lift(), add=True)
        else:
            self._undock_button = None
    
    @property
    def undock_button(self):
        return self._undock_button
    
    def place_undock_button(
            self, *, anchor='se', relx=1., rely=1., x=-2, y=-2, **kw):
        assert self._undock_button is not None
        
        self._undock_button.place(
            anchor=anchor, relx=relx, rely=rely, x=x, y=y, **kw)
        self._undock_button._place_info = self._undock_button.place_info()
    
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
        if (manager := self.winfo_manager()) == 'pack':
            self._layout_manager = self.pack
            self._layout_info = self.pack_info()
        elif manager == 'grid':
            self._layout_manager = self.grid
            self._layout_info = self.grid_info()
        elif manager == 'place':
            self._layout_manager = self.place
            self._layout_info = self.place_info()
        else:
            raise RuntimeError(
                f"Unknown layout manager: {repr(manager)}. Should be any of "
                "'pack', 'grid', or 'place'."
            )
        
        callback_begin, callback_final = self._undock_callbacks
        
        if callback_begin:
            callback_begin()
        
        tk.Wm.wm_manage(self, self)  # make self frame become a toplevel
        tk.Wm.wm_withdraw(self)
        tk.Wm.wm_title(self, self._window_title)
        tk.Wm.wm_protocol(self, 'WM_DELETE_WINDOW', self.dock)
        
        if self._undock_button and self._undock_button._place_info:
            self._undock_button.place_forget()
        
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
        self._layout_manager(**self._layout_info)
        
        if self._undock_button and self._undock_button._place_info:
            self._undock_button.place(**self._undock_button._place_info)
        
        if callback_final:
            callback_final()

