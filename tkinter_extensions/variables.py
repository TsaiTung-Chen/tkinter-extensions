#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 22:18:34 2024

@author: tungchentsai
"""

from weakref import ref, WeakMethod
import tkinter as tk
# =============================================================================
# ---- Patches
# =============================================================================
class _Memory:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tk_variable_get = lambda: tk.Variable.get(self)
        self._previous_value = self._tk_variable_get()
    
    @property
    def previous_value(self):
        return self._previous_value
    
    def value_changed(self, update: bool = False):
        prev_value = self._previous_value
        new_value = self._tk_variable_get()
        if update:
            self._previous_value = new_value
        
        return new_value != prev_value


class _Weakref:
    def trace_add(self, mode, callback, weak: bool = False):
        if weak:
            if hasattr(callback, '__self__') and hasattr(callback, '__func__'):
                wref = WeakMethod(callback)
            else:
                wref = ref(callback)
            callback = lambda *args, **kwargs: wref()(*args, **kwargs)
        
        return super().trace_add(mode, callback)


# =============================================================================
# ---- Variables
# =============================================================================
class StringVar(_Memory, _Weakref, tk.StringVar): pass
class IntVar(_Memory, _Weakref, tk.IntVar): pass
class DoubleVar(_Memory, _Weakref, tk.DoubleVar): pass
class BooleanVar(_Memory, _Weakref, tk.BooleanVar): pass

