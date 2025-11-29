"""
Created on Wed Jul 17 22:18:34 2024
@author: tungchentsai
"""
from typing import TypeAlias, Any, Literal
from collections.abc import Callable
from weakref import ref, WeakMethod
import tkinter as tk

from tkinter_extensions.utils import DropObject, mixin_base
# =============================================================================
# MARK: Mixin Patches
# =============================================================================
class _Memory(mixin_base(tk.Variable), metaclass=DropObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self, tk.Variable), (
            f'{type(self)} must be a subclass of tk.Variable'
        )
        self._tk_variable_get: Callable[[], str] = (  #TODO: rename => get_raw
            lambda: tk.Variable.get(self)
        )
        self._previous_value: str = self._tk_variable_get()
    
    @property
    def previous_value(self):
        return self._previous_value
    
    def value_changed(self, update: bool = False) -> bool:
        prev_value: str = self._previous_value
        new_value: str = self._tk_variable_get()
        if update:
            self._previous_value = new_value
        
        return new_value != prev_value


class _Weakref(mixin_base(tk.Variable), metaclass=DropObject):
    def trace_add(
        self,
        mode: Literal['array', 'read', 'write', 'unset'],
        callback: 'VariableTraceCommand',
        weak: bool = False
    ) -> str:
        if weak:
            if hasattr(callback, '__self__') and hasattr(callback, '__func__'):
                # `callback` is a bound method
                wref = WeakMethod(callback)
            else:
                # `callback` is a function or unbound method
                wref = ref(callback)
            callback = self._wrap_callback(wref)
        
        return super().trace_add(
            mode, callback
        )
    
    def _wrap_callback(
        self, wref: ref['VariableTraceCommand']
    ) -> 'VariableTraceCommand':
        def _wrapped_callback(*args, **kwargs):
            callback = wref()
            
            if callback:
                return callback(*args, **kwargs)
        #> end of _wrapped_callback()
        
        return _wrapped_callback


# =============================================================================
# MARK: Variables
# =============================================================================
class StringVar(_Memory, _Weakref, tk.StringVar): pass
class IntVar(_Memory, _Weakref, tk.IntVar): pass
class DoubleVar(_Memory, _Weakref, tk.DoubleVar): pass
class BooleanVar(_Memory, _Weakref, tk.BooleanVar): pass


_VariableTraceCommand: TypeAlias = Callable[[str, str, str], Any]
_Variable = StringVar | IntVar | DoubleVar | BooleanVar

type VariableTraceCommand = _VariableTraceCommand
type Variable = _Variable

