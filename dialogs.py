#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:18:31 2022

@author: tungchentsai
@source: https://github.com/TsaiTung-Chen/tk-utils
"""

from typing import Union, Callable, Optional

from ttkbootstrap.icons import Icon
from ttkbootstrap.dialogs import QueryDialog, MessageDialog
from ttkbootstrap.dialogs.colorchooser import ColorChooserDialog
# =============================================================================
# ---- Classes
# =============================================================================
class _Positioned:
    def show(self,
             position:Union[tuple, Callable, None]=None,
             wait=True,
             callback:Optional[Callable]=None):
        self._callback = callback  # this function must receive the result value
        self._result = None
        self.build()
        
        if callable(position):  # add support for position function
            position(self._toplevel)
        elif position is None:
            self._locate()
        else:
            try:
                x, y = position
                self._toplevel.geometry(f'+{x}+{y}')
            except:
                self._locate()
        
        self._toplevel.deiconify()
        if self._alert:
            self._toplevel.bell()
        
        if self._initial_focus:
            self._initial_focus.focus_force()
        
        if wait:  # add a switch
            self._toplevel.grab_set()
            self._toplevel.wait_window()


class PositionedQueryDialog(_Positioned, QueryDialog):
    def on_submit(self, *args, **kw):
        super().on_submit(*args, **kw)
        if self._callback:
            self._callback(self._result)
    
    def on_cancel(self, *args, **kw):
        super().on_cancel(*args, **kw)
        if self._callback:
            self._callback(self._result)


class PositionedMessageDialog(_Positioned, MessageDialog):
    def on_button_press(self, *args, **kw):
        super().on_button_press(*args, **kw)
        if self._callback:
            self._callback(self._result)


class PositionedColorChooserDialog(_Positioned, ColorChooserDialog):
    def on_button_press(self, *args, **kw):
        super().on_button_press(*args, **kw)
        if self._callback:
            self._callback(self._result)


# =============================================================================
# ---- Convenience Static Methods
# =============================================================================
class PositionedMessagebox:
    """This class contains various static methods that show popups with
    a message to the end user with various arrangments of buttons
    and alert options.
    """
    @staticmethod
    def show_info(message,
                  title=" ",
                  parent=None,
                  buttons=['OK:primary'],
                  alert=False,
                  **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            message=message,
            title=title,
            alert=alert,
            parent=parent,
            buttons=buttons,
            icon=Icon.info,
            localize=True,
            **kwargs
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result
    
    @staticmethod
    def show_warning(message,
                     title=" ",
                     parent=None,
                     alert=True,
                     buttons=['OK:primary'],
                     **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            message=message,
            title=title,
            parent=parent,
            buttons=buttons,
            icon=Icon.warning,
            alert=alert,
            localize=True,
            **kwargs
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result
    
    @staticmethod
    def show_error(message,
                   title=" ",
                   parent=None,
                   buttons=['OK:primary'],
                   alert=True,
                   **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            message=message,
            title=title,
            parent=parent,
            buttons=buttons,
            icon=Icon.error,
            alert=alert,
            localize=True,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result
    
    @staticmethod
    def show_question(
        message,
        title=" ",
        parent=None,
        buttons=["No:secondary", "Yes:primary"],
        alert=True,
        **kwargs,
    ):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            message=message,
            title=title,
            parent=parent,
            buttons=buttons,
            icon=Icon.question,
            alert=alert,
            localize=True,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result
    
    @staticmethod
    def ok(message,
           title=" ",
           parent=None,
           buttons=['OK:primary'],
           alert=False,
           **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            title=title,
            message=message,
            parent=parent,
            alert=alert,
            buttons=buttons,
            localize=True,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result
    
    @staticmethod
    def okcancel(message, title=" ", alert=False, parent=None, **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            title=title,
            message=message,
            parent=parent,
            alert=alert,
            localize=True,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result
    
    @staticmethod
    def yesno(message, title=" ", alert=False, parent=None, **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            title=title,
            message=message,
            parent=parent,
            buttons=["No", "Yes:primary"],
            alert=alert,
            localize=True,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result

    @staticmethod
    def yesnocancel(message, title=" ", alert=False, parent=None, **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            title=title,
            message=message,
            parent=parent,
            alert=alert,
            buttons=["Cancel", "No", "Yes:primary"],
            localize=True,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result

    @staticmethod
    def retrycancel(message, title=" ", alert=False, parent=None, **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            title=title,
            message=message,
            parent=parent,
            alert=alert,
            buttons=["Cancel", "Retry:primary"],
            localize=True,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result


class PositionedQuerybox:
    """This class contains various static methods that request data
    from the end user.
    """
    @staticmethod
    def get_color(parent=None, title="Color Chooser", initialcolor=None, **kw):
        assert not (set(kw) - {"wait", "position", "callback"}), kw
        dialog = PositionedColorChooserDialog(parent, title, initialcolor)
        dialog.show(**kw)
        
        return dialog.result
    
    @staticmethod
    def get_string(
        prompt="", title=" ", initialvalue=None, parent=None, **kwargs
    ):
        initialvalue = initialvalue or ''
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedQueryDialog(
            prompt, title, initialvalue, parent=parent, **kwargs
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog._result
    
    @staticmethod
    def get_integer(
        prompt="",
        title=" ",
        initialvalue=None,
        minvalue=None,
        maxvalue=None,
        parent=None,
        **kwargs,
    ):
        initialvalue = initialvalue or ''
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedQueryDialog(
            prompt,
            title,
            initialvalue,
            minvalue,
            maxvalue,
            datatype=int,
            parent=parent,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog._result

    @staticmethod
    def get_float(
        prompt="",
        title=" ",
        initialvalue=None,
        minvalue=None,
        maxvalue=None,
        parent=None,
        **kwargs,
    ):
        initialvalue = initialvalue or ''
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedQueryDialog(
            prompt,
            title,
            initialvalue,
            minvalue,
            maxvalue,
            datatype=float,
            parent=parent,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog._result

