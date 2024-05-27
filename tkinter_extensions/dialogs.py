#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:18:31 2022

@author: tungchentsai
"""

import tkinter as tk
from typing import Union, Callable, Optional

import ttkbootstrap as ttk
from ttkbootstrap.icons import Icon
from ttkbootstrap.dialogs import QueryDialog, MessageDialog, FontDialog
from ttkbootstrap.dialogs.colorchooser import ColorChooserDialog
# =============================================================================
# ---- Classes
# =============================================================================
class _Positioned:
    def show(self,
             position:Union[tuple, list, Callable, None]=None,
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
    def create_body(self, master):
        super().create_body(master=master)
        self._initial_focus.select_range(0, 'end')
    
    def on_submit(self, *args, **kw):
        res = super().on_submit(*args, **kw)
        if self._callback:
            self._callback(self._result)
        return res
    
    def on_cancel(self, *args, **kw):
        res = super().on_cancel(*args, **kw)
        if self._callback:
            self._callback(self._result)
        return res


class PositionedFontDialog(_Positioned, FontDialog):
    from ttkbootstrap.localization import MessageCatalog as _MessageCatalog
    
    def __init__(self,
                 title="Font Selector",
                 parent=None,
                 default:Optional[tk.font.Font]=None):
        # Edit: set the default font as `default`
        
        #EDIT
        assert isinstance(default, (tk.font.Font, type(None))), default
        
        title = self._MessageCatalog.translate(title)
        super().__init__(parent=parent, title=title)
        
        #EDIT
        if default is None:
            default = tk.font.nametofont('TkDefaultFont')
        
        self._style = ttk.Style()
        self._default:tk.font.Font = default  #EDIT
        self._actual = self._default.actual()
        self._size = ttk.Variable(value=self._actual["size"])
        self._family = ttk.Variable(value=self._actual["family"])
        self._slant = ttk.Variable(value=self._actual["slant"])
        self._weight = ttk.Variable(value=self._actual["weight"])
        self._overstrike = ttk.Variable(value=self._actual["overstrike"])
        self._underline = ttk.Variable(value=self._actual["underline"])
        self._preview_font = tk.font.Font()
        self._slant.trace_add("write", self._update_font_preview)
        self._weight.trace_add("write", self._update_font_preview)
        self._overstrike.trace_add("write", self._update_font_preview)
        self._underline.trace_add("write", self._update_font_preview)
        
        _headingfont = tk.font.nametofont("TkHeadingFont")
        _headingfont.configure(weight="bold")
        
        self._update_font_preview()
        self._families = set([self._family.get()])
        for f in tk.font.families():
            if all([f, not f.startswith("@"), "emoji" not in f.lower()]):
                self._families.add(f)
        self._families = sorted(self._families)  #EDIT
    
    def create_body(self, master):
        # Edit: use natural window size
        
        #EDIT: width = utility.scale_size(master, 600)
        #EDIT: height = utility.scale_size(master, 500)
        #EDIT: self._toplevel.geometry(f"{width}x{height}")
        
        family_size_frame = ttk.Frame(master, padding=10)
        family_size_frame.pack(fill='x', anchor='n')
        self._initial_focus = self._font_families_selector(family_size_frame)
        self._font_size_selector(family_size_frame)
        self._font_options_selectors(master, padding=10)
        self._font_preview(master, padding=10)
    
    def _font_options_selectors(self, master, padding: int):
        # Edit: don't change the values of the tk variables
        
        container = ttk.Frame(master, padding=padding)
        container.pack(fill='x', padx=2, pady=2, anchor='n')

        weight_lframe = ttk.Labelframe(
            container, text=self._MessageCatalog.translate("Weight"), padding=5
        )
        weight_lframe.pack(side='left', fill='x', expand=1)
        opt_normal = ttk.Radiobutton(
            master=weight_lframe,
            text=self._MessageCatalog.translate("normal"),
            value="normal",
            variable=self._weight,
        )
        #EDIT: opt_normal.invoke()
        opt_normal.pack(side='left', padx=5, pady=5)
        opt_bold = ttk.Radiobutton(
            master=weight_lframe,
            text=self._MessageCatalog.translate("bold"),
            value="bold",
            variable=self._weight,
        )
        opt_bold.pack(side='left', padx=5, pady=5)

        slant_lframe = ttk.Labelframe(
            container, text=self._MessageCatalog.translate("Slant"), padding=5
        )
        slant_lframe.pack(side='left', fill='x', padx=10, expand=1)
        opt_roman = ttk.Radiobutton(
            master=slant_lframe,
            text=self._MessageCatalog.translate("roman"),
            value="roman",
            variable=self._slant,
        )
        #EDIT: opt_roman.invoke()
        opt_roman.pack(side='left', padx=5, pady=5)
        opt_italic = ttk.Radiobutton(
            master=slant_lframe,
            text=self._MessageCatalog.translate("italic"),
            value="italic",
            variable=self._slant,
        )
        opt_italic.pack(side='left', padx=5, pady=5)

        effects_lframe = ttk.Labelframe(
            container, text=self._MessageCatalog.translate("Effects"), padding=5
        )
        effects_lframe.pack(side='left', padx=(2, 0), fill='x', expand=1)
        opt_underline = ttk.Checkbutton(
            master=effects_lframe,
            text=self._MessageCatalog.translate("underline"),
            variable=self._underline,
        )
        opt_underline.pack(side='left', padx=5, pady=5)
        opt_overstrike = ttk.Checkbutton(
            master=effects_lframe,
            text=self._MessageCatalog.translate("overstrike"),
            variable=self._overstrike,
        )
        opt_overstrike.pack(side='left', padx=5, pady=5)
    
    def _font_preview(self, master, padding:int):
        # Edit: don't turn off `pack_propagate` and set a small width
        
        container = ttk.Frame(master, padding=padding)
        container.pack(fill='both', expand=1, anchor='n')

        header = ttk.Label(
            container,
            text=self._MessageCatalog.translate('Preview'),
            font='TkHeadingFont',
        )
        header.pack(fill='x', pady=2, anchor='n')

        content = self._MessageCatalog.translate(
            'The quick brown fox jumps over the lazy dog.'
        )
        self._preview_text = ttk.Text(
            master=container,
            height=3,
            width=1,   #EDIT: prevent the width from becoming too large
            font=self._preview_font,
            highlightbackground=self._style.colors.primary
        )
        self._preview_text.insert('end', content)
        self._preview_text.pack(fill='both', expand=1)
        #EDIT: container.pack_propagate(False)
    
    def _update_font_preview(self, *_):
        # Edit: configure the weight of text and update `self._result` when 
        # submitted
        
        self._preview_font.config(
            family=self._family.get(),
            size=self._size.get(),
            slant=self._slant.get(),
            weight=self._weight.get(),   #EDIT
            overstrike=self._overstrike.get(),
            underline=self._underline.get()
        )
        try:
            self._preview_text.configure(font=self._preview_font)
        except:
            pass
        #EDIT: self._result = self._preview_font
    
    def _on_submit(self):
        # Edit: update `self._result` when submitted
        
        self._result = self._preview_font  #EDIT
        res = super()._on_submit()
        if self._callback:
            self._callback(self._result)
        return res
    
    def _on_cancel(self):
        # Edit: update `self._result` when submitted
        
        res = super()._on_cancel()
        if self._callback:
            self._callback(self._result)
        return res


class PositionedMessageDialog(_Positioned, MessageDialog):
    def on_button_press(self, *args, **kw):
        res = super().on_button_press(*args, **kw)
        if self._callback:
            self._callback(self._result)
        return res


class PositionedColorChooserDialog(_Positioned, ColorChooserDialog):
    def on_button_press(self, *args, **kw):
        res = super().on_button_press(*args, **kw)
        if self._callback:
            self._callback(self._result)
        return res


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

