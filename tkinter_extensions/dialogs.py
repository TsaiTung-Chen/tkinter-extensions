#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:18:31 2022

@author: tungchentsai
"""

import tkinter as tk
from weakref import WeakMethod
from typing import Union, Callable, Optional

import ttkbootstrap as ttk
from ttkbootstrap.icons import Icon
from ttkbootstrap.validation import validator, add_validation
from ttkbootstrap.dialogs import Dialog, QueryDialog, MessageDialog, FontDialog
from ttkbootstrap.dialogs.colorchooser import ColorChooserDialog

from .widgets import Combobox
# =============================================================================
# ---- Classes
# =============================================================================
class _Positioned:
    def show(self,
             position: Union[tuple, list, Callable, None] = None,
             wait: bool = True,
             callback: Optional[Callable] = None):
        self._callback = callback  #EDITED: this must receive the result value
        self._result = None
        self.build()
        self._toplevel.wm_resizable(True, True)  #EDITED: make it resizable
        
        if callable(position):  #EDITED: add support for position function
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
        
        if wait:  #EDITED: add a switch
            self._toplevel.grab_set()
            self._toplevel.wait_window()
    
    def build(self):
        super().build()
        self._toplevel.bind('<Destroy>', lambda e: self.destroy())


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


class PositionedMessageDialog(_Positioned, MessageDialog):
    def on_button_press(self, *args, **kw):
        res = super().on_button_press(*args, **kw)
        if self._callback:
            self._callback(self._result)
        return res


class PositionedColorChooserDialog(_Positioned, ColorChooserDialog):
    def __init__(self, *args, **kwargs):  #EDITED
        super().__init__(*args, **kwargs)
        
        # Remove and set the callback with weakref to avoid circular refs
        trace_info = self.dropper.result.trace_info()
        assert len(trace_info) == 1, trace_info
        
        wref_trace_dropper_color = WeakMethod(self.trace_dropper_color)
        trace_dropper_color = lambda *_: wref_trace_dropper_color()(*_)
        
        cbname = trace_info[0][1]
        self.dropper.result.trace_remove('write', cbname)
        self.dropper.result.trace_add('write', trace_dropper_color)
    
    def on_button_press(self, *args, **kw):
        res = super().on_button_press(*args, **kw)
        if self._callback:
            self._callback(self._result)
        return res


class PositionedFontDialog(_Positioned, FontDialog):
    from ttkbootstrap.localization import MessageCatalog as _MessageCatalog
    
    def __init__(self,
                 title="Font Selector",
                 parent=None,
                 default: Optional[tk.font.Font] = None,
                 scale:float=1.):  # actual font size = int(option size * scale)
        #EDITED: set the default font as `default`
        
        #EDITED
        assert isinstance(default, (tk.font.Font, type(None))), default
        assert scale > 0, scale
        
        title = self._MessageCatalog.translate(title)
        Dialog.__init__(self, parent=parent, title=title)
        
        #EDITED
        if default is None:
            default = tk.font.nametofont('TkDefaultFont')
        assert scale == 1 or hasattr(default, '_unscaled_size'), (scale, default)
        unscaled_size = getattr(default, '_unscaled_size', default.actual('size'))
        default = default.copy()
        
        # EDITED: use weakref to avoid circular refs
        wref_update_font_preview = WeakMethod(self._update_font_preview)
        update_font_preview = lambda *_: wref_update_font_preview()(*_)
        
        self._scale = scale  #EDITED
        self._style = ttk.Style()
        self._default:tk.font.Font = default  #EDITED
        self._actual = self._default.actual()
        self._size = ttk.Variable(value=self._actual["size"])
        self._family = ttk.Variable(value=self._actual["family"])
        self._slant = ttk.Variable(value=self._actual["slant"])
        self._weight = ttk.Variable(value=self._actual["weight"])
        self._overstrike = ttk.Variable(value=self._actual["overstrike"])
        self._underline = ttk.Variable(value=self._actual["underline"])
        self._preview_font = default.copy()  #EDITED
        self._preview_font._unscaled_size = unscaled_size  #EDITED
        self._size.trace_add('write', update_font_preview)  #EDITED
        self._family.trace_add('write', update_font_preview)  #EDITED
        self._slant.trace_add("write", update_font_preview)  #EDITED
        self._weight.trace_add("write", update_font_preview)  #EDITED
        self._overstrike.trace_add("write", update_font_preview)  #EDITED
        self._underline.trace_add("write", update_font_preview)  #EDITED
        
        #EDITED: _headingfont = font.nametofont("TkHeadingFont")
        #EDITED: _headingfont.configure(weight="bold")
        
        #EDITED: self._update_font_preview()
        self._families = set([self._family.get()])
        for f in tk.font.families():
            if all([f, not f.startswith("@"), "emoji" not in f.lower()]):
                self._families.add(f)
        self._families = sorted(self._families)  #EDITED
    
    def create_body(self, master):
        #EDITED: use natural window size
        
        #EDITED: width = utility.scale_size(master, 600)
        #EDITED: height = utility.scale_size(master, 500)
        #EDITED: self._toplevel.geometry(f"{width}x{height}")
        
        family_size_frame = ttk.Frame(master, padding=10)
        family_size_frame.pack(fill='x', anchor='n')
        self._initial_focus = self._font_families_selector(family_size_frame)
        self._font_size_selector(family_size_frame)
        self._font_options_selectors(master, padding=10)
        self._font_preview(master, padding=10)
    
    def _font_families_selector(self, master):
        container = ttk.Frame(master)
        container.pack(fill='both', expand=1, side='left')
        
        header = ttk.Label(
            container,
            text=self._MessageCatalog.translate("Family"),
            font="TkHeadingFont"
        )
        header.pack(fill='x', pady=(0, 2), anchor='n')
        
        #EDITED: add new font family optionmenu
        om = ttk.OptionMenu(
            container,
            self._family,
            None,
            *self._families,
            bootstyle='outline',
            command=lambda value: style.configure(om_style, font=(value,))
        )
        om.pack(fill='x', expand=1)
        
        om_menu = om["menu"]
        for idx, family in enumerate(self._families):
            om_menu.entryconfigure(idx, font=(family,))
        
        om_style = f'{id(om)}.{om["style"]}'
        style = om._root().style
        style.configure(om_style, font=(self._family.get(),))
        om.configure(style=om_style)
        
        return om
        
        '''EDIT: remove the Treeview
        listbox = ttk.Treeview(
            master=container,
            height=5,
            show="",
            columns=[0],
        )
        .
        .
        .
        '''
    
    def _font_size_selector(self, master):
        container = ttk.Frame(master)
        container.pack(side='left', fill='y', padx=(10, 0))
        
        header = ttk.Label(
            container,
            text=self._MessageCatalog.translate("Size"),
            font="TkHeadingFont",
        )
        header.pack(fill='x', pady=(0, 2), anchor='n')
        
        # EDIT: add new size combobox
        @validator
        def _positive_int(event):
            try:
                value = int(event.postchangetext)
            except ValueError:
                return False
            if value <= 0:
                return False
            self._size.set(int(size_buffer.get() * self._scale))
             # update the actual size
            self._preview_font._unscaled_size = size_buffer.get()
             # update the unscaled size
            self._update_font_preview()
            return True
        
        sizes = [*range(8, 13), *range(13, 30, 2), 36, 48, 72]
        size_buffer = tk.IntVar(value=self._preview_font._unscaled_size)
        cb = Combobox(
            container,
            textvariable=size_buffer,
            values=sizes,
            width=3
        )
        cb.pack()
        add_validation(cb, _positive_int)
        cb.bind('<Return>', lambda e: container.focus_set())
        cb.bind('<<ComboboxSelected>>', lambda e: container.focus_set())
        
        '''#EDITED: remove the Treeview
        sizes_listbox = ttk.Treeview(container, height=7, columns=[0], show="")
        .
        .
        .
        '''
    
    def _font_options_selectors(self, master, padding: int):
        #EDITED: don't change the values of the tk variables
        
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
        #EDITED: opt_normal.invoke()
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
        #EDITED: opt_roman.invoke()
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
    
    def _font_preview(self, master, padding: int):
        #EDITED: don't turn off `pack_propagate` and set a small width
        
        container = ttk.Frame(master, padding=padding)
        container.pack(fill='both', expand=1, anchor='n')

        header = ttk.Label(
            container,
            text=self._MessageCatalog.translate('Preview'),
            font="TkHeadingFont",
        )
        header.pack(fill='x', pady=2, anchor='n')

        content = self._MessageCatalog.translate(
            'The quick brown fox jumps over the lazy dog.'
        )
        self._preview_text = ttk.Text(
            master=container,
            height=3,
            width=1,   #EDITED: prevent the width from becoming too large
            font=self._preview_font,
            highlightbackground=self._style.colors.primary
        )
        self._preview_text.insert('end', content)
        self._preview_text.pack(fill='both', expand=1)
        #EDITED: container.pack_propagate(False)
    
    def _update_font_preview(self, *_):
        #EDITED: configure the weight of text and update `self._result` when 
        # submitted
        
        self._preview_font.config(
            family=self._family.get(),
            size=self._size.get(),   #EDITED
            slant=self._slant.get(),
            weight=self._weight.get(),   #EDITED
            overstrike=self._overstrike.get(),
            underline=self._underline.get()
        )
        '''#EDITED
        try:
            self._preview_text.configure(font=self._preview_font)
        except:
            pass
        self._result = self._preview_font
        '''
    
    def _on_submit(self):
        #EDITED: update `self._result` when submitted
        
        self._result = self._preview_font  #EDITED
        res = super()._on_submit()
        if self._callback:
            self._callback(self._result)
        return res
    
    def _on_cancel(self):
        #EDITED: update `self._result` when submitted
        
        res = super()._on_cancel()
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
    def show_info(
            parent=None,
            title=" ",
            message='',
            buttons=['OK:primary'],
            alert=False,
            **kwargs
    ):
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
    def show_warning(
            parent=None,
            title=" ",
            message='',
            buttons=['OK:primary'],
            alert=True,
            **kwargs
    ):
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
    def show_error(
            parent=None,
            title=" ",
            message='',
            buttons=['OK:primary'],
            alert=True,
            **kwargs
    ):
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
        parent=None,
        title=" ",
        message='',
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
    def ok(
        parent=None,
        title=" ",
        message='',
        buttons=['OK:primary'],
        alert=False,
        **kwargs
    ):
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
    def okcancel(parent=None, title=" ", message='', alert=False, **kwargs):
        position = kwargs.pop("position", None)
        wait = kwargs.pop("wait", True)
        callback = kwargs.pop("callback", None)
        dialog = PositionedMessageDialog(
            title=title,
            message=message,
            parent=parent,
            buttons=['Cancel:secondary', 'OK:primary'],
            alert=alert,
            localize=True,
            **kwargs,
        )
        dialog.show(position, wait=wait, callback=callback)
        return dialog.result
    
    @staticmethod
    def yesno(parent=None, title=" ", message='', alert=False, **kwargs):
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
    def yesnocancel(parent=None, title=" ", message='', alert=False, **kwargs):
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
    def retrycancel(parent=None, title=" ", message='', alert=False, **kwargs):
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
    def get_font(
            parent=None, title="Font Selector", default=None, scale=1.0, **kw):
        assert not (set(kw) - {"wait", "position", "callback"}), kw
        dialog = PositionedFontDialog(
            parent=parent, title=title, default=default, scale=scale)
        dialog.show(**kw)
        
        return dialog.result
    
    @staticmethod
    def get_color(parent=None, title="Color Chooser", initialcolor=None, **kw):
        assert not (set(kw) - {"wait", "position", "callback"}), kw
        dialog = PositionedColorChooserDialog(parent, title, initialcolor)
        dialog.show(**kw)
        
        return dialog.result
    
    @staticmethod
    def get_string(
        parent=None, title=" ", prompt="", initialvalue=None, **kwargs
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
        parent=None,
        title=" ",
        prompt="",
        initialvalue=None,
        minvalue=None,
        maxvalue=None,
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
        parent=None,
        title=" ",
        prompt="",
        initialvalue=None,
        minvalue=None,
        maxvalue=None,
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

