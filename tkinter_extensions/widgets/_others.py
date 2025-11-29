"""
Created on Sun Jun 16 23:52:18 2024
@author: tungchentsai
"""
import tkinter as tk
from types import TracebackType
from typing import Any, Literal, Self
from collections.abc import Callable
from traceback import format_exc
from tkinter import ttk

import ttkbootstrap as tb
from ttkbootstrap import style as tb_style
from ttkbootstrap import utility as tb_utility
from ttkbootstrap.style import Colors
from ttkbootstrap.widgets import tooltip as tb_tooltip
_StyleBuilderTTK = tb_style.StyleBuilderTTK  # backup

from tkinter_extensions import utils
from tkinter_extensions import variables as vrb
from tkinter_extensions._constants import (
    Int, IntFloat, Anchor, TakeFocus, ScreenUnits, Padding
)
# =============================================================================
# MARK: Patching ttkbootstrap
# =============================================================================
class PatchedStyleBuilderTTK(_StyleBuilderTTK):
    def scale_size(self, size: IntFloat) -> Int:
        return utils.scale_size(self.style.master, size)
    
    def update_combobox_popdown_style(self, *args, **kwargs):
        result = super().update_combobox_popdown_style(*args, **kwargs)
        
        # Fix combobox' scrollbar
        self.create_scrollbar_style()
        
        return result


# =============================================================================
# MARK: Widgets
# =============================================================================
class Window(tb.Window):
    _root: Callable[[], Self]
    
    def __init__(self, *args, **kwargs):
        self._patch_ttkbootstrap()
        super().__init__(*args, **kwargs)
        self._error_message: vrb.StringVar = vrb.StringVar(self, name='err_msg')
    
    def _patch_ttkbootstrap(self) -> None:
        # Fix platform-dependent scaling factor
        tb_utility.scale_size = utils.scale_size
        
        # Patch StyleBuilderTTK
        tb_style.StyleBuilderTTK = PatchedStyleBuilderTTK
    
    def report_callback_exception(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        exc: type[BaseException],
        val: BaseException,
        tb: TracebackType | None = None
    ) -> None:
        """
        Catch error message.
        """
        super().report_callback_exception(exc, val, tb)
        self._error_message.set(format_exc())


#TODO: New Mixin class for `redirect_layout_managers`


class UndockedFrame(tk.Frame):  # tb can't be undocked so we use tk instead
    _root: Callable[[], tb.Window]
    
    def __init__(
        self,
        master: tk.Misc,
        *args,
        window_title: str = '',
        dock_callbacks:
            tuple[Callable | None, Callable | None]
            = (None, None),
        undock_callbacks:
            tuple[Callable | None, Callable | None]
            = (None, None),
        undock_button: bool | ttk.Button | Callable[..., ttk.Button] = True,
        place_button: bool = True,
        **kwargs
    ):
        assert isinstance(undock_button, (bool, Callable)), undock_button
        
        super().__init__(master, *args, **kwargs)
        self._window_title = window_title
        self._layout_manager: Callable | None = None
        self._layout_info: dict[str, Any] = {}
        self._dock_callbacks: tuple[Callable | None, Callable | None]
        self._undock_callbacks: tuple[Callable | None, Callable | None]
        self.set_dock_callbacks(dock_callbacks)
        self.set_undock_callbacks(undock_callbacks)
        
        self._undock_button: ttk.Button | None
        if not undock_button:
            self._undock_button = None
            return
        
        if undock_button == True:
            self._undock_button = bt = tb.Button(
                self,
                text='Undock',
                takefocus=False,
                bootstyle='link-primary',
                command=self.undock
            )
        else:
            bt = undock_button() if callable(undock_button) else undock_button
            assert isinstance(bt, ttk.Button), bt
            self._undock_button = bt
            self._undock_button.configure(command=self.undock)
        
        if place_button:
            self.place_undock_button()
        else:
            setattr(bt, '_place_info', None)
        
        self.bind('<<MapChild>>', lambda e: bt.lift(), add=True)
    
    @property
    def undock_button(self):
        return self._undock_button
    
    def place_undock_button(
        self,
        *,
        anchor: Anchor = 'se',
        x: ScreenUnits = -2,
        y: ScreenUnits = -2,
        relx: str | IntFloat = 1.,
        rely: str | IntFloat = 1.,
        **kw
    ) -> None:
        assert self._undock_button is not None
        
        undock_bt = self._undock_button
        undock_bt.place(
            anchor=anchor, relx=relx, rely=rely, x=x, y=y,  # pyright: ignore [reportArgumentType] (extended types)
            **kw
        )
        setattr(undock_bt, '_place_info', self._undock_button.place_info())
    
    def set_dock_callbacks(
        self, callbacks: tuple[Callable | None, Callable | None] = (None, None)
    ) -> None:
        assert len(callbacks) == 2, callbacks
        assert all( c is None or callable(c) for c in callbacks ), callbacks
        self._dock_callbacks = callbacks
    
    def set_undock_callbacks(
        self, callbacks: tuple[Callable | None, Callable | None] = (None, None)
    ) -> None:
        assert len(callbacks) == 2, callbacks
        assert all( c is None or callable(c) for c in callbacks ), callbacks
        self._undock_callbacks = callbacks
    
    def undock(self) -> None:
        if (manager := self.winfo_manager()) == 'pack':
            self._layout_manager = self.pack
            self._layout_info = self.pack_info()  # pyright: ignore [reportAttributeAccessIssue]
        elif manager == 'grid':
            self._layout_manager = self.grid
            self._layout_info = self.grid_info()  # pyright: ignore [reportAttributeAccessIssue] (extended types)
        elif manager == 'place':
            self._layout_manager = self.place
            self._layout_info = self.place_info()  # pyright: ignore [reportAttributeAccessIssue] (extended types)
        else:
            raise RuntimeError(
                f"Unknown layout manager: {repr(manager)}. Should be any of "
                "'pack', 'grid', or 'place'."
            )
        
        callback_begin, callback_final = self._undock_callbacks
        
        if callback_begin:
            callback_begin()
        
        tk.Wm.wm_manage(self, self)  # make self frame become a toplevel # pyright: ignore [reportArgumentType] (extended types)
        tk.Wm.wm_withdraw(self)  # pyright: ignore [reportArgumentType] (extended types)
        tk.Wm.wm_title(self, self._window_title)  # pyright: ignore [reportArgumentType, reportCallIssue] (extended types)
        tk.Wm.wm_protocol(self, 'WM_DELETE_WINDOW', self.dock)  # pyright: ignore [reportArgumentType, reportCallIssue] (extended types)
        
        undock_bt = self._undock_button
        if undock_bt and hasattr(undock_bt, '_place_info'):
            undock_bt.place_forget()
        
        self._root().focus_set()
        self.focus_set()
        
        if callback_final:
            callback_final()
        
        tk.Wm.wm_deiconify(self)  # pyright: ignore[reportArgumentType] (extended types)
        self.lift()
    
    def dock(self) -> None:
        assert self._layout_manager is not None, (
            "This frame is not undocked. Call `undock()` first to undock it."
        )
        
        callback_begin, callback_final = self._dock_callbacks
        
        if callback_begin:
            callback_begin()
        
        tk.Wm.wm_forget(self, self)  # pyright: ignore [reportArgumentType] (extended types)
        self._layout_manager(**self._layout_info)
        
        undock_bt = self._undock_button
        place_info = getattr(undock_bt, '_place_info', None)
        if undock_bt and place_info:
            undock_bt.place(**place_info)
        
        self._root().focus_set()
        self.focus_set()
        
        if callback_final:
            callback_final()


class OptionMenu(tb.OptionMenu):
    _root: Callable[[], tb.Window]
    
    def __init__(
        self,
        master: tk.Misc,
        variable: tk.Variable | None = None,
        values: tuple[str, ...] = (),
        default: str | None = None,
        command: Callable[[str], Any] | None = None,
        direction: Literal['above', 'below', 'left', 'right', 'flush'] = 'below',
        takefocus: TakeFocus = False,
        style: str = '',
        **kwargs
    ):
        super().__init__(
            master,
            variable,
            default,
            *values,
            style=style,
            direction=direction,
            command=command
        )
        self._variable: tk.Variable
        self.configure(takefocus=takefocus, **kwargs)
    
    #TODO: set_variable()
    
    def set_command(
        self, command: Callable[[str], Any] | None = None
    ) -> None:
        assert command is None or callable(command), command
        
        self._callback = command
        menu = self["menu"]
        max_idx = menu.index('end')
        if max_idx is not None:
            for i in range(max_idx + 1):
                menu.entryconfigure(i, command=command)


class Combobox(tb.Combobox):
    _root: Callable[[], tb.Window]
    
    def configure_listbox(self, **kw) -> Any:
        popdown = self.tk.eval(f'tb::combobox::PopdownWindow {self}')
        listbox = f'{popdown}.f.l'
        options: tuple[str, ...] = self._options(kw)  # pyright: ignore [reportAttributeAccessIssue] (internal function)
        return self.tk.call(listbox, 'configure', *options)
    
    def itemconfigure(self, index: Int | str, **kw) -> Any:
        popdown = self.tk.eval(f'tb::combobox::PopdownWindow {self}')
        listbox = f'{popdown}.f.l'
        values = self["values"]
        try:
            self.tk.call(listbox, 'itemconfigure', len(values) - 1)
        except tk.TclError:
            for i, value in enumerate(values):
                self.tk.call(listbox, 'insert', i, value)
        
        options: tuple[str, ...] = self._options(kw)  # pyright: ignore [reportAttributeAccessIssue] (internal function)
        return self.tk.call(listbox, 'itemconfigure', index, *options)


class ColorButton(tb.Button):
    _root: Callable[[], tb.Window]
    
    @property
    def background(self):
        return self._background
    
    def __init__(
        self,
        master: tk.Misc,
        *args,
        background: str | None = None,
        **kw
    ):
        super().__init__(master, *args, **kw)
        if background is None:
            style = self._root().style
            assert isinstance(style, tb_style.Style), style
            background = style.configure(self["style"], 'background')
        assert isinstance(background, str), background
        
        self._background: str = background
        self.set_color(background)
        self.bind('<<ThemeChanged>>', lambda e: self.set_color(), add=True)
    
    def set_color(self, background: str | None = None) -> str:
        """
        Ref: `ttkbootstrap.style.StyleBuilderTTK.create_button_style`
        """
        style_name = f'{id(self)}.TButton'
        style = self._root().style
        assert isinstance(style, tb_style.Style), style
        colors: tb_style.Colors | list = style.colors
        assert isinstance(colors, tb_style.Colors), colors
        
        bordercolor = background = background or self._background
        disabled_bg = Colors.make_transparent(0.1, colors.fg, colors.bg)
        disabled_fg = Colors.make_transparent(0.3, colors.fg, colors.bg)
        pressed = Colors.make_transparent(0.6, background, colors.bg)
        hover = Colors.make_transparent(0.7, background, colors.bg)
        
        style._build_configure(
            style_name,
            background=background,
            bordercolor=bordercolor,
            darkcolor=background,
            lightcolor=background
        )
        style.map(
            style_name,
            foreground=[("disabled", disabled_fg)],
            background=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
            bordercolor=[("disabled", disabled_bg)],
            darkcolor=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
            lightcolor=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
        )
        style._register_ttkstyle(style_name)
        
        self.configure(style=style_name)
        self._background = background
        
        return self._background


class WrapLabel(tb.Label):
    _root: Callable[[], tb.Window]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind(
            '<Configure>', lambda e: self.configure(wraplength=e.width), add=True
        )


class ToolTip(tb_tooltip.ToolTip):
    def __init__(
        self,
        widget,
        text: str = 'widget info',
        bootstyle: str | tuple[str, ...] | None = None,
        wraplength: Int | None = None,
        delay: Int = 250,   # milliseconds
        padding: Padding = (4, 1),
        **kwargs,
    ):
        super().__init__(
            widget=widget,
            text=text,
            bootstyle=bootstyle,
            wraplength=wraplength,  # pyright: ignore [reportArgumentType] (extended types)
            delay=delay,  # pyright: ignore [reportArgumentType] (extended types)
            **kwargs
        )
        
        if wraplength is None:
            self.wraplength = wraplength  # override the value
        self.padding = padding  # save for later use
    
    def show_tip(self, *_) -> None:
        if self.toplevel:
            return
        
        # Create the tooltip `Toplevel` window
        super().show_tip(*_)
        assert isinstance(self.toplevel, tb.Toplevel), self.toplevel
        
        self.toplevel: tb.Toplevel
        lb, = self.toplevel.winfo_children()  # pyright: ignore [reportGeneralTypeIssues] (always non-empty)
        assert isinstance(lb, tb.Label), (type(lb), lb)
        
        lb.configure(padding=self.padding)
         # override the value set the in super func

