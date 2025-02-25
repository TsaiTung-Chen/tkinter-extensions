#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import time
import tkinter as tk
from tkinter.font import nametofont
from typing import Literal

import ttkbootstrap as ttk

from ..constants import MODIFIER_MASKS, COMMAND, MOUSESCROLL
from ..utils import (
    defer,
    bind_recursively,
    unbind_recursively,
    redirect_layout_managers
)
# =============================================================================
# ---- Scrollbar
# =============================================================================
class AutoHiddenScrollbar(ttk.Scrollbar):  # hide if all visible
    def __init__(self,
                 master=None,
                 autohide: bool = True,
                 autohide_ms: int = 300,
                 command=None,
                 **kwargs):
        super().__init__(master, command=command, **kwargs)
        self.autohide: bool = bool(autohide)
        self._autohide_ms: int = int(autohide_ms)
        self._manager = None
        self._last_func: dict = {"name": 'show', "id": None}
    
    @property
    def autohide(self) -> bool:
        return self._autohide
    
    @autohide.setter
    def autohide(self, auto: bool | None):
        if auto is not None:
            self._autohide = bool(auto)
    
    @property
    def hidden(self):
        return self._last_func["name"] == 'hide'
    
    @property
    def all_visible(self):
        return [ float(v) for v in self.get() ] == [0., 1.]
    
    def set(self, first, last):
        if self._last_func["id"] is None:  # init
            self.show()
        
        if self.autohide and self.hidden and (not self.all_visible):
            self.show()
        
        super().set(first, last)
        
        if self.autohide and (not self.hidden) and self.all_visible:
            self.hide(after_ms=self._autohide_ms)
    
    def show(self, after_ms: int = -1, autohide: bool | None = None):
        if self._manager is None:
            self._manager = self.winfo_manager()
        assert self._manager == 'grid', self._manager
        
        self.autohide = autohide
        id_ = time.monotonic()
        self._last_func = {"name": 'show', "id": id_}
        
        if after_ms < 0:
            self._show(id_)
        else:
            self.after(after_ms, self._show, id_)
    
    def hide(self, after_ms: int = -1, autohide: bool | None = None):
        if self._manager is None:
            self._manager = self.winfo_manager()
        assert self._manager == 'grid', self._manager
        
        self.autohide = autohide
        id_ = time.monotonic()
        self._last_func = {"name": 'hide', "id": id_}
        
        if after_ms < 0:
            self._hide(id_)
        else:
            self.after(after_ms, self._hide, id_)
    
    def _show(self, id_):
        if self._last_func == {"name": 'show', "id": id_}:
            self.grid()
    
    def _hide(self, id_):
        if self._last_func == {"name": 'hide', "id": id_}:
            self.grid_remove()


# =============================================================================
# ---- Patches
# =============================================================================
class _GeneralView:
    def __init__(self, widget, orient, sensitivity: float = 1.):
        assert orient in ('x', 'y'), orient
        self._widget = widget
        self._orient = orient
        self.sensitivity = sensitivity
        self.start = 0  # pixel location
    
    @property
    def sensitivity(self):
        return self._sensitivity
    
    @sensitivity.setter
    def sensitivity(self, value: float):
        self._sensitivity = float(value)
    
    @property
    def stop(self):  # pixel location
        content_size = self._get_content_size()
        showing = self._get_showing_size(content_size=content_size)
        return self.start + showing
    
    @property
    def step(self):
        style = ttk.Style.get_instance()
        font_name = style.lookup(self._widget.winfo_class(), 'font')
        font = nametofont(font_name or 'TkDefaultFont')
        linespace = font.metrics()["linespace"]
        if self._orient == 'y':
            return self._sensitivity * linespace * 1  # 2-linespace height
        return self._sensitivity * linespace * 2  # 4-linespace width
    
    def view(self, *args):
        """Update the vertical position of the inner widget within the outer
        frame.
        """
        if not args:
            return self._to_fraction(self.start, self.stop)
        
        action, args = args[0], args[1:]
        if action == 'moveto':
            return self.view_moveto(float(args[0]))
        elif action == 'scroll':
            return self.view_scroll(int(args[0]), args[1])
        raise ValueError("The first argument must be 'moveto' or 'scroll' but "
                         f"got: {args[0]}")
    
    def view_moveto(self, fraction: float):
        """Update the position of the inner widget within the outer frame.
        """
        # Check the start and stop locations are valid
        content_size = self._get_content_size()
        showing = self._get_showing_size(content_size=content_size)
        self.start = self._to_pixel(fraction, content_size=content_size)
        self.start, stop = self._confine_region(
            self.start, content_size, showing)
        
        # Update widgets
        self._move_content_and_scrollbar(self.start, stop, content_size)
    
    def view_scroll(self, number: int, what: str):
        """Update the position of the inner widget within the outer frame.
        Note: If `what == 'units'` and `number == 1`, the content will be 
        scrolled down 1 line (y orientation). If `what == 'pages'`, the content 
        will be scrolled down five times the amount of the aforementioned lines 
        (y orientation)
        """
        if what == 'pages':
            pixel = number * self.step * 5
        else:
            pixel = number * self.step
        
        # Check the start and stop locations are valid
        content_size = self._get_content_size()
        showing = self._get_showing_size(content_size=content_size)
        self.start += pixel
        self.start, stop = self._confine_region(
            self.start, content_size, showing)
        
        # Update widgets
        self._move_content_and_scrollbar(self.start, stop, content_size)
    
    def _confine_region(self, start, content_size, showing):
        stop = start + showing
        if start < 0:
            start = 0
            stop = showing
        elif stop > content_size:
            stop = content_size
            start = stop - showing
        return start, stop
    
    def _move_content_and_scrollbar(self, start, stop, content_size):
        first, last = self._to_fraction(start, stop, content_size=content_size)
        if self._orient == 'x':  # X orientation
            self._widget.content_place(x=-start)
            if self._widget._set_xscrollbar:
                self._widget._set_xscrollbar(first, last)
            return self._widget.update_idletasks()
        
        # Y orientation
        self._widget.content_place(y=-start)
        if self._widget._set_yscrollbar:
            self._widget._set_yscrollbar(first, last)
        self._widget.update_idletasks()
    
    def _to_fraction(self, *pixels, content_size=None):
        content_size = content_size or self._get_content_size()
        return tuple( pixel / content_size for pixel in pixels )
    
    def _to_pixel(self, *fractions, content_size=None):
        content_size = content_size or self._get_content_size()
        numbers = tuple( round(fraction * content_size) for fraction in fractions )
        if len(numbers) == 1:
            return numbers[0]
        return numbers
    
    def _get_showing_size(self, content_size=None):
        self._widget.update_idletasks()
        if self._orient == 'x':
            showing = self._widget.cropper.winfo_width()
        else:
            showing = self._widget.cropper.winfo_height()
        return min(showing, content_size or self._get_content_size())
    
    def _get_content_size(self):
        self._widget.update_idletasks()
        if self._orient == 'x':
            return self._widget.winfo_width()
        return self._widget.winfo_height()


class _GeneralXYView:
    """This class is a workaround to mimic the `tkinter.XView` behavior.
    
    This class is designed to be used with multiple inheritance and must be 
    the 2nd parent class. This means that the 1st parent class must call this 
    class' `__init__` function. After that this class will automatically call 
    the 3rd parent class' `__init__` function
    """
    
    def __init__(self, *args, **kwargs):
        # Init the 2nd parent class
        self._set_xscrollbar = None
        self._set_yscrollbar = None
        kwargs = self._configure_scrollcommands(kwargs)
        super().__init__(*args, **kwargs)
        
        # Init x and y GeneralViews
        self._xview = _GeneralView(widget=self, orient='x', sensitivity=0.5)
        self._yview = _GeneralView(widget=self, orient='y')
    
    def xview(self, *args, **kwargs):
        return self._xview.view(*args, **kwargs)
    
    def xview_moveto(self, *args, **kwargs):
        return self._xview.view_moveto(*args, **kwargs)
    
    def xview_scroll(self, *args, **kwargs):
        return self._xview.view_scroll(*args, **kwargs)
    
    def yview(self, *args, **kwargs):
        return self._yview.view(*args, **kwargs)
    
    def yview_moveto(self, *args, **kwargs):
        return self._yview.view_moveto(*args, **kwargs)
    
    def yview_scroll(self, *args, **kwargs):
        return self._yview.view_scroll(*args, **kwargs)
    
    def configure(self, *args, **kwargs):
        kwargs = self._configure_scrollcommands(kwargs)
        return super().configure(*args, **kwargs)
    
    def _configure_scrollcommands(self, kwargs):
        self._set_xscrollbar = kwargs.pop("xscrollcommand", self._set_xscrollbar)
        self._set_yscrollbar = kwargs.pop("yscrollcommand", self._set_yscrollbar)
        return kwargs
    
    def _refresh(self):
        self.xview_scroll(0, 'unit')
        self.yview_scroll(0, 'unit')


class _Scrolled:
    """This class is designed to be used with multiple inheritance and must be 
    the 1st parent class. This means that it will automatically call the 2nd 
    parent class' `__init__` function
    """
    
    _scrollbar_padding = (0, 1)
    
    def __init__(
            self,
            master=None,
            scroll_orient: str = 'vertical',
            autohide: bool = True,
            hbootstyle='round',
            vbootstyle='round',
            scroll_sensitivities: float
                                  | tuple[float, float]
                                  | list[float, float] = 1.,
            builtin_method=False,
            propagate_geometry=True,
            container_type: Literal['tk', 'ttk'] = 'ttk',
            **kwargs
    ):
        assert container_type in ('tk', 'ttk'), container_type
        valid_orients = ('vertical', 'horizontal', 'both')
        assert scroll_orient in valid_orients, (valid_orients, scroll_orient)
        self._builtin_method = builtin_method
        self.set_scroll_sensitivities(scroll_sensitivities)
        
        # Outer frame (container)
        container = (tk.Frame if container_type == 'tk' else ttk.Frame)(
            master=master, relief='flat', borderwidth=0
        )
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.grid_propagate(propagate_geometry)
        self._container = container
        
        self._cropper = None
        if builtin_method:
            # Main widget
            super().__init__(master=container, **kwargs)
            self.grid(row=0, column=0, sticky='nesw')
        else:
            # Inner frame (cropper)
            self._cropper = cropper = ttk.Frame(
                master=container, relief='flat', borderwidth=0)
            cropper.grid(row=0, column=0, sticky='nesw')
            
            # Main widget
            super().__init__(master=cropper, **kwargs)
            self.place(x=0, y=0)
        
        # Scrollbars
        self._hbar = self._vbar = None
        if scroll_orient in ('horizontal', 'both'):
            self._hbar = hbar = AutoHiddenScrollbar(
                master=container,
                autohide=autohide,
                command=self.xview,
                bootstyle=hbootstyle,
                orient='horizontal',
            )
            hbar.grid(row=1, column=0, sticky='ew', pady=self._scrollbar_padding)
            self.configure(xscrollcommand=hbar.set)
        elif not builtin_method:
            self.place(relwidth=1.)
        
        if scroll_orient in ('vertical', 'both'):
            self._vbar = vbar = AutoHiddenScrollbar(
                master=container,
                autohide=autohide,
                command=self.yview,
                bootstyle=vbootstyle,
                orient='vertical',
            )
            vbar.grid(row=0, column=1, sticky='ns', padx=self._scrollbar_padding)
            self.configure(yscrollcommand=vbar.set)
        elif not builtin_method:
            self.place(relheight=1.)
        
        self.bind('<Map>', self._on_map, add=True)
        self.bind('<<MapChild>>', self._on_map_child, add=True)
        self.bind('<Configure>', self._on_configure, add=True)
        self._os = self.tk.call('tk', 'windowingsystem')
        
        redirect_layout_managers(self, container, orig_prefix='content_')
    
    @property
    def container(self) -> ttk.Frame:
        return self._container  # outer frame
    
    @property
    def cropper(self) -> ttk.Frame | None:
        return self._cropper  # inner frame
    
    @property
    def hbar(self) -> AutoHiddenScrollbar | None:
        return self._hbar
    
    @property
    def vbar(self) -> AutoHiddenScrollbar | None:
        return self._vbar
    
    def set_scroll_sensitivities(
            self,
            sens: float | tuple[float, float] | list[float, float] | None = None
    ) -> tuple[float, float]:
        assert isinstance(sens, (float, tuple, list)), sens
        
        try:
            self._scroll_sensitivities = _, _ = tuple( float(s) for s in sens )
        except TypeError:
            self._scroll_sensitivities = (float(sens), float(sens))
        
        return self._scroll_sensitivities
    
    def set_autohide_scrollbars(
            self, enable: bool | None = None
    ) -> tuple[bool, bool]:
        
        states = [None, None]
        if self.hbar:
            self.hbar.autohide = enable
            states[0] = self.hbar.autohide
        
        if self.vbar:
            self.vbar.autohide = enable
            states[1] = self.vbar.autohide
        
        return tuple(states)
    
    def show_scrollbars(
            self, after_ms: int = -1, autohide: bool | None = None):
        if self.hbar:
            self.hbar.show(after_ms, autohide=autohide)
        if self.vbar:
            self.vbar.show(after_ms, autohide=autohide)
    
    def hide_scrollbars(
            self, after_ms: int = -1, autohide: bool | None = None):
        if self.hbar:
            self.hbar.hide(after_ms, autohide=autohide)
        if self.vbar:
            self.vbar.hide(after_ms, autohide=autohide)
    
    def rebind_mousewheel(self):
        self.unbind_mousewheel()
        
        # Bind mousewheel
        funcs = [self._mousewheel_scroll] * len(MOUSESCROLL)
        bind_recursively(
            self, MOUSESCROLL, funcs,
            add=True,
            key='scrolled-wheel',
            skip_toplevel=True
        )
    
    def unbind_mousewheel(self):
        unbind_recursively(self, key='scrolled-wheel')
    
    def _on_configure(self, event=None):
        if not self._builtin_method:
            self._refresh()
    
    def _on_map(self, event=None):
        self.rebind_mousewheel()
        
        if not self._builtin_method:
            if not self.hbar:
                self.cropper.configure(width=self.winfo_reqwidth())
            if not self.vbar:
                self.cropper.configure(height=self.winfo_reqheight())
    
    def _on_map_child(self, event):
        # Rebind the mapchild callback to all descendants
        for child in event.widget.winfo_children():
            unbind_recursively(child, key='scrolled-mapchild')
            bind_recursively(
                child,
                '<<MapChild>>',
                self._on_map_child,
                add=True,
                key='scrolled-mapchild',
                skip_toplevel=True
            )
        
        if self.container.winfo_ismapped():
            self._on_map(event)
    
    def _mousewheel_scroll(self, event):
        """Callback for when the mouse wheel is scrolled.
        Modified from: `ttkbootstrap.scrolled.ScrolledFrame._on_mousewheel`
        """
        if event.num == 4:  # Linux
            delta = 10.
        elif event.num == 5:  # Linux
            delta = -10.
        elif self._os == "win32":  # Windows
            delta = event.delta / 120.
        else:  # Mac
            delta = event.delta
        
        x_direction = (event.state & MODIFIER_MASKS["Shift"]) \
            == MODIFIER_MASKS["Shift"]
        sensitivity = self._scroll_sensitivities[0 if x_direction else 1]
        number = -round(delta * sensitivity)
        
        if x_direction:
            if self.hbar:
                self.xview('scroll', number, 'units')
        elif self.vbar:
            self.yview('scroll', number, 'units')
        
        return 'break'
    
    def content_size(
            self, hbar: bool = False, vbar: bool = False) -> tuple[int, int]:
        self.update_idletasks()
        content_width = self.winfo_reqwidth()
        content_height = self.winfo_reqheight()
        
        pad_width = pad_height = sum(self._scrollbar_padding)
        if vbar and self.vbar:
            vbar_width = self.vbar.winfo_reqwidth()
        else:
            vbar_width = pad_width = 0
        
        if hbar and self.hbar:
            hbar_height = self.hbar.winfo_reqheight()
        else:
            hbar_height = pad_height = 0
        
        return (content_width + vbar_width + pad_width,
                content_height + hbar_height + pad_height)
    
    def set_size(self, width: int | None = None, height: int | None = None):
        if self._builtin_method:
            raise TypeError("This function does not support built-in methods.")
        
        self.cropper.configure(width=width, height=height)
        self.container.configure(width=width, height=height)


# =============================================================================
# ---- Scrolled Widgets with the GeneralXYView
# =============================================================================
def create_scrolledwidget(widget: tk.Misc = ttk.Frame):
    """Structure:
    <. Container (outer frame) >
        <.1 Cropper (inner frame) >
            <.1.1 wrapped widget >
        <.2 horizontal scrollbar >
        <.3 vertical scrollbar >
    """
    assert issubclass(widget, tk.Misc), widget
    class _ScrolledWidget(_Scrolled, _GeneralXYView, widget): pass
    return _ScrolledWidget


def ScrolledWidget(master=None,
                   widget: tk.Misc = ttk.Frame,
                   **kwargs):
    """A convenience function working like a class instance init function.
    """
    return create_scrolledwidget(widget=widget)(
        master=master, builtin_method=False, **kwargs)


# =============================================================================
# ---- Scrolled Widgets with the builtin method
# =============================================================================
class ScrolledTreeview(_Scrolled, ttk.Treeview):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, builtin_method=True, **kwargs)


class ScrolledText(_Scrolled, ttk.Text):
    def __init__(self,
                 *args,
                 readonly: bool = False,
                 bind_select_all: bool = True,
                 **kwargs):
        super().__init__(*args, builtin_method=True, **kwargs)
        if readonly:
            self.bind('<KeyPress>', self._prevent_modification, add=True)
        
        if bind_select_all:
            self.bind(f'<{COMMAND}-A>', self._select_all, add=True)
            self.bind(f'<{COMMAND}-a>', self._select_all, add=True)
    
    def _prevent_modification(self, event):
        command_mask = MODIFIER_MASKS[COMMAND]
        if (event.keysym.lower() in ('c', 'a')) and (
                (event.state & command_mask) == command_mask):
            return
        return 'break'
    
    def _select_all(self, event=None):
        self.event_generate('<<SelectAll>>')
        return 'break'


class ScrolledCanvas(_Scrolled, tk.Canvas):
    def __init__(self, *args, fill: bool = False, **kwargs):
        assert isinstance(fill, bool), (type(fill), fill)
        
        self._on_map_child = defer(200)(self._on_map_child)
        super().__init__(*args, builtin_method=True, **kwargs)
        self._fill = fill
    
    def xview(self, *args):
        if super().xview() != (0.0, 1.0):  # prevent from over scrolling
            super().xview(*args)
    
    def yview(self, *args):
        if super().yview() != (0.0, 1.0):  # prevent from over scrolling
            super().yview(*args)
    
    def _update_scrollregion(self) -> tuple:
        self.update_idletasks()
        x1, y1, x2, y2 = self.bbox('all')
        x2, y2 = max(x2, 0), max(y2, 0)
        scrollregion = (0, 0, x2, y2)
        self.configure(scrollregion=scrollregion)
        return scrollregion
    
    def _on_configure(self, event=None):
        # Fill the space with content in the non-scrollable direction
        if self._fill:
            self.update_idletasks()
            x1, y1, x2, y2 = self.bbox('all')
            xscale = 1.0 if self.hbar else self.winfo_width() / x2
            yscale = 1.0 if self.vbar else self.winfo_height() / y2
            
            ## Scale objects
            self.scale('all', 0, 0, xscale, yscale)
        
        self._update_scrollregion()
    
    def _on_map(self, event=None):
        self.refresh()
    
    def refresh(self):
        self.rebind_mousewheel()
        
        if self._fill:
            # Init widget size to enable widget scaling in `self._on_configure`
            self.update_idletasks()
            for oid in self.find_all():
                if self.type(oid) != 'window':
                    continue
                elif (self.hbar or int(self.itemcget(oid, 'width')) != 0) and (
                        self.vbar or int(self.itemcget(oid, 'height')) != 0):
                    continue
                widget = self.nametowidget(self.itemcget(oid, 'window'))
                width = None if self.hbar else widget.winfo_reqwidth()
                height = None if self.vbar else widget.winfo_reqheight()
                self.itemconfigure(oid, width=width, height=height)
        
        # Resize the canvas to fit the content
        if bbox := self.bbox('all'):
            x1, y1, x2, y2 = bbox
            self.configure(width=x2, height=y2)
    
    def content_size(
            self, hbar: bool = False, vbar: bool = False) -> tuple[int, int]:
        self.update_idletasks()
        x1, y1, content_width, content_height = self.bbox('all')
        
        pad_width = pad_height = sum(self._scrollbar_padding)
        if vbar and self.vbar:
            vbar_width = self.vbar.winfo_reqwidth()
        else:
            vbar_width = pad_width = 0
        
        if hbar and self.hbar:
            hbar_height = self.hbar.winfo_reqheight()
        else:
            hbar_height = pad_height = 0
        
        return (content_width + vbar_width + pad_width,
                content_height + hbar_height + pad_height)
    
    def set_size(self, width: int | None = None, height: int | None = None):
        self.configure(width=width, height=height)
        self._on_configure()


class _CanvasBasedScrolled:
    def __init__(
            self,
            master=None,
            scroll_orient: str = 'vertical',
            autohide: bool = True,
            hbootstyle='round',
            vbootstyle='round',
            scroll_sensitivities: float
                                  | tuple[float, float]
                                  | list[float, float] = 1.,
            propagate_geometry=True,
            **kwargs
    ):
        canvas_kw = {
            "scroll_orient": scroll_orient,
            "autohide": autohide,
            "hbootstyle": hbootstyle,
            "vbootstyle": vbootstyle,
            "scroll_sensitivities": scroll_sensitivities,
            "propagate_geometry": propagate_geometry
        }
        
        # [master [ScrolledCanvas [self]]]
        self._canvas = ScrolledCanvas(master, fill=True, **canvas_kw)
        super().__init__(self._canvas, **kwargs)
        self._id = self._canvas.create_window(0, 0, anchor='nw', window=self)
        self.bind('<<MapChild>>', self._canvas._on_map_child, add=True)
        
        redirect_layout_managers(self, self._canvas, orig_prefix='content_')
    
    @property
    def container(self) -> ttk.Frame:
        return self._canvas._container  # outer container
    
    @property
    def canvas(self) -> ScrolledCanvas:
        return self._canvas  # scrollable canvas
    
    @property
    def hbar(self) -> AutoHiddenScrollbar | None:
        return self._canvas.hbar
    
    @property
    def vbar(self) -> AutoHiddenScrollbar | None:
        return self._canvas.vbar
    
    def set_scroll_sensitivities(self, *args, **kwargs):
        return self.canvas.set_scroll_sensitivities(*args, **kwargs)
    
    def set_autohide_scrollbars(self, *args, **kwargs):
        return self.canvas.set_autohide_scrollbars(*args, **kwargs)
    
    def show_scrollbars(self, *args, **kwargs):
        return self.canvas.show_scrollbars(*args, **kwargs)
    
    def hide_scrollbars(self, *args, **kwargs):
        return self.canvas.hide_scrollbars(*args, **kwargs)
    
    def rebind_mousewheel(self, *args, **kwargs):
        return self.canvas.rebind_mousewheel(*args, **kwargs)
    
    def unbind_mousewheel(self, *args, **kwargs):
        return self.canvas.unbind_mousewheel(*args, **kwargs)
    
    def content_size(self, *args, **kwargs):
        return self.canvas.content_size(*args, **kwargs)
    
    def set_size(self, *args, **kwargs):
        return self.canvas.set_size(*args, **kwargs)
    
    def refresh(self, *args, **kwargs):
        return self.canvas.refresh(*args, **kwargs)


class ScrolledFrame(_CanvasBasedScrolled, ttk.Frame): pass
class ScrolledTkFrame(_CanvasBasedScrolled, tk.Frame): pass
class ScrolledLabelframe(_CanvasBasedScrolled, ttk.Labelframe): pass


# =============================================================================
# %% Main
# =============================================================================
if __name__ == '__main__':
    from ..utils import quit_if_all_closed
    from ._others import Window
    
    
    root = Window(themename='cyborg')
    root.withdraw()
    
    
    win1 = ttk.Toplevel(title='ScrolledText')
    win1.lift()
    
    st = ScrolledText(win1, autohide=True, wrap='none', readonly=True)
    st.insert('end', ttk.tk.__doc__)
    st.pack(fill='both', expand=True)
    
    
    win2 = ttk.Toplevel(title='ScrolledFrame')
    win2.lift()
    
    sf = ScrolledFrame(win2, autohide=False, scroll_orient='vertical')
    sf.pack(fill='both', expand=True)
    for i in range(20):
        text = str(i) + ': ' + '_'.join(str(i) for i in range(30))
        ttk.Button(sf, text=text).pack(anchor='e')
    
    
    win3 = ttk.Toplevel(title='ScrolledCanvas')
    win3.lift()
    
    sc = ScrolledCanvas(
        win3, scroll_orient='vertical', autohide=False, vbootstyle='round-light'
    )
    sc.pack(fill='both', expand=True)
    sc.create_polygon(
        (10, 5), (600, 300), (900, 600), (300, 600), (300, 600),
        outline='red', stipple='gray25'
    )
    sc.configure(bg='gray')
    
    
    for win in [win1, win2, win3]:
        win.place_window_center()
        win.protocol('WM_DELETE_WINDOW', quit_if_all_closed(win))
    
    
    root.mainloop()

