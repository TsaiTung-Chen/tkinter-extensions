#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import gc
import copy
import time
import tkinter as tk
import tkinter.font
from contextlib import contextmanager
from typing import Callable, Literal, Generator

import numpy as np
import polars as pl
import ttkbootstrap as ttk
from ttkbootstrap.icons import Icon
from ttkbootstrap.colorutils import color_to_hsl

from ..constants import (
    RIGHTCLICK, MOUSESCROLL, MODIFIERS, MODIFIER_MASKS, COMMAND, SHIFT, LOCK
)
from ..utils import (
    get_modifiers, center_window, modify_hsl,
    df_full, df_set_values, df_concat
)
from .. import dialogs
from .. import variables as vrb
from .dnd import OrderlyDnDItem, RearrangedDnDContainer
from .scrolled import AutoHiddenScrollbar, ScrolledFrame
from ._others import OptionMenu
# =============================================================================
# ---- Classes
# =============================================================================
class DuplicateNameError(ValueError): pass


class History:
    @property
    def step(self) -> int:
        return self._step
    
    @property
    def backable(self) -> bool:
        return self.step > 0
    
    @property
    def forwardable(self) -> bool:
        return self.step < len(self._stack["forward"])
    
    def __init__(
            self,
            max_height: int,
            callback: Callable | None = None
    ):
        assert callback is None or callable(callback), callback
        
        self._callback: Callable | None = callback
        self._sequence: dict[str, list[Callable]] | None = None
        self._stack = {"forward": list(), "backward": list()}
        self._max_height = self.set_max_height(max_height)
        self._step = 0
    
    def set_max_height(self, height: int):
        assert isinstance(height, int), height
        assert height >= 1, height
        self._max_height = height
        return self._max_height
    
    def reset(self, callback: Callable | None = None):
        self.__init__(callback=callback)
    
    def add(self, forward: Callable, backward: Callable):
        assert callable(forward) and callable(backward), (forward, backward)
        
        if self._sequence is None:
            forward = self._stack["forward"][:self.step] + [forward]
            backward = self._stack["backward"][:self.step] + [backward]
            n = len(forward)
            forward = forward[-self._max_height:]
            backward = backward[-self._max_height:]
            self._step -= (n - len(forward))
            self._step += 1
            self._stack.update(forward=forward, backward=backward)
            if self._callback:
                self._callback()
        else:
            self._sequence["forward"].append(forward)
            self._sequence["backward"].append(backward)
    
    @contextmanager
    def add_sequence(self):
        yield self.start_sequence()
        self.stop_sequence()
    
    def start_sequence(self) -> dict:
        assert self._sequence is None, self._sequence
        self._sequence = {"forward": list(), "backward": list()}
        return self._sequence
    
    def stop_sequence(self):
        assert isinstance(self._sequence, dict), self._sequence
        forward, backward = self._sequence["forward"], self._sequence["backward"]
        
        if forward or backward:
            self.add(
                forward=lambda: [ func() for func in forward ],
                backward=lambda: [ func() for func in backward[::-1] ]
            )
        
        self._sequence = None
    
    def pop(self) -> dict[str, list[Callable]]:
        assert self.step > 0, self.step
        
        self._step -= 1
        trailing = {"forward": self._stack["forward"][self.step:],
                    "backward": self._stack["backward"][self.step:]}
        self._stack.update(forward=self._stack["forward"][:self.step],
                            backward=self._stack["backward"][:self.step])
        
        if self._callback:
            self._callback()
        
        return trailing
    
    def back(self):
        assert self.step > 0, self.step
        self._step -= 1
        self._stack["backward"][self.step]()
        
        if self._callback:
            self._callback()
        
        return self.step
    
    def forward(self):
        forward_stack = self._stack["forward"]
        assert self.step < len(forward_stack), (self.step, self._stack)
        forward_stack[self.step]()
        self._step += 1
        
        if self._callback:
            self._callback()
        
        return self.step
    
    def set_callback(self, func: Callable) -> Callable:
        assert callable(func), func
        self._callback = func
        return self._callback
    
    def get_callback(self) -> Callable | None:
        return self._callback
    
    def remove_callback(self) -> Callable | None:
        callback = self.get_callback()
        self._callback = None
        return callback


class Sheet(ttk.Frame):
    _valid_header_states = ('normal', 'hover', 'selected')
    _valid_cell_states = ('normal', 'readonly')
    _valid_scales = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0)
    
    @property
    def canvas(self) -> tk.Canvas:  # cell canvas
        return self._canvas
    
    @property
    def cornercanvas(self) -> tk.Canvas:
        return self._cornercanvas
    
    @property
    def rowcanvas(self) -> tk.Canvas:
        return self._rowcanvas
    
    @property
    def colcanvas(self) -> tk.Canvas:
        return self._colcanvas
    
    @property
    def hbar(self) -> AutoHiddenScrollbar:
        return self._hbar
    
    @property
    def vbar(self) -> AutoHiddenScrollbar:
        return self._vbar
    
    @property
    def values(self) -> pl.DataFrame:
        return self._values
    
    @property
    def shape(self) -> tuple:
        return self._values.shape
    
    def __init__(
            self,
            master,
            df: pl.DataFrame | None = None,
            shape: tuple[int, int] | list[int] = (10, 10),
            cell_width: int = 80,
            cell_height: int = 25,
            min_width: int = 20,
            min_height: int = 10,
            scale: float = 1.0,
            get_style: Callable | None = None,
            max_undo: int = 20,
            lock_number_of_rows: bool = False,
            lock_number_of_cols: bool = False,
            autohide_scrollbars: bool = True,
            scrollbar_bootstyle='round',
            scroll_sensitivities: float
                                  | tuple[float, float]
                                  | list[float, float] = 1.,
            **kw
    ):
        super().__init__(master)
        
        # Stacking order: CornerCanvas > ColCanvas > RowCanvas > CoverFrame
        # > Entry > CellCanvas
        top_left = {"row": 0, "column": 0}
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self._canvas = canvas = tk.Canvas(self, **kw)  # cell canvas
        self._canvas.grid(**top_left, sticky='nesw', rowspan=2, columnspan=2)
        self._rowcanvas = rowcanvas = tk.Canvas(self, **kw)
        self._rowcanvas.grid(**top_left, rowspan=2, sticky='nesw')
        self._colcanvas = colcanvas = tk.Canvas(self, **kw)
        self._colcanvas.grid(**top_left, columnspan=2, sticky='nesw')
        self._cornercanvas = cornercanvas = tk.Canvas(self, **kw)
        self._cornercanvas.grid(**top_left, sticky='nesw')
        
        self._hbar = AutoHiddenScrollbar(
            master=self,
            autohide=autohide_scrollbars,
            command=self.xview,
            bootstyle=scrollbar_bootstyle,
            orient='horizontal',
        )
        self._hbar.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(1, 0))
        self._vbar = AutoHiddenScrollbar(
            master=self,
            autohide=autohide_scrollbars,
            command=self.yview,
            bootstyle=scrollbar_bootstyle,
            orient='vertical',
        )
        self._vbar.grid(row=0, column=2, rowspan=2, sticky='ns', padx=(1, 0))
        self._cover = ttk.Frame(self)  # covers the entry widget
        self._cover.grid(row=2, column=2, sticky='nesw')  # right bottom corner
        self._cover.lift()
        self.set_scroll_sensitivities(scroll_sensitivities)
        
        # Create an invisible background (lowest item) which makes this sheet
        # become the focus if it is clicked
        for _canvas in [canvas, rowcanvas, colcanvas]:
            oid = _canvas.create_rectangle(
                0, 0, 0, 0, width=0, tags='invisible-bg')
            _canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
            _canvas.tag_bind('invisible-bg', '<Button-1>', self._focus)
        
        # Create the selection frame
        oid = canvas.create_rectangle(0, 0, 0, 0, fill='', tags='selection-frame')
        canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
        
        # Init the backend states
        self.set_states(
            shape=shape,
            cell_width=cell_width,
            cell_height=cell_height,
            min_width=min_width,
            min_height=min_height,
            scale=scale,
            get_style=get_style,
            max_undo=max_undo,
            lock_number_of_rows=lock_number_of_rows,
            lock_number_of_cols=lock_number_of_cols,
            draw=False
        )
        
        # Create an entry widget
        self._entry = entry = tk.Entry(
            self, textvariable=self._focus_value, takefocus=False)
        entry.place(x=0, y=0)
        entry.lower()
        entry.bind('<KeyPress>', self._on_entry_key_press)
        
        # Bindings
        self.bind('<<ThemeChanged>>', self._on_theme_changed)
        self.bind('<KeyPress>', self._on_key_press)
        self.bind('<<SelectAll>>', self._on_select_all)
        self.bind('<<Copy>>', self._on_copy)
        self.bind('<<Paste>>', self._on_paste)
        canvas.bind('<Configure>', self._on_canvas_configured)
        for widget in [canvas, rowcanvas, colcanvas, entry]:
            widget.configure(takefocus=False)
            for scrollseq in MOUSESCROLL:
                widget.bind(scrollseq, self._on_mousewheel_scroll)
        
        for _canvas in [canvas, cornercanvas, rowcanvas, colcanvas]:
            _canvas.bind('<ButtonPress-1>', self._on_leftbutton_press)
            _canvas.bind('<B1-Motion>', self._on_leftbutton_motion)
            _canvas.bind('<ButtonRelease-1>', self._on_leftbutton_release)
        canvas.bind('<Double-ButtonPress-1>', self._on_double_leftclick)
        
        # Update values
        if df is not None:
            self.paste_values(0, 0, df, draw=False)
        
        # Refresh the canvases and scrollbars
        self.refresh(trace='first')
        self.focus_set()
    
    def set_states(
            self,
            shape: tuple[int, int] | list[int] = (10, 10),
            cell_width: int = 80,
            cell_height: int = 25,
            min_width: int = 20,
            min_height: int = 10,
            scale: float = 1.0,
            get_style: Callable | None = None,
            max_undo: int = 20,
            lock_number_of_rows: bool = False,
            lock_number_of_cols: bool = False,
            draw: bool = True
    ):
        """
        Reset the cell sizes, table shape, contents, styles, and lock numbers.
        """
        assert len(shape) == 2 and all( s > 0 for s in shape ), shape
        assert cell_width >= 1 and cell_height >= 1, (cell_width, cell_height)
        assert min_width >= 1 and min_height >= 1, (min_width, min_height)
        assert get_style is None or callable(get_style), get_style
        
        shape = tuple( int(s) for s in shape )
        cell_width, cell_height = int(cell_width), int(cell_height)
        min_width, min_height = int(min_width), int(min_height)
        
        # Init the backend states
        if hasattr(self, '_history'):
            self._history.set_max_height(max_undo)
        else:
            self._history = History(max_height=max_undo)
        self._lock_number_of_rows: bool = bool(lock_number_of_rows)
        self._lock_number_of_cols: bool = bool(lock_number_of_cols)
        
        self._values = df_full(shape, value='', dtype=pl.String)
        self._cell_sizes = [
            np.full(shape[0] + 1, cell_height, dtype=float),
            np.full(shape[1] + 1, cell_width, dtype=float)
        ]
        self._cell_styles = np.array(
            [ [ dict() for c in range(shape[1]) ] for r in range(shape[0]) ],
            dtype=object
        )  # an array of dictionaries
        
        self._focus_old_value: str | None = None
        if not hasattr(self, '_focus_row'):
            self._focus_row = vrb.IntVar(self)
        if not hasattr(self, '_focus_col'):
            self._focus_col = vrb.IntVar(self)
        if not hasattr(self, '_focus_value'):
            self._focus_value = tk.StringVar(self)
        self._prev_scale: float = 1.0
        if not hasattr(self, '_scale'):
            self._scale = vrb.DoubleVar(self, value=scale)
            self._scale.trace_add('write', self._zoom, weak=True)
        else:
            self.zoom(scale)
        
        self._get_style = get_style
        self._default_styles = self._update_default_styles()
        self._default_cell_shape = shape
        self._default_cell_sizes = (cell_height, cell_width)
        self._min_sizes = (min_height, min_width)
        
        self.update_idletasks()
        self._canvas_size = (
            self._canvas.winfo_width(), self._canvas.winfo_height()
        )
        self._content_size = self._update_content_size()
        self._view = [(0, 0), (0, 0)]  # x view and y view in pixel
        gyx2s, visible_xys, visible_rcs = self._update_visible_and_p2s()
        self._visible_xys: list[tuple[int, int]] = visible_xys
        self._visible_rcs: list[tuple[int, int]] = visible_rcs
        self._gy2s_gx2s: tuple[np.ndarray, np.ndarray] = gyx2s
        
        self._selection_rcs: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._resize_start: dict | None = None
        self._hover: dict[str, str] | None = None
        self._motion_select_id: str | None = None
        
        if draw:
            self.refresh(trace='first')
    
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
        if self.hbar:
            self.hbar.autohide = enable
        if self.vbar:
            self.vbar.autohide = enable
        
        return (self.hbar.autohide, self.vbar.autohide)
    
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
    
    def __view(self, axis: int, *args):
        """Update the view from the canvas
        """
        assert axis in (0, 1), axis
        
        if not args:
            start, stop = self._view
            f1, f2 = self.__to_fraction(axis, start, stop)
            return max(f1, 0.), min(f2, 1.)
        
        action, args = args[0], args[1:]
        if action == 'moveto':
            return self.__view_moveto(axis, float(args[0]))
        elif action == 'scroll':
            return self.__view_scroll(axis, int(args[0]), args[1])
        raise ValueError("The first argument must be 'moveto' or 'scroll' but "
                         f"got: {repr(args[0])}")
    
    xview = lambda self, *args: self.__view(0, *args)
    yview = lambda self, *args: self.__view(1, *args)
    
    def __view_moveto(self, axis: int, fraction: float):
        """Move the view from the canvas
        """
        start = self.__to_pixel(axis, fraction)
        
        # Update the canvas and scrollbar
        self.__update_content_and_scrollbar(axis, start)
    
    xview_moveto = lambda self, *args, **kw: self.__view_moveto(0, *args, **kw)
    yview_moveto = lambda self, *args, **kw: self.__view_moveto(1, *args, **kw)
    
    def __view_scroll(self, axis: int, number: int, what: str):
        """Scroll the view from the canvas
        Note that the possible value "pixels" for `what` actually does not 
        magnify `number`. The value "pixels" for `what` is an additional, 
        convenient unit that is not included in the tkinter built in methods
        `xview_scroll` and `yview_scroll`
        """
        magnification = {"units": 10., "pages": 50., "pixels": 1.}[what]
        start, _ = self._view[axis]
        start += round(number * magnification)
        
        # Update widgets
        self.__update_content_and_scrollbar(axis, start)
    
    xview_scroll = lambda self, *args, **kw: self.__view_scroll(0, *args, **kw)
    yview_scroll = lambda self, *args, **kw: self.__view_scroll(1, *args, **kw)
    
    def __to_fraction(self,
                      axis: int,
                      *pixels) -> tuple[float, ...] | float:
        assert axis in (0, 1), axis
        complete = self._content_size[axis]
        fractions = tuple( pixel / complete for pixel in pixels )
        if len(fractions) == 1:
            return fractions[0]
        return fractions
    
    def __to_pixel(self,
                   axis: int,
                   *fractions) -> tuple[int, ...] | int:
        assert axis in (0, 1), axis
        complete = self._content_size[axis]
        pixels = tuple( round(fraction * complete) for fraction in fractions )
        if len(pixels) == 1:
            return pixels[0]
        return pixels
    
    def __confine_region(self, axis: int, start):
        complete = self._content_size[axis]
        showing = min(self._canvas_size[axis], complete)
        
        start, stop = (start, start + showing)
        
        if start < 0:
            start = 0
            stop = showing
        elif stop > complete:
            stop = complete
            start = stop - showing
        
        return start, stop
    
    def __update_content_and_scrollbar(self, axis: int, start: int):
        new_start, new_stop = self.__confine_region(axis, start)
        old_start, old_stop = self._view[axis]
        (r1_old, r2_old), (c1_old, c2_old) = self._visible_rcs
        self._view[axis] = (new_start, new_stop)
        *_, [(new_r1, new_r2), (new_c1, new_c2)] = self._update_visible_and_p2s()
        
        # Move xscrollable or yscrollable items
        delta_canvas = old_start - new_start  # -delta_view
        if axis == 0:
            key = "col"
            old_i1, old_i2, new_i1, new_i2 = (c1_old, c2_old, new_c1, new_c2)
            header_canvas = self.colcanvas
            
            self.canvas.move('xscroll', delta_canvas, 0)
            header_canvas.move('xscroll', delta_canvas, 0)
        else:
            key = "row"
            old_i1, old_i2, new_i1, new_i2 = (r1_old, r2_old, new_r1, new_r2)
            header_canvas = self.rowcanvas
            
            self.canvas.move('yscroll', 0, delta_canvas)
            header_canvas.move('yscroll', 0, delta_canvas)
        
        # Delete out-of-view items
        idc_out = set(range(old_i1, old_i2+1)) - set(range(new_i1, new_i2+1))
        tags_out = [ self._make_tag(key, row=i, col=i) for i in idc_out ]
        for tag in tags_out:
            for canvas in (self.canvas, header_canvas):
                canvas.delete(tag)
        
        # Draw new items
        self.draw(
            update_visible_rcs=False,
            skip_existing=key,   # reduce operating time
            trace=None
        )
        
        # Update x or y scrollbar
        first, last = self.__to_fraction(axis, new_start, new_stop)
        (self.hbar, self.vbar)[axis].set(first, last)
    
    def _update_default_styles(self) -> dict:
        def _scale_fonts(_default_styles:dict):
            scale = self._scale.get()
            for font in self._collect_fonts(_default_styles):
                font._unscaled_size = font.actual('size')
                font.configure(size=int(font._unscaled_size * scale))
        #
        
        if self._get_style:
            self._default_styles = default_styles = dict(self._get_style())
            assert isinstance(default_styles["header"]["font"], tk.font.Font)
            assert isinstance(default_styles["cell"]["font"], tk.font.Font)
            
            _scale_fonts(default_styles)
            
            return self._default_styles
        
        style = ttk.Style.get_instance()
        
        # Create some dummy widgets and get the ttk style name from them
        header = ttk.Checkbutton(self, bootstyle='primary-outline-toolbutton')
        header_style = header["style"]
        
        cell = ttk.Entry(self, bootstyle='secondary')
        cell_style = cell["style"]
        
        selection = ttk.Frame(self, bootstyle='primary')
        selection_style = selection["style"]
        
        # The ttkbootstrap styles of the header button above usually use the 
        # same color in both the button background and border. So we slightly 
        # modify the lightness of the border color to distinguish between them
        lighter_or_darker1 = lambda h, s, l: (h, s, l+10 if l < 50 else l-10)
        lighter_or_darker2 = lambda h, s, l: (h, s, l+15 if l < 50 else l-15)
        header_bg = style.lookup(header_style, "background")
        _, _, header_bg_l = color_to_hsl(header_bg)
        
        # Generate a dictionary to store the default styles
        self._default_styles = {
            "header": {
                "background": {
                    "normal": header_bg,
                    "hover": style.lookup(
                        header_style, "background", ('hover', '!disabled')),
                    "selected": modify_hsl(
                        style.lookup(
                            header_style,
                            "background",
                            ('selected', '!disabled')
                        ),
                        func=lambda h, s, l: (h, s, min(l+header_bg_l, 200) / 2)
                    ),
                },
                "foreground": {
                    "normal": style.lookup(header_style, "foreground"),
                    "hover": style.lookup(
                        header_style, "foreground", ('hover', '!disabled')),
                    "selected": style.lookup(
                        header_style, "foreground", ('selected', '!disabled'))
                },
                "bordercolor": {
                    "normal": modify_hsl(
                        style.lookup(header_style, "bordercolor"),
                        func=lighter_or_darker2
                    ),
                    "hover": style.lookup(
                        header_style,
                        "bordercolor",
                        ('hover', '!disabled')
                    ),
                    "selected": modify_hsl(
                        style.lookup(
                            header_style,
                            "bordercolor",
                            ('selected', '!disabled')
                        ),
                        func=lighter_or_darker1
                    )
                },
                "handle": {
                    "width": 1
                },
                "separator": {
                    "color": style.lookup(header_style, "bordercolor"),
                    "width": 2
                },
                "font": tk.font.nametofont('TkDefaultFont').copy(),
                "cursor": {
                    "default": 'arrow',
                    "hhandle": 'sb_v_double_arrow',
                    "vhandle": 'sb_h_double_arrow'
                }
            },
            "cell": {
                "background": {
                    "normal": style.lookup(cell_style, "fieldbackground"),
                    "readonly": style.lookup(cell_style, "foreground")
                },
                "foreground": {
                    "normal": style.lookup(cell_style, "foreground"),
                    "readonly": style.lookup(cell_style, "fieldbackground")
                },
                "bordercolor": {
                    "normal": style.lookup(cell_style, "bordercolor"),
                    "focus": style.lookup(selection_style, "background")
                },
                "font": tk.font.nametofont('TkDefaultFont').copy(),
                "alignx": 'w',   # w, e, or center
                "aligny": 'n',   # n, s, or center
                "padding": (3, 2)  # (padx, pady)
            },
            "selection": {
                "color": style.lookup(selection_style, "background"),
                "width": 3
            }
        }
        
        _scale_fonts(self._default_styles)
        
        # Release the resources
        header.destroy()
        cell.destroy()
        selection.destroy()
        
        return self._default_styles
    
    def _update_content_size(self) -> tuple:
        self._content_size = tuple(
            np.sum(self._cell_sizes[axis]) + 1 for axis in range(2)
        )[::-1]
        return self._content_size
    
    def _update_visible_and_p2s(self) -> tuple:
        heights, widths = self._cell_sizes
        (gx1_view, gx2_view), (gy1_view, gy2_view) = self._view
        gx1_vis, gy1_vis = (gx1_view + widths[0], gy1_view + heights[0])
        r12, c12, gy2s_gx2s = [None, None], [None, None], [None, None]
        for axis, [(gp1_vis, gp2_vis), i12] in enumerate(
                zip([(gy1_vis, gy2_view), (gx1_vis, gx2_view)], [r12, c12])):
            gy2s_gx2s[axis] = np.cumsum(self._cell_sizes[axis])
            gp2s = gy2s_gx2s[axis][1:]
            visible = (gp1_vis <= gp2s) & (gp2s <= gp2_vis)
            i12[0] = i1 = 0 if visible.all() else visible.argmax()
            i12[1] = len(visible) - 1 if (tail := visible[i1:]).all() \
                else tail.argmin() + i1
        
        self._gy2s_gx2s = tuple(gy2s_gx2s)  # (y2s_headers, x2s_headers)
        self._visible_xys = [(gx1_vis, gx2_view), (gy1_vis, gy2_view)]
        self._visible_rcs = [tuple(r12), tuple(c12)]  # [(r1, r2), (c1, c2)]
        
        return self._gy2s_gx2s, self._visible_xys, self._visible_rcs
    
    def _collect_fonts(self, dictionary: dict) -> list:
        fonts = []
        for value in dictionary.values():
            if isinstance(value, tk.font.Font):
                fonts.append(value)
            elif isinstance(value, dict):
                fonts.extend(self._collect_fonts(value))
        
        return fonts
    
    def _canvasx(self, xs: np.ndarray | list):
        header_width = self._cell_sizes[1][0]
        (gx1, gx2), (gy1, gy2) = self._visible_xys
        return np.asarray(xs) - gx1 + header_width  # => to canvas coordinates
    
    def _canvasy(self, ys: np.ndarray | list):
        header_height = self._cell_sizes[0][0]
        (gx1, gx2), (gy1, gy2) = self._visible_xys
        return np.asarray(ys) - gy1 + header_height  # => to canvas coordinates
    
    def _fit_size(self, text: str, font, width: int, height: int) -> str:
        width, height = (max(width, 0), max(height, 0))
        canvas = self.canvas
        lines = text.split('\n')
        oid = canvas.create_text(*self._canvas_size, text=text, font=font)
        canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
        x1, y1, x2, y2 = canvas.bbox(oid)
        canvas.delete(oid)
        longest_line = sorted(lines, key=lambda t: len(t))[-1]
        n_chars = int( len(longest_line) / (x2 - x1) * width )
        n_lines = int( len(lines) / (y2 - y1) * height )
        
        return '\n'.join( t[:n_chars] for t in lines[:n_lines] )
    
    def _focus(self, *_, **__):
        self._focus_out_cell()
        self.focus_set()
    
    def _center_window(self, toplevel: tk.BaseWidget):
        center_window(to_center=toplevel, center_of=self.winfo_toplevel())
    
    def _make_tags(
            self,
            oid=None,
            type_=None,
            subtype=None,
            row=None,
            col=None,
            others: tuple = tuple(),
            *,
            raw: bool = True,
            to_tuple: bool = False
    ) -> dict | tuple:
        assert isinstance(others, tuple), others
        
        tagdict = {
            "oid": f'oid={oid}',
            "type": f'type={type_}',
            "subtype": f'subtype={subtype}',
            "row": f'row={row}',
            "col": f'col={col}',
            "row:col": f'row:col={row}:{col}',
            "type:row": f'type:row={type_}:{row}',
            "type:col":f'type:col={type_}:{col}',
            "type:row:col": f'type:row:col={type_}:{row}:{col}',
            "type:subtype": f'type:subtype={type_}:{subtype}',
            "type:subtype:row": f'type:subtype:row={type_}:{subtype}:{row}',
            "type:subtype:col": f'type:subtype:col={type_}:{subtype}:{col}',
            "type:subtype:row:col":
                f'type:subtype:row:col={type_}:{subtype}:{row}:{col}',
        }
        
        if not raw:
            tagdict = {
                k: self._parse_raw_tag(k, v) for k, v in tagdict.items()
            }
        
        if to_tuple:
            return tuple(tagdict.values()) + others
        
        tagdict["others"] = others
        return tagdict
    
    def _make_tag(
            self,
            key: str,
            oid=None,
            type_=None,
            subtype=None,
            row=None,
            col=None,
            others: tuple = tuple(),
    ) -> str | tuple:
        assert isinstance(others, tuple), others
        
        if key == 'oid':
            return f'oid={oid}'
        elif key == 'type':
            return f'type={type_}'
        elif key == 'subtype':
            return f'subtype={subtype}'
        elif key == 'row':
            return f'row={row}'
        elif key == 'col':
            return f'col={col}'
        elif key == 'row:col':
            return f'row:col={row}:{col}'
        elif key == 'type:row':
            return f'type:row={type_}:{row}'
        elif key == 'type:col':
            return f'type:col={type_}:{col}'
        elif key == 'type:row:col':
            return f'type:row:col={type_}:{row}:{col}'
        elif key == 'type:subtype':
            return f'type:subtype={type_}:{subtype}'
        elif key == 'type:subtype:row':
            return f'type:subtype:row={type_}:{subtype}:{row}'
        elif key == 'type:subtype:col':
            return f'type:subtype:col={type_}:{subtype}:{col}'
        elif key == 'type:subtype:row:col':
            return f'type:subtype:row:col={type_}:{subtype}:{row}:{col}'
        elif key == 'others':
            return others
        else:
            raise ValueError(f"Invalid `key`: {key}.")
    
    def _get_tags(
            self,
            oid: int | str,
            raw: bool = False,
            to_tuple: bool = False,
            canvas: tk.Canvas = None,
            _key: str | None = None
    ) -> dict | tuple:
        assert isinstance(oid, (int, str)), oid
        assert isinstance(canvas, (type(None), tk.Canvas)), canvas
        
        canvas = canvas or self.canvas
        
        others = tuple()
        tagdict = dict.fromkeys(
            [
                "oid",
                "type", "subtype",
                "row", "col", "row:col",
                "type:row", "type:col",
                "type:subtype",
                "type:subtype:row", "type:subtype:col",
                "type:subtype:row:col"
            ],
            None
        )
        
        tags = canvas.gettags(oid)
        for tag in tags:
            try:
                key, value = tag.split('=')
            except ValueError:
                others += (tag,)
            else:
                if key in tagdict:
                    if _key is None:  # found a tag
                        tagdict[key] = tag
                    elif _key == key:  # found the requested tag
                        tagdict = {_key: tag}
                        break
        
        if _key == 'others':
            return {_key: others}
        
        if not raw:
            tagdict = {
                k: self._parse_raw_tag(k, v) for k, v in tagdict.items()
            }
        
        if to_tuple:
            return tuple(tagdict.values()) + others
        
        tagdict["others"] = others
        return tagdict
    
    def _get_tag(
            self, key: str, oid: int | str, canvas: tk.Canvas = None
    ) -> str | tuple | None:
        return self._get_tags(oid=oid, canvas=canvas, _key=key)[key]
    
    def _parse_raw_tag(self, key: str, tag: str | None) -> str:
        if tag is None or (_tag := tag.split('=', 1)[1]) == 'None':
            return None
        elif key in ('oid', 'row', 'col'):
            return int(_tag)
        return _tag
    
    def _get_rc(
            self,
            oid_or_tagdict: int | dict | str,
            to_tuple: bool = False,
            canvas: tk.Canvas = None
    ) -> dict | tuple:
        assert isinstance(oid_or_tagdict, (int, dict, str)), oid_or_tagdict
        
        if isinstance(oid_or_tagdict, dict):
            tagdict = oid_or_tagdict
        else:
            tagdict = self._get_tags(oid_or_tagdict, canvas=canvas)
        
        if to_tuple:
            return (tagdict["row"], tagdict["col"])
        return {"row": tagdict["row"], "col": tagdict["col"]}
    
    def _get_rcs(self, tag: str) -> dict:
        return dict(zip(
            ["rows", "cols"],
            map(
                set,
                zip(*[ self._get_rc(oid, to_tuple=True)
                       for oid in self.canvas.find_withtag(tag) ])
            )
        ))
    
    def _build_general_rightclick_menu(
            self, menu: tk.Menu | None = None) -> tk.Menu:
        menu = menu or tk.Menu(self, tearoff=0)
        
        # Manipulate values in cells
        menu.add_command(
            label='Erase Value(s)',
            command=lambda: self._selection_erase_values(undo=True)
        )
        menu.add_command(
            label='Copy Value(s)',
            command=self._selection_copy_values
        )
        menu.add_command(
            label='Paste Value(s)',
            command=lambda: self._selection_paste_values(undo=True)
        )
        menu.add_separator()
        
        # Change text colors
        menu_textcolor = tk.Menu(menu, tearoff=0)
        menu_textcolor.add_command(
            label='Choose Color...',
            command=lambda: self._selection_set_foregroundcolors(
                dialog=True, undo=True)
        )
        menu_textcolor.add_command(
            label='Reset Color(s)',
            command=lambda: self._selection_set_foregroundcolors(undo=True)
        )
        menu.add_cascade(label='Text Color(s)', menu=menu_textcolor)
        
        # Change background colors
        menu_bgcolor = tk.Menu(menu, tearoff=0)
        menu_bgcolor.add_command(
            label='Choose Color...',
            command=lambda: self._selection_set_backgroundcolors(
                dialog=True, undo=True)
        )
        menu_bgcolor.add_command(
            label='Reset Color(s)',
            command=lambda: self._selection_set_backgroundcolors(undo=True)
        )
        menu.add_cascade(label='Background Color(s)', menu=menu_bgcolor)
        
        # Change fonts
        menu_font = tk.Menu(menu, tearoff=0)
        menu_font.add_command(
            label='Choose Font...',
            command=lambda: self._selection_set_fonts(dialog=True, undo=True)
        )
        menu_font.add_command(
            label='Reset Font(s)',
            command=lambda: self._selection_set_fonts(undo=True)
        )
        menu.add_cascade(label='Font(s)', menu=menu_font)
        
        # Change text alignments
        menu_align = tk.Menu(menu, tearoff=0)
        menu_align.add_command(
            label='← Left',
            command=lambda: self._selection_set_styles(
                "alignx", 'w', undo=True)
        )
        menu_align.add_command(
            label='→ Right',
            command=lambda: self._selection_set_styles(
                "alignx", 'e', undo=True)
        )
        menu_align.add_command(
            label='⏐ Center',
            command=lambda: self._selection_set_styles(
                "alignx", 'center', undo=True)
        )
        menu_align.add_command(
            label='⤬ Reset',
            command=lambda: self._selection_set_styles(
                "alignx", None, undo=True)
        )
        menu_align.add_separator()
        menu_align.add_command(
            label='↑ Top',
            command=lambda: self._selection_set_styles(
                "aligny", 'n', undo=True)
        )
        menu_align.add_command(
            label='↓ Bottom',
            command=lambda: self._selection_set_styles(
                "aligny", 's', undo=True)
        )
        menu_align.add_command(
            label='⎯ Center',
            command=lambda: self._selection_set_styles(
                "aligny", 'center', undo=True)
        )
        menu_align.add_command(
            label='⤬ Reset',
            command=lambda: self._selection_set_styles(
                "aligny", None, undo=True)
        )
        menu.add_cascade(label='Align', menu=menu_align)
        
        # Reset styles
        menu.add_command(
            label='Reset Style(s)',
            command=lambda: self._selection_reset_styles(undo=True)
        )
        
        return menu
    
    def _redirect_widget_event(self, event) -> tk.Event:  # currently not used
        widget, canvas = (event.widget, self.canvas)
        event.x += widget.winfo_x() - canvas.winfo_x()
        event.y += widget.winfo_y() - canvas.winfo_y()
        event.widget = canvas
        return event
    
    def _on_theme_changed(self, event=None):
        self._update_default_styles()
        self.refresh()
    
    def _on_canvas_configured(self, event):
        self._canvas_size = canvas_size = (event.width, event.height)
        self.canvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.rowcanvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.colcanvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.xview_scroll(0, 'units')
        self.yview_scroll(0, 'units')
    
    def _on_mousewheel_scroll(self, event):
        """Callback for when the mouse wheel is scrolled.
        Modified from: `ttkbootstrap.scrolled.ScrolledFrame._on_mousewheel`
        """
        if event.num == 4:  # Linux
            delta = 10.
        elif event.num == 5:  # Linux
            delta = -10.
        elif self._windowingsystem == "win32":  # Windows
            delta = event.delta / 120.
        else:  # Mac
            delta = event.delta
        
        x_direction = (event.state & MODIFIER_MASKS["Shift"]) \
            == MODIFIER_MASKS["Shift"]
        sensitivity = self._scroll_sensitivities[0 if x_direction else 1]
        number = -round(delta * sensitivity * 2.)
        
        if x_direction:
            self.xview_scroll(number, 'units')
        else:
            self.yview_scroll(number, 'units')
    
    def _on_select_all(self, event=None):
        self.select_cells()
    
    def _on_copy(self, event=None):
        self._selection_copy_values()
    
    def _on_paste(self, event=None):
        self._selection_paste_values(undo=True)
    
    def _on_entry_key_press(self, event) -> str | None:
        keysym = event.keysym
        modifiers = get_modifiers(event.state)
        
        if (keysym in ('z', 'Z')) and (COMMAND in modifiers):
            self.undo()
            return 'break'
        
        elif (keysym in ('y', 'Y')) and (COMMAND in modifiers):
            self.redo()
            return 'break'
        
        elif keysym in ('minus', 'equal') and COMMAND in modifiers:
            direction = 'down' if keysym == 'minus' else 'up'
            self._zoom_to_next_scale_level(direction)
            return 'break'
        
        elif keysym in ('Return', 'Tab'):
            if keysym == 'Return':
                direction = 'up' if SHIFT in modifiers else 'down'
            else:
                direction = 'left' if SHIFT in modifiers else 'right'
            self._move_selections(direction)
            return 'break'
        
        elif keysym == 'Escape':
            self._focus_out_cell(discard=True)
            return 'break'
    
    def _on_key_press(self, event) -> str | None:
        keysym, char = event.keysym, event.char
        modifiers = get_modifiers(event.state)
        
        if self._on_entry_key_press(event):  # same as entry events
            return 'break'
        
        elif keysym in ('Up', 'Down', 'Left', 'Right'):  # move selection
            direction = keysym.lower()
            area = 'paragraph' if COMMAND in modifiers else None
            expand = SHIFT in modifiers
            self._move_selections(direction, area=area, expand=expand)
            return 'break'
        
        elif keysym in ('Home', 'End', 'Prior', 'Next'):  # move selection
            direction = {
                "Home": 'left',
                "End": 'right',
                "Prior": 'up',
                "Next": 'down'
            }[keysym]
            expand = SHIFT in modifiers
            self._move_selections(direction, area='all', expand=expand)
            return 'break'
        
        elif keysym == 'Delete':  # delete all characters in the selected cells
            self._selection_erase_values(undo=True)
            return 'break'
        
        elif (MODIFIERS.isdisjoint(modifiers) and keysym == 'BackSpace') or (
                not modifiers - {SHIFT, LOCK} and char):
            # Normal typing
            self._focus_in_cell()
            self._entry.delete(0, 'end')
            self._entry.insert('end', char)
            return 'break'
    
    def __mouse_select(
            self, x, y, canvas, expand: bool, dry: bool = False) -> tuple:
        heights, widths = self._cell_sizes
        gy2s, gx2s = self._gy2s_gx2s
        x2s, y2s = (self._canvasx(gx2s[1:]), self._canvasy(gy2s[1:]))
        (r1_vis, r2_vis), (c1_vis, c2_vis) = self._visible_rcs
        
        if canvas in (self.cornercanvas, self.colcanvas):
            r2 = None
        else:
            r2 = np.clip(
                above.argmax() if (above := y <= y2s).any() else y2s.size - 1,
                r1_vis,
                r2_vis
            )
        if canvas in (self.cornercanvas, self.rowcanvas):
            c2 = None
        else:
            c2 = np.clip(
                left.argmax() if (left := x <= x2s).any() else x2s.size - 1,
                c1_vis,
                c2_vis
            )
        
        if expand:
            r1, c1, *_ = self._selection_rcs
        else:
            r1, c1 = (r2, c2)
        
        if not dry:
            self.select_cells(r1, c1, r2, c2)
        
        return (r1, c1, r2, c2)
    
    def _on_leftbutton_press(self, event, select: bool = True):
        self._focus()
        x, y, canvas = (event.x, event.y, event.widget)
        return self.__mouse_select(x, y, canvas, expand=False, dry=not select)
    
    def _on_leftbutton_motion(self, event, _dxdy: tuple | None = None):
        # Move the viewing window if the mouse cursor is moving outside the 
        # canvas
        x, y, canvas = (event.x, event.y, event.widget)
        if _dxdy is None:
            heights, widths = self._cell_sizes
            top_bd, left_bd = (heights[0], widths[0])  # headers' bottom/right
            right_bd, bottom_bd = self._canvas_size
            dx = (x - left_bd if x < left_bd else max(x - right_bd, 0)) / 2.
            dy = (y - top_bd if y < top_bd else max(y - bottom_bd, 0)) / 8.
        else:
            dx, dy = _dxdy
        
        if dx != 0:
            self.xview_scroll(dx, 'pixels')
        if dy != 0:
            self.yview_scroll(dy, 'pixels')
        self.__mouse_select(x - dx, y - dy, canvas, expand=True)
        
        # Cancel the old autoscroll function loop and then setup a new one
        # This function loop will autoscroll the canvas with the (dx, dy) above. 
        # This makes the user, once the first motion event has been triggered, 
        # don't need to continue moving the mouse to trigger the motion events
        if (funcid := self._motion_select_id) is not None:
            self.after_cancel(funcid)
        self._motion_select_id = self.after(  # about 30 FPS
            33, self._on_leftbutton_motion, event, (dx, dy)
        )
    
    def _on_leftbutton_release(self, event=None):
        # Remove the autoscroll function loop
        if (funcid := self._motion_select_id) is not None:
            self.after_cancel(funcid)
            self._motion_select_id = None
    
    def _on_double_leftclick(self, event=None):
        assert event.widget == self.canvas, event.widget
        self._focus_in_cell()
    
    def _index_generator(
            self, start: int | np.integer, last: int | np.integer
    ) -> Generator[int, bool, None]:
        """
        Create an index generator which yields an index each time after
        `.send(skip_last_index)` is called. The `.send` method will return the
        next index if the generator hasn't received twice `False` values to
        the argument `skip_last_index`.
        
        We assume the indices which should be skipped is contained in a block,
        i.e. they are contiguous. So we divide all the possible indices into
        two halves, revert the second half, and interleave the first half with
        the reverted second half. Once we find the two skipped indices at the
        edges, we stop the iteration. This can accelerate the looping by not
        generating the indices which should be skipped. For example, if `start`
        is 1, `last` is 10, and the indices which should be skipped are [2, 3,
        4, 5, 6], The yielded indices will be [1, 10, 2, 9, 8, 7, 6].
        
        Parameters
        ----------
        start : int | np.integer
            This is the smallest index value. This will be the first value
            yielded by the generator.
        last : int | np.integer
            This is the largest index value. This will be the second value
            yielded by the generator.
        
        Yields
        ------
        Generator[int, bool, None]
            The created index generator. Use `next(generator)` to get the first
            index. And use `generator.send(skip_last_index)` to get the
            subsequent indices.
        """
        assert isinstance(start, (int, np.integer)), (type(start), start)
        assert isinstance(last, (int, np.integer)), (type(last), last)
        assert 0 <= start <= last, (start, last)
        
        n_half = (start + last) // 2
        upper_idc = list(range(start, n_half+1, +1))
        lower_idc = list(range(last, n_half, -1))
        indices = [None] * (len(upper_idc) + len(lower_idc))
        indices[0::2] = upper_idc
        indices[1::2] = lower_idc
        
        start_skip: bool = False
        for i, idx in enumerate(indices):
            start_skip = yield idx
            if start_skip:
                break
        else:
            return
        
        # Start to skip from current `idx`
        # => now find the end index where skip ends
        if i % 2 == 0:  # current `idx` is in `upper_idc`
            indices = lower_idc[i//2:] + upper_idc[i//2+1:][::-1]
        else:  # current `idx` is in `lower_idc`
            indices = upper_idc[(i+1)//2:] + lower_idc[(i+1)//2+1:][::-1]
        
        for idx in indices:
            end_skip = yield idx
            if end_skip:
                break
    
    def draw_cornerheader(self, skip_existing: bool = False):
        assert isinstance(skip_existing, bool), skip_existing
        
        type_ = 'cornerheader'
        x1, y1, x2, y2 = (0, 0, self._cell_sizes[1][0], self._cell_sizes[0][0])
        width, height = (x2 - x1, y2 - y1)
        
        style = self._default_styles["header"]
        background, bordercolor = style["background"], style["bordercolor"]
        handle = style["handle"]
        separator = style["separator"]
        dw_sep = separator["width"] // 2
        canvas = self.cornercanvas
        canvas.configure(width=x2 - x1, height=y2 - y1)
        self.rowcanvas.configure(width=width)
        self.colcanvas.configure(height=height)
        
        tag = self._make_tag("type", type_=type_)
        if skip_existing:
            for oid in canvas.find_withtag(tag):
                subtype = self._get_tag("subtype", oid, canvas=canvas)
                if subtype in ('background', 'hhandle', 'vhandle'):
                    return
        
        # Delete the existing components
        canvas.delete(tag)
        
        # Draw components for the cornerheader
        kw = {
            "type_": type_, 
            "row": -1, "col": -1,
            "others": ('temp',),
            "to_tuple": True
        }
        
        ## Background
        tags = self._make_tags(subtype='background', **kw)
        oid = canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=background["normal"],
            outline=bordercolor["normal"],
            tags=tags
        )
        canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
        
        ## Handles (invisible)
        tags = self._make_tags(subtype='hhandle', **kw)
        oid_hh = canvas.create_line(
            x1, y2, x2, y2, width=handle["width"], fill='', tags=tags)
        canvas.addtag_withtag(self._make_tag("oid", oid=oid_hh), oid_hh)
        
        tags = self._make_tags(subtype='vhandle', **kw)
        oid_vh = canvas.create_line(
            x2, y1, x2, y2, width=handle["width"], fill='', tags=tags)
        canvas.addtag_withtag(self._make_tag("oid", oid=oid_vh), oid_vh)
        
        ## Separators (always redrawn)
        ### Horizontal
        kw.pop("col")
        tags = self._make_tags(subtype='separator', **kw)
        oid = canvas.create_line(
            x1, y2-dw_sep, x2, y2-dw_sep,
            width=separator["width"],
            fill=separator["color"],
            tags=tags
        )
        canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
        
        ### Vertical
        kw.pop("row")
        kw["col"] = -1
        tags = self._make_tags(subtype='separator', **kw)
        oid = canvas.create_line(
            x2-dw_sep, y1, x2-dw_sep, y2,
            width=separator["width"],
            fill=separator["color"],
            tags=tags
        )
        canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
        
        ## Handles > separators > background
        canvas.tag_raise(oid_hh)
        canvas.tag_raise(oid_vh)
        
        # Bindings
        tag_cornerheader = self._make_tag("type", type_=type_)
        canvas.tag_bind(tag_cornerheader, '<Enter>', self._on_header_enter)
        canvas.tag_bind(tag_cornerheader, '<Leave>', self._on_header_leave)
        canvas.tag_bind(
            tag_cornerheader, RIGHTCLICK, self._on_header_rightbutton_press)
        
        for handle in ['hhandle', 'vhandle']:
            tag_cornerhandle = self._make_tag(
                "type:subtype", type_=type_, subtype=handle)
            canvas.tag_bind(
                tag_cornerhandle,
                '<ButtonPress-1>',
                getattr(self, f'_on_{handle}_leftbutton_press')
            )
    
    def draw_headers(
            self,
            i1: int | None = None,
            i2: int | None = None,
            *,
            axis: int,
            skip_existing: bool = False
    ):
        axis = int(axis)
        assert (i1 is not None) or (i2 is None), (i1, i2)
        assert axis in (0, 1), axis
        assert isinstance(skip_existing, bool), skip_existing
        
        r12_vis, c12_vis = self._visible_rcs
        i1_vis, i2_vis = r12_vis if axis == 0 else c12_vis
        
        if i1 is None:
            i1, i2 = (i1_vis, i2_vis)
        elif i2 is None:
            i2 = i2_vis
        i1, i2 = sorted([ np.clip(i, i1_vis, i2_vis) for i in (i1, i2) ])
        
        max_i = self.shape[axis] - 1
        assert 0 <= i1 <= i2 <= max_i, (i1, i2, max_i)
        
        style = self._default_styles["header"]
        background, foreground = style["background"], style["foreground"]
        bordercolor, font = style["bordercolor"], style["font"]
        w_hand = style["handle"]["width"]
        separator = style["separator"]
        dw_sep = separator["width"] // 2
        
        (gx1_vis, gx2_vis), (gy1_vis, gy2_vis) = self._visible_xys
        heights, widths = self._cell_sizes
        if axis == 0:
            type_, prefix, handle = ('rowheader', 'R', 'hhandle')
            x1, x2 = (0, widths[0])
            y2s = self._canvasy(self._gy2s_gx2s[axis][1:])
            y1s = y2s - heights[1:]
            properties = [
                (
                    {
                        "type_": type_,
                        "row": r,
                        "col": -1,
                        "others": ('yscroll', 'temp')
                    },
                    (x1, y1, x2, y2),  # rectangle
                    (x1, y2, x2, y2)  # horizontal
                )
                for r, (y1, y2) in enumerate(zip(y1s[i1:i2+1], y2s[i1:i2+1]), i1)
            ]
            xys_sep = (x2-dw_sep, y1s[i1], x2-dw_sep, y2s[i2])  # vertical
            canvas = self.rowcanvas
        else:
            type_, prefix, handle = ('colheader', 'C', 'vhandle')
            y1, y2 = (0, heights[0])
            x2s = self._canvasx(self._gy2s_gx2s[axis][1:])
            x1s = x2s - widths[1:]
            properties = [
                (
                    {
                        "type_": type_,
                        "row": -1,
                        "col": c,
                        "others": ('xscroll', 'temp')
                    },
                    (x1, y1, x2, y2),  # rectangle
                    (x2, y1, x2, y2)  # vertical
                )
                for c, (x1, x2) in enumerate(zip(x1s[i1:i2+1], x2s[i1:i2+1]), i1)
            ]
            xys_sep = (x1s[i1], y2-dw_sep, x2s[i2], y2-dw_sep)  # horizontal
            canvas = self.colcanvas
        
        # Draw components for each header
        i_generator = self._index_generator(i1, i2)
        i = next(i_generator)
        while True:
            kw, (x1, y1, x2, y2), xys_handle = properties[i-i1]
            tag = self._make_tag("type:row:col", **kw)
            
            if skip_existing and canvas.find_withtag(tag):  # skip this row/col
                try:
                    i = i_generator.send(True)  # get next `i`
                    continue  # got to next row/col
                except StopIteration:
                    break  # stop this iteration
            
            ## Delete the existing components
            canvas.delete(tag)
            
            kw["to_tuple"] = True
            
            ## Create new components
            ### Background
            tags = self._make_tags(subtype='background', **kw)
            oid = canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=background["normal"],
                outline=bordercolor["normal"],
                tags=tags
            )
            canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
            
            ### Text
            tags = self._make_tags(subtype='text', **kw)
            text = self._fit_size(
                f'{prefix}{i}', font, width=x2 - x1, height=y2 - y1)
            oid = canvas.create_text(
                (x1 + x2) // 2, (y1 + y2) // 2,
                text=text,
                font=font,
                fill=foreground["normal"],
                tags=tags
            )
            canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
            
            ### Handle (invisible)
            tags = self._make_tags(subtype=handle, **kw)
            oid = canvas.create_line(
                *xys_handle, width=w_hand, fill='', tags=tags)
            canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
            
            try:
                i = i_generator.send(False)  # get next `i`
            except StopIteration:
                break  # stop this iteration
        
        # Separator (always redrawn)
        kw.pop("to_tuple", None)
        kw.pop("row" if axis == 0 else "col")
        tag = self._make_tag("type:subtype", subtype='separator', **kw)
        canvas.delete(tag)
        tags = self._make_tags(subtype='separator', to_tuple=True, **kw)
        oid_sep = canvas.create_line(
            *xys_sep,
            width=separator["width"],
            fill=separator["color"],
            tags=tags
        )
        canvas.addtag_withtag(self._make_tag("oid", oid=oid_sep), oid_sep)
        
        # Stacking order: CornerHeader > Row/ColSeparator > Row/ColHeaderHandle
        # > Row/ColHeader
        tag_bg = self._make_tag("type:subtype", type_=type_, subtype='background')
        tag_text = self._make_tag("type:subtype", type_=type_, subtype='text')
        tag_handle = self._make_tag("type:subtype", type_=type_, subtype=handle)
        canvas.tag_raise(tag_bg)
        canvas.tag_raise(tag_text)
        canvas.tag_raise(tag_handle)  # second from the front
        canvas.tag_raise(oid_sep)  # frontmost
        
        # Bindings
        tag_header = self._make_tag("type", type_=type_)
        canvas.tag_bind(tag_header, '<Enter>', self._on_header_enter)
        canvas.tag_bind(tag_header, '<Leave>', self._on_header_leave)
        canvas.tag_bind(tag_header, RIGHTCLICK, self._on_header_rightbutton_press)
        canvas.tag_bind(
            tag_handle,
            '<ButtonPress-1>',
            getattr(self, f'_on_{handle}_leftbutton_press')
        )
    
    def _set_header_state(
            self,
            tagdict: dict,
            state: Literal[_valid_header_states]
    ):
        assert isinstance(tagdict, dict), tagdict
        assert state is None or state in self._valid_header_states, state
        
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), type_
        
        if type_ == 'cornerheader':
            canvas = self.cornercanvas
            tag = self._make_tag("type", type_=type_)
            tag_bg = self._make_tag(
                "type:subtype", type_=type_, subtype='background')
            tag_text = None
        else:
            if type_ == 'rowheader':
                col_or_row = "row"
                canvas = self.rowcanvas
            else:
                col_or_row = "col"
                canvas = self.colcanvas
            
            kw = {"type_": type_, col_or_row: tagdict[col_or_row]}
            tag = self._make_tag(f"type:{col_or_row}", **kw)
            tag_bg = self._make_tag(
                f"type:subtype:{col_or_row}", subtype='background', **kw)
            tag_text = self._make_tag(
                f"type:subtype:{col_or_row}", subtype='text', **kw)
        
        if not canvas.find_withtag(tag):  # items have not been created yet
            return
        
        style = self._default_styles["header"]
        
        # Update the background color and border color
        canvas.itemconfigure(
            tag_bg,
            fill=style["background"][state],
            outline=style["bordercolor"][state]
        )
        
        if type_ != 'cornerheader':
            # Update the stacking order
            ## Front to back: separator > handlers > texts > backgrounds
            ## > invisible-bg
            if state in ('selected', 'hover'):
                tag_all_text = self._make_tag(
                    "type:subtype", type_=type_, subtype='text')
                canvas.tag_lower(tag_bg, tag_all_text)  # frontmost background
            else:
                try:
                    # Lower than the selected
                    canvas.tag_lower(tag_bg, 'selected')
                except tk.TclError:
                    pass  # the selected not in the view
            
            # Update the text color
            canvas.itemconfigure(tag_text, fill=style["foreground"][state])
    
    def __on_header_enter_leave(self, event, enter_or_leave: str):
        # Set header state
        canvas = event.widget
        tagdict = self._get_tags('current', canvas=canvas)
        if tagdict["subtype"] != 'separator':
            if enter_or_leave == 'enter':  # hover on
                state = 'hover'
            elif 'selected' in tagdict["others"]:  # hover off, selected
                state = 'selected'
            else:  # hover off, not selected
                state = 'normal'
            self._set_header_state(tagdict, state=state)
        
        # Set mouse cursor style
        cursor_styles = self._default_styles["header"]["cursor"]
        if enter_or_leave == 'enter':
            subtype = tagdict.get("subtype", None)
            cursor = cursor_styles.get(subtype, cursor_styles["default"])
        else:
            cursor = cursor_styles["default"]
        canvas.configure(cursor=cursor)
        
        self._hover = tagdict if enter_or_leave == 'enter' else None
    
    _on_header_enter = lambda self, event: self.__on_header_enter_leave(
        event, 'enter')
    _on_header_leave = lambda self, event: self.__on_header_enter_leave(
        event, 'leave')
    
    def _on_header_rightbutton_press(self, event):
        tagdict = self._hover
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), self._hover
        
        # Select the current row/col if it is not selected
        _r1, _c1, _r2, _c2 = self._selection_rcs
        (r_low, r_high), (c_low, c_high) = sorted([_r1, _r2]), sorted([_c1, _c2])
        r1, c1, r2, c2 = self._on_leftbutton_press(event, select=False)
        r_max, c_max = [ s - 1 for s in self.shape ]
        
        if type_ == 'rowheader':
            if (c_low, c_high) != (0, c_max) or not r_low <= r1 <= r_high:
                self.select_cells(r1, c1, r2, c2)
            
            axis_name, axis = ('Row', 0)
            modifiable_nheaders = 'disabled' if self._lock_number_of_rows \
                                  else 'active'
        elif type_ == 'colheader':
            if (r_low, r_high) != (0, r_max) or not c_low <= c1 <= c_high:
                self.select_cells(r1, c1, r2, c2)
            
            axis_name, axis = ('Column', 1)
            modifiable_nheaders = 'disabled' if self._lock_number_of_cols \
                                  else 'active'
        else:
            self.select_cells(r1, c1, r2, c2)
            
            axis_name, axis = ('Row', 0)
            modifiable_nheaders = 'active'
        
        # Setup the right click menu
        menu = tk.Menu(self, tearoff=0)
        
        if type_ in ('rowheader', 'colheader'):
            menu.add_command(
                label=f'Insert New {axis_name}s Ahead',
                command=lambda: self._selection_insert_cells(
                    axis, mode='ahead', dialog=True, undo=True),
                state=modifiable_nheaders
            )
            menu.add_command(
                label=f'Insert New {axis_name}s Behind',
                command=lambda: self._selection_insert_cells(
                    axis, mode='behind', dialog=True, undo=True),
                state=modifiable_nheaders
            )
        menu.add_command(
            label=f'Delete Selected {axis_name}(s)',
            command=lambda: self._selection_delete_cells(undo=True),
            state=modifiable_nheaders
        )
        menu.add_separator()
        if type_ in ('cornerheader', 'rowheader'):
            menu_height = tk.Menu(menu, tearoff=0)
            menu_height.add_command(
                label='Set Height...',
                command=lambda: self._selection_resize_cells(
                    axis=0, dialog=True, undo=True)
            )
            menu_height.add_command(
                label='Reset Height(s)',
                command=lambda: self._selection_resize_cells(axis=0, undo=True)
            )
            menu.add_cascade(label="Rows' Height(s)", menu=menu_height)
        if type_ in ('cornerheader', 'colheader'):
            menu_width = tk.Menu(menu, tearoff=0)
            menu_width.add_command(
                label='Set Width...',
                command=lambda: self._selection_resize_cells(
                    axis=1, dialog=True, undo=True)
            )
            menu_width.add_command(
                label='Reset Width(s)',
                command=lambda: self._selection_resize_cells(axis=1, undo=True)
            )
            menu.add_cascade(label="Columns' Width(s)", menu=menu_width)
        menu.add_separator()
        self._build_general_rightclick_menu(menu)
        
        menu.post(event.x_root, event.y_root)
        self.after_idle(menu.destroy)
    
    def __on_handle_leftbutton_press(self, event, axis: int):  # resize starts
        tagdict = self._hover
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), self._hover
        assert axis in (0, 1), axis
        
        r, c = self._get_rc(tagdict, to_tuple=True)
        
        # Bind (overwrite) the event functions with the handle callbacks
        canvas = event.widget
        old_b1motion = canvas.bind('<B1-Motion>')
        old_b1release = canvas.bind('<ButtonRelease-1>')
        if axis == 0:
            i = r
            rcs = (r, None, r, None)
            canvas.bind('<B1-Motion>', self._on_hhandle_leftbutton_motion)
            canvas.bind('<ButtonRelease-1>', self._on_handle_leftbutton_release)
        else:
            i = c
            rcs = (None, c, None, c)
            canvas.bind('<B1-Motion>', self._on_vhandle_leftbutton_motion)
            canvas.bind('<ButtonRelease-1>', self._on_handle_leftbutton_release)
        self.after_idle(self.select_cells, *rcs)
        
        _i = i + 1
        self._resize_start = {
            "x": event.x,
            "y": event.y,
            "i": i,
            "size": self._cell_sizes[axis][_i],  # scaled size
            "scale": self._scale.get(),
            "step": self._history.step,
            "b1motion": old_b1motion,
            "b1release": old_b1release
        }
    
    _on_hhandle_leftbutton_press = lambda self, event: (
        self.__on_handle_leftbutton_press(event, axis=0))
    _on_vhandle_leftbutton_press = lambda self, event: (
        self.__on_handle_leftbutton_press(event, axis=1))
    
    def __on_handle_leftbutton_motion(self, event, axis: int):  # resizing
        start = self._resize_start
        scale = start["scale"]
        if axis == 0:
            size = start["size"] + event.y - start["y"]
        else:
            size = start["size"] + event.x - start["x"]
        size /= scale  # scaled size => unscaled size
        start_size = start["size"] / scale  # scaled size => unscaled size
        self.resize_cells(start["i"], axis=axis, N=1, sizes=[size], undo=False)
        
        history = self._history
        if history.step > start["step"]:
            history.pop()
        history.add(
            forward=lambda: self.resize_cells(
                start["i"], axis=axis, N=1, sizes=[size], trace='first'),
            backward=lambda: self.resize_cells(
                start["i"], axis=axis, sizes=[start_size], trace='first')
        )
    
    _on_hhandle_leftbutton_motion = lambda self, event: (
        self.__on_handle_leftbutton_motion(event, axis=0))
    _on_vhandle_leftbutton_motion = lambda self, event: (
        self.__on_handle_leftbutton_motion(event, axis=1))
    
    def _on_handle_leftbutton_release(self, event):  # resize ends
        # Overwrite the handle callbacks with the originals
        start = self._resize_start
        event.widget.bind('<B1-Motion>', start["b1motion"])
        event.widget.bind('<ButtonRelease-1>', start["b1release"])
        self._resize_start = None
    
    def draw_cells(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            skip_existing: str | None = None,
    ):
        assert (r1 is not None) or (r2 is None), (r1, r2)
        assert (c1 is not None) or (c2 is None), (c1, c2)
        assert skip_existing in ('row', 'col', None), skip_existing
        
        (r1_vis, r2_vis), (c1_vis, c2_vis) = self._visible_rcs
        gy2s, gx2s = self._gy2s_gx2s
        
        if r1 is None:
            r1, r2 = (r1_vis, r2_vis)
        elif r2 is None:
            r2 = r2_vis
        if c1 is None:
            c1, c2 = (c1_vis, c2_vis)
        elif c2 is None:
            c2 = c2_vis
        
        r1, r2 = sorted([ np.clip(r, r1_vis, r2_vis) for r in (r1, r2) ])
        c1, c2 = sorted([ np.clip(c, c1_vis, c2_vis) for c in (c1, c2) ])
        r_max, c_max = [ s - 1 for s in self.shape ]
        assert 0 <= r1 <= r2 <= r_max, (r1, r2, r_max)
        assert 0 <= c1 <= c2 <= c_max, (c1, c2, c_max)
        
        heights, widths = self._cell_sizes
        x2s, y2s = (self._canvasx(gx2s[1:]), self._canvasy(gy2s[1:]))
        x1s, y1s = (x2s - widths[1:], y2s - heights[1:])
        type_ = 'cell'
        default_style = self._default_styles["cell"]
        default_style = {
            "background": default_style["background"]["normal"],
            "foreground": default_style["foreground"]["normal"],
            "bordercolor": default_style["bordercolor"]["normal"],
            "font": default_style["font"],
            "alignx": default_style["alignx"],
            "aligny": default_style["aligny"],
            "padding": default_style["padding"]
        }
        
        # Draw components for each header
        canvas = self.canvas
        values = self.values
        cell_styles = self._cell_styles
        r_generator = self._index_generator(r1, r2)
        r = next(r_generator)  # get first `r`
        while True:
            skip_this_row = False
            y1, y2 = y1s[r], y2s[r]
            c_generator = self._index_generator(c1, c2)
            c = next(c_generator)  # get first `c`
            while True:
                cell_style = default_style.copy()
                cell_style.update(cell_styles[r, c])
                kw = {"row": r, "col": c,
                      "others": ('xscroll', 'yscroll', 'temp')}
                tag = self._make_tag("type:row:col", type_=type_, **kw)
                
                if skip_existing and canvas.find_withtag(tag):
                    if skip_existing == 'row':  # skip this row
                        skip_this_row = True
                        break
                    # Skip this column
                    try:
                        c = c_generator.send(True)  # get next `c`
                        continue  # go to next column
                    except StopIteration:
                        break  # stop this column iteration
                
                x1, x2 = x1s[c], x2s[c]
                
                ## Delete the existing components
                canvas.delete(tag)
                
                kw["to_tuple"] = True
                
                ## Create new components
                ### Background
                tags = self._make_tags(type_=type_, subtype='background', **kw)
                oid = canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=cell_style["background"],
                    outline=cell_style["bordercolor"],
                    tags=tags
                )
                canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
                
                ## Text
                if not (text := values[r, c]):
                    try:
                        c = c_generator.send(False)  # no text => get next `c`
                        continue  # go to next column
                    except StopIteration:
                        break  # stop this column iteration
                
                tags = self._make_tags(type_=type_, subtype='text', **kw)
                padx, pady = cell_style["padding"]
                pad = {"n": pady, "w": padx, "s": -pady, "e": -padx}
                anchor = [cell_style["alignx"], cell_style["aligny"]]
                xy = []
                for i, (a, p1, p2) in enumerate(zip(anchor, [x1, y1], [x2, y2])):
                    if a in ('n', 'w'):
                        xy.append(p1 + pad[a] + 1)
                    elif a in ('s', 'e'):
                        xy.append(p2 + pad[a])
                    else:
                        anchor[i] = ''
                        xy.append((p1 + p2) // 2)
                justify = 'left' if 'w' in anchor else (
                          'right' if 'e' in anchor else 'center')
                anchor = 'center' if anchor[0] == anchor[1] \
                                  else ''.join(anchor)[::-1]
                text_fit = self._fit_size(
                    text,
                    cell_style["font"],
                    width=x2 - x1 - padx,
                    height=y2 - y1 - pady
                )
                oid = canvas.create_text(
                    *xy,
                    anchor=anchor,
                    text=text_fit,
                    justify=justify,
                    font=cell_style["font"],
                    fill=cell_style["foreground"],
                    tags=tags
                )
                canvas.addtag_withtag(self._make_tag("oid", oid=oid), oid)
                try:
                    c = c_generator.send(False)  # get next `c`
                except StopIteration:
                    break  # stop this column iteration
            
            try:
                r = r_generator.send(skip_this_row)  # get next `r`
            except StopIteration:
                break  # stop this row iteration
        
        # Bindings
        tag_cell = self._make_tag("type", type_=type_)
        canvas.tag_bind(tag_cell, RIGHTCLICK, self._on_cell_rightbutton_press)
        
        # Keep the selection frame on the top
        canvas.tag_raise('selection-frame')
    
    def _on_cell_rightbutton_press(self, event):
        tagdict = self._get_tags('current')
        r, c = self._get_rc(tagdict, to_tuple=True)
        r1, c1, r2, c2 = self._selection_rcs
        (r_low, r_high), (c_low, c_high) = sorted([r1, r2]), sorted([c1, c2])
        
        if not ((r_low <= r <= r_high) and (c_low <= c <= c_high)):
            self.select_cells(r, c, r, c)
        
        menu = self._build_general_rightclick_menu()
        menu.post(event.x_root, event.y_root)
        self.after_idle(menu.destroy)
    
    def _refresh_entry(self, r: int | None = None, c: int | None = None):
        assert (r is None) == (c is None), (r, c)
        
        if (r is None) and (c is None):
            r1, c1, r2, c2 = self._selection_rcs
            r, c = min(r1, r2), min(c1, c2)
        
        heights, widths = self._cell_sizes
        x2 = np.cumsum(widths)[c+1]
        y2 = np.cumsum(heights)[r+1]
        x1, y1 = (x2 - widths[c+1], y2 - heights[r+1])
        x1, x2 = self._canvasx([x1, x2])
        y1, y2 = self._canvasy([y1, y2])
        old_text: str = self.values[r, c]
        
        default_style = self._default_styles["cell"]
        cell_style = {
            "background": default_style["background"]["normal"],
            "foreground": default_style["foreground"]["normal"],
            "bordercolor": default_style["bordercolor"]["normal"],
            "font": default_style["font"],
            "alignx": default_style["alignx"]
        }
        cell_style.update(self._cell_styles[r, c])
        
        en = self._entry
        en.configure(
            background=cell_style["background"],
            foreground=cell_style["foreground"],
            highlightcolor=cell_style["bordercolor"],
            font=cell_style["font"],
            justify={"w": 'left', "e": 'right', "center": 'center'}[
                cell_style["alignx"]]
        )
        en.place(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)
        en.lift(self.canvas)
        self._focus_old_value = old_text
    
    def _focus_in_cell(self, r: int | None = None, c: int | None = None):
        self._focus_out_cell()
        self._refresh_entry(r, c)
        self._entry.focus_set()
    
    def _focus_out_cell(self, discard: bool = False):
        if self._focus_old_value is not None:
            r, c = self._focus_row.get(), self._focus_col.get()
            old_value = self._focus_old_value
            rcs = (r, c, r, c)
            if (not discard) and (new_value := self._entry.get()) != old_value:
                # Apply the new value
                self.set_values(*rcs, values=new_value, draw=False)
                self.draw_cells(*rcs)
                 # put draw here to avoid recursive function calls
                self._history.add(
                    forward=lambda: self.set_values(
                        *rcs, values=new_value, trace='first'),
                    backward=lambda: self.set_values(
                        *rcs, values=old_value, trace='first')
                )
            else:  # Restore the old value
                self._entry.delete(0, 'end')
                self._entry.insert('end', old_value)
            self._focus_old_value = None
        
        self._entry.lower()
        self.focus_set()
    
    def _set_selection(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None
    ) -> tuple:
        assert (r1 is not None) or (r2 is None), (r1, r2)
        assert (c1 is not None) or (c2 is None), (c1, c2)
        
        r_max, c_max = [ s - 1 for s in self.shape ]
        r1 = 0 if r1 is None else int(np.clip(r1, 0, r_max))
        c1 = 0 if c1 is None else int(np.clip(c1, 0, c_max))
        r2 = r_max if r2 is None else int(np.clip(r2, 0, r_max))
        c2 = c_max if c2 is None else int(np.clip(c2, 0, c_max))
        
        self._selection_rcs: tuple[int, int, int, int] = (r1, c1, r2, c2)
        
        return self._selection_rcs
    
    def _update_selection_tags(self) -> tuple:
        # Update the headers' tags
        r1, c1, r2, c2 = self._selection_rcs
        r_low, r_high = sorted([r1, r2])
        c_low, c_high = sorted([c1, c2])
        r_max, c_max = [ s - 1 for s in self.shape ]
        cornercanvas = self.cornercanvas
        rowcanvas, colcanvas = self.rowcanvas, self.colcanvas
        
        for canvas in (cornercanvas, rowcanvas, colcanvas):
            canvas.dtag('selected', 'selected')
        
        if (r_low, r_high) == (0, r_max) and (c_low, c_high) == (0, c_max):
            tag = self._make_tag("type", type_='cornerheader')
            cornercanvas.addtag_withtag('selected', tag)
        
        (r1_vis, r2_vis), (c1_vis, c2_vis) = self._visible_rcs
        rows_on = set(range(r_low, r_high+1)) & set(range(r1_vis, r2_vis+1))
        cols_on = set(range(c_low, c_high+1)) & set(range(c1_vis, c2_vis+1))
        
        for r in rows_on:
            tag = self._make_tag("type:row", type_='rowheader', row=r)
            rowcanvas.addtag_withtag('selected', tag)
        for c in cols_on:
            tag = self._make_tag("type:col", type_='colheader', col=c)
            colcanvas.addtag_withtag('selected', tag)
        
        return (rows_on, cols_on)
    
    def select_cells(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            trace: str | None = None
    ) -> tuple:
        assert trace in (None, 'first', 'last'), trace
        
        self._focus_out_cell()
        
        r1, c1, r2, c2 = self._set_selection(r1, c1, r2, c2)
        r_low, r_high = sorted([r1, r2])
        c_low, c_high = sorted([c1, c2])
        
        # Update selection frame's styles
        selection_style = self._default_styles["selection"]
        color, w = selection_style["color"], selection_style["width"]
        self.canvas.itemconfigure('selection-frame', outline=color, width=w)
        
        # Relocate the selection frame
        gy2s, gx2s = self._gy2s_gx2s
        x1, x2 = self._canvasx([gx2s[c_low], gx2s[c_high+1]])
        y1, y2 = self._canvasy([gy2s[r_low], gy2s[r_high+1]])
        dw = int(w % 2 == 0)
        self.canvas.coords('selection-frame', x1, y1, x2+dw, y2+dw)
        
        # Relocate the viewing window to trace the first selected cell (r1, c1) 
        # or the last selected cell (r2, c2)
        if trace:
            r, c = (r1, c1) if trace == 'first' else (r2, c2)
            (gx1_vis, gx2_vis), (gy1_vis, gy2_vis) = self._visible_xys
            heights, widths = self._cell_sizes
            gx2, gy2 = (gx2s[c+1], gy2s[r+1])
            gx1, gy1 = (gx2 - widths[c+1], gy2 - heights[r+1])
            if (dx := gx1 - gx1_vis) < 0 or (dx := gx2 - gx2_vis + 1) > 0:
                self.xview_scroll(dx, 'pixels')
            if (dy := gy1 - gy1_vis) < 0 or (dy := gy2 - gy2_vis + 1) > 0:
                self.yview_scroll(dy, 'pixels')
        
        # Set each header's state
        rows_on, cols_on = self._update_selection_tags()
        (r1_vis, r2_vis), (c1_vis, c2_vis) = self._visible_rcs
        r_max, c_max = [ s - 1 for s in self.shape ]
        
        ## Rowheaders
        for r in range(r1_vis, r2_vis+1):
            tagdict = self._make_tags(type_='rowheader', row=r, raw=False)
            self._set_header_state(
                tagdict, state='selected' if r in rows_on else 'normal'
            )
        
        ## Colheaders
        for c in range(c1_vis, c2_vis+1):
            tagdict = self._make_tags(type_='colheader', col=c, raw=False)
            self._set_header_state(
                tagdict, state='selected' if c in cols_on else 'normal'
            )
        
        ## Cornerheader
        state = 'selected' if ((r_low, r_high) == (0, r_max) and
                               (c_low, c_high) == (0, c_max)) \
                           else 'normal'
        tagdict = self._make_tags(type_='cornerheader', raw=False)
        self._set_header_state(tagdict, state=state)
        
        # Update focus indices and focus value
        self._focus_row.set(r_low)
        self._focus_col.set(c_low)
        self._focus_value.set(self.values[r_low, c_low])
        
        return self._selection_rcs
    
    _reselect_cells = lambda self, *args, **kw: self.select_cells(
        *self._selection_rcs, *args, **kw)
    
    def _move_selections(
            self,
            direction: str,
            area: str | None = None,
            expand: bool = False
    ) -> tuple[int, int, int, int]:
        assert direction in ('up', 'down', 'left', 'right'), direction
        assert area in ('paragraph', 'all', None), area
        assert isinstance(expand, bool), expand
        
        r1_old, c1_old, r2_old, c2_old = self._selection_rcs
        n_rows, n_cols = self.shape
        rc_max = [n_rows - 1, n_cols - 1]
        axis = 0 if direction in ('up', 'down') else 1
        rc1_new = [r1_old, c1_old]
        rc2_old, rc2_new = [r2_old, c2_old], [r2_old, c2_old]
        
        if area == 'all':
            # Move the last selection to the edge
            rc2_new[axis] = 0 if direction in ('up', 'left') else rc_max[axis]
            
            if not expand:  # single-cell selection
                rc1_new = rc2_new
            
            return self.select_cells(*rc1_new, *rc2_new, trace='last')
        
        elif area == 'paragraph':
            # Move the last selection to the nearset nonempty cell in the same
            # paragraph or next paragraph
            
            ## Slice the cell value array to an 1-D DataFrame
            if direction in ('up', 'left'):
                rc_lim = [0, 0]
                if axis == 0:
                    values = self.values[:r2_old+1, c2_old]
                else:
                    values = self.values[r2_old, :c2_old+1]
                values = values.to_numpy().ravel()[::-1]  # flip
                correction1, correction2 = (-1, values.size - 1)
            else:  # down or right
                rc_lim = rc_max
                correction1 = 1
                if axis == 0:
                    correction2 = r2_old
                    values = self.values[r2_old:, c2_old]
                else:
                    correction2 = c2_old
                    values = self.values[r2_old, c2_old:]
                values = values.to_numpy().ravel()
            
            ## Find the nearset nonempty cell that in the same paragraph or next
            #  paragraph
            diff = np.diff((values != '').astype(int))
            vary_at = np.flatnonzero(diff)
            if vary_at.size and (vary_at[0] == 0) and (diff[0] == -1):
                vary_at = vary_at[1:]
            
            if vary_at.size:  # found
                rc2_new[axis] = (vary_at[0] if diff[vary_at[0]] == -1
                    else vary_at[0] + 1) * correction1 + correction2
            else:  # not found
                rc2_new[axis] = rc_lim[axis]
            
            if not expand:  # single-cell selection
                rc1_new = rc2_new
            
            return self.select_cells(*rc1_new, *rc2_new, trace='last')
        
        # Move the last selection by 1 step
        step = -1 if direction in ('up', 'left') else +1
        rc2_new[axis] = int(np.clip(rc2_old[axis] + step, 0, rc_max[axis]))
        
        if not expand:  # single-cell selection
            rc1_new = rc2_new
        
        return self.select_cells(*rc1_new, *rc2_new, trace='last')
    
    def draw(
            self,
            update_visible_rcs: bool = True,
            skip_existing: str | None = None,
            trace: str | None = None
    ):
        assert skip_existing in ('row', 'col', None), skip_existing
        
        if update_visible_rcs:
            self._update_visible_and_p2s()
        self.draw_cornerheader(skip_existing=skip_existing is not None)
        self.draw_headers(axis=0, skip_existing=skip_existing == 'row')
        self.draw_headers(axis=1, skip_existing=skip_existing == 'col')
        self.draw_cells(skip_existing=skip_existing)
        self._reselect_cells(trace=trace)
    
    def refresh(self, scrollbar: str = 'both', trace: str | None = None):
        assert scrollbar in ('x', 'y', 'both'), scrollbar
        
        self._canvases_delete('temp')
        
        if scrollbar in ('x', 'both'):
            self.xview_scroll(0, 'units')
        if scrollbar in ('y', 'both'):
            self.yview_scroll(0, 'units')
        
        self._reselect_cells(trace=trace)
    
    def _canvases_delete(self, tag: str):
        self.cornercanvas.delete(tag)
        self.rowcanvas.delete(tag)
        self.colcanvas.delete(tag)
        self.canvas.delete(tag)
    
    def undo(self):
        self._focus_out_cell()
        history = self._history
        if history.backable:
            self._history.back()
        else:
            print('Not backable: current step =', history.step)
    
    def redo(self):
        self._focus_out_cell()
        history = self._history
        if history.forwardable:
            self._history.forward()
        else:
            print('Not forwardable: current step =', history.step)
    
    def zoom(self, scale: float = 1.):
        """
        Zoom in when `scale` > 1.0 or zoom out when `scale` < 1.0.
        """
        assert scale in self._valid_scales, [self._valid_scales, scale]
        self._scale.set(scale)
    
    def _zoom(self, *_):
        # Update scale state
        scale = self._scale.get()
        ratio = scale / self._prev_scale
        self._prev_scale = scale
        
        # Update cell sizes
        self._cell_sizes = [ (sizes * ratio) for sizes in self._cell_sizes ]
        self._update_content_size()
        
        # Update text sizes
        ## Default fonts
        for font in self._collect_fonts(self._default_styles):
            font.configure(size=int(font._unscaled_size * scale))
        
        ## Cell fonts
        for row_styles in self._cell_styles:
            for style in row_styles:
                if font := style.get("font"):
                    font.configure(size=int(font._unscaled_size * scale))
        
        # Refresh canvases
        self.refresh()
    
    def _zoom_to_next_scale_level(self, direction: str = 'up'):
        """
        If `direction` is 'up', zoom in with next scaling factor. If `direction`
        is 'down', zoom out with next scaling factor.
        """
        assert direction in ('up', 'down'), direction
        
        current_idx = self._valid_scales.index(self._scale.get())
        max_idx = len(self._valid_scales) - 1
        
        if direction == 'up':
            if current_idx == max_idx:
                return print('Already reached the highest scaling factor!')
            next_idx = current_idx + 1
        else:
            if current_idx == 0:
                return print('Already reached the lowest scaling factor!')
            next_idx = current_idx - 1
        
        self.zoom(self._valid_scales[next_idx])
    
    def resize_cells(
            self,
            i: int,
            axis: int,
            N: int = 1,
            sizes: np.ndarray | None = None,
            dialog: bool = False,
            trace: str | None = None,
            undo: bool = False
    ) -> np.ndarray:
        assert axis in (0, 1), axis
        max_i = self.shape[axis] - 1
        assert -1 <= i <= max_i + 1, (i, max_i)
        assert N >= 1, N
        
        scale = self._scale.get()
        _idc = np.arange(i+1, i+N+1)
        unscaled_old_sizes = self._cell_sizes[axis][_idc] / scale
        
        # Check for the new sizes
        if dialog:  # ask for new size
            dimension = ('height', 'width')[axis]
            size = dialogs.Querybox.get_integer(
                parent=self,
                prompt=f'Enter the new {dimension}:',
                initialvalue=int(unscaled_old_sizes[0]),
                width=40,
                position=self._center_window
            )
            if not isinstance(size, int):  # cancelled
                return
            unscaled_new_sizes = sizes = np.full(N, size)
        elif sizes is None:  # reset the rows or cols sizes
            unscaled_new_sizes = np.full(N, self._default_cell_sizes[axis])
        else:  # new size
            assert np.shape(sizes) == (N,), (sizes, N)
            unscaled_new_sizes = np.asarray(sizes)
        new_sizes = unscaled_new_sizes * scale
        old_sizes = unscaled_old_sizes * scale
        
        # Update the status of the resized rows or cols
        key = ("row", "col")[axis]
        min_size = self._min_sizes[axis] * scale
        deltas = np.maximum(new_sizes - old_sizes, min_size - old_sizes)
        self._cell_sizes[axis][_idc] += deltas
        self._update_content_size()
        
        # Move the bottom rows or right cols
        (r1_vis, r2_vis), (c1_vis, c2_vis) = self._visible_rcs
        if axis == 0:
            header_canvas = self.rowcanvas
            i2 = r2_vis
            dx, dy = (0, deltas.sum())
        else:
            header_canvas = self.colcanvas
            i2 = c2_vis
            dx, dy = (deltas.sum(), 0)
        
        canvas = self.canvas
        for i_move in range(i+N, i2+1):
            tag_move = self._make_tag(key, row=i_move, col=i_move)
            header_canvas.move(tag_move, dx, dy)
            canvas.move(tag_move, dx, dy)
        
        # Delete the resized rows or cols
        for i_resized in _idc - 1:
            tag_resized = self._make_tag(key, row=i_resized, col=i_resized)
            self._canvases_delete(tag_resized)
        
        # Redraw the deleted rows or cols
        if axis == 0:
            self.yview_scroll(0, 'units')
            rcs = {"r1": i, "r2": i + N - 1}
        else:
            self.xview_scroll(0, 'units')
            rcs = {"c1": i, "c2": i + N - 1}
        self.draw_headers(axis=axis)
        self.draw_cells(**rcs)
        self.select_cells(**rcs, trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.resize_cells(
                    i, axis=axis, N=N, sizes=copy.copy(sizes), trace='first'),
                backward=lambda: self.resize_cells(
                    i,
                    axis=axis,
                    N=N,
                    sizes=copy.copy(unscaled_old_sizes),
                    trace='first'
                )
            )
        
        return unscaled_new_sizes
    
    def insert_cells(
            self,
            i: int | None = None,
            *,
            axis: int,
            N: int = 1,
            df: pl.DataFrame | None = None,
            sizes: np.ndarray | None = None,
            styles=None,
            dialog: bool = False,
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        assert axis in (0, 1), axis
        old_df, old_shape = self.values, self.shape
        max_i = old_shape[axis] - 1
        i = max_i + 1 if i is None else i
        assert 0 <= i <= max_i + 1, (i, max_i)
        
        if dialog:
            # Ask for number of rows/cols to insert
            axis_name = ('rows', 'columns')[axis]
            N = dialogs.Querybox.get_integer(
                parent=self,
                title='Rename Sheet',
                prompt=f'Enter the number of {axis_name} to insert:',
                initialvalue=1,
                minvalue=1,
                maxvalue=100000,
                width=40,
                position=self._center_window
            )
            if not isinstance(N, int):
                return
        else:
            N = int(N)
        assert N >= 1, N
        
        new_shape = list(old_shape)
        new_shape[axis] = N
        new_shape = tuple(new_shape)
        
        # Create a list of new sizes (a 1-D list)
        if sizes is None:
            unscaled_new_size = self._default_cell_sizes[axis]
            unscaled_new_sizes = np.array([ unscaled_new_size for i in range(N) ])
        else:
            assert np.shape(sizes) == (N,), (sizes, N)
            unscaled_new_sizes = np.asarray(sizes)
        new_sizes = unscaled_new_sizes * self._scale.get()
        
        # Create a list of new styles
        if styles is None:
            new_styles = np.array([ [ dict() for _ in range(new_shape[1]) ]
                                    for _ in range(new_shape[0]) ])
        else:
            assert np.shape(styles) == new_shape, styles
            new_styles = np.asarray(styles)
        
        # Create a dataframe containing the new values (a 2-D dataframe)
        if df is None:
            inserted_df = df_full(new_shape, '', dtype=pl.String)
        else:
            assert isinstance(df, pl.DataFrame), (type(df), df)
            assert np.shape(df) == new_shape, (df, new_shape)
            inserted_df = df
        
        # Extract the leading and trailing partitions
        if axis == 0:  # new rows
            orient = 'vertical'
            leading, trailing = old_df[:i, :], old_df[i:, :]
        else:  # new columns
            orient = 'horizontal'
            leading, trailing = old_df[:, :i], old_df[:, i:]
        
        # Insert the new values
        self._values = df_concat(
            [leading, inserted_df, trailing],
            how=orient
        )
        
        # Insert the new sizes
        idc = [i+1] * N  # add 1 to skip the header size
        self._cell_sizes[axis] = np.insert(self._cell_sizes[axis], idc, new_sizes)
        self._update_content_size()
        
        # Insert the new styles
        idc = [i] * N
        self._cell_styles = np.insert(
            self._cell_styles, idc, new_styles, axis=axis)
        
        if axis == 0:
            selection_kw = dict(r1=i, r2=i+N-1)
            scrollbar = 'y'
        else:
            selection_kw = dict(c1=i, c2=i+N-1)
            scrollbar = 'x'
        
        # Select cells
        self._set_selection(**selection_kw)
        
        # Redraw
        if draw:
            self.refresh(scrollbar=scrollbar, trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.insert_cells(
                    i,
                    axis=axis,
                    N=N,
                    df=None if df is None else df.copy(),
                    sizes=None if sizes is None else copy.copy(sizes),
                    styles=None if styles is None else np.array(
                        [ [ d.copy() for d in dicts] for dicts in styles ]),
                    draw=draw,
                    trace='first'
                ),
                backward=lambda: self.delete_cells(
                    i, axis=axis, N=N, draw=draw, trace='first')
            )
    
    def delete_cells(
            self,
            i: int,
            axis: int,
            N: int = 1,
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        assert axis in (0, 1), axis
        max_i = self.shape[axis] - 1
        N = int(N)
        assert 0 <= i <= max_i, (i, max_i)
        assert N >= 1, N
        
        # Delete the values
        old_df = self.values
        idc = np.arange(i, i+N)
        if axis == 0:
            orient = 'vertical'
            idc_2d = (idc, slice(None))
            leading, trailing = old_df[:i, :], old_df[i+N:, :]
        else:
            orient = 'horizontal'
            idc_2d = (slice(None), idc)
            leading, trailing = old_df[:, :i], old_df[:, i+N:]
        deleted_df = self.values[idc_2d].clone()
        self._values = df_concat([leading, trailing], how=orient)
        
        # Delete the sizes
        _idc = idc + 1  # add 1 to skip the header size
        all_sizes = [ sizes.copy() for sizes in self._cell_sizes ]
        deleted_sizes = self._cell_sizes[axis][_idc].copy()
        self._cell_sizes[axis] = np.delete(self._cell_sizes[axis], _idc)
        self._update_content_size()
        
        # Delete the styles
        deleted_styles = np.array([ [ d.copy() for d in dicts ]
                                    for dicts in self._cell_styles[idc_2d] ])
        self._cell_styles = np.delete(self._cell_styles, idc, axis=axis)
        
        
        # Select cells
        if axis == 0:
            selection_kw = dict(r1=i, r2=i+N-1)
        else:
            selection_kw = dict(c1=i, c2=i+N-1)
        self._set_selection(**selection_kw)
        
        # Redraw
        was_reset = False
        if self.values.is_empty():  # reset the Sheet if no cells exist
            self.set_states(
                shape=self._default_cell_shape,
                cell_width=self._default_cell_sizes[1],
                cell_height=self._default_cell_sizes[0],
                min_width=self._min_sizes[1],
                min_height=self._min_sizes[0],
                get_style=self._get_style,
                max_undo=self._history._max_height,
                lock_number_of_rows=self._lock_number_of_rows,
                lock_number_of_cols=self._lock_number_of_cols,
                scale=self._scale.get(),
                draw=draw
            )
            deleted_sizes = all_sizes
            was_reset = True
        elif draw:
            self.refresh(trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.delete_cells(
                    i, axis=axis, N=N, draw=draw, trace='first'),
                backward=lambda: self._undo_delete_cells(
                    i,
                    axis=axis,
                    N=N,
                    df=deleted_df,
                    sizes=deleted_sizes,
                    styles=deleted_styles,
                    was_reset=was_reset,
                    draw=draw,
                    trace='first'
                )
            )
    
    def _undo_delete_cells(
            self,
            i: int,
            axis: int,
            N: int,
            df: pl.DataFrame,
            sizes: np.ndarray | list[np.ndarray],
            styles: np.ndarray,
            was_reset: bool,
            draw: bool = True,
            trace: str | None = None
    ):
        assert isinstance(df, pl.DataFrame), df
        assert isinstance(styles, np.ndarray), styles
        assert isinstance(sizes, (np.ndarray, list)), sizes
        assert styles.shape == df.shape, (styles.shape, df.shape)
        
        if was_reset:
            n_rows, n_cols = df.shape
            assert isinstance(sizes, list), type(sizes)
            assert len(sizes) == 2, sizes
            assert all( isinstance(ss, np.ndarray) for ss in sizes ), sizes
            assert sizes[0].shape == (n_rows+1,), (sizes[0].shape, df.shape)
            assert sizes[1].shape == (n_cols+1,), (sizes[1].shape, df.shape)
            
            self._values = df.clone()
            self._cell_styles = np.array([ [ d.copy() for d in dicts ]
                                           for dicts in styles ])
            self._cell_sizes = [ ss.copy() for ss in sizes ]
            self._update_content_size()
            
            self._set_selection()
            
            if draw:
                self.refresh(trace=trace)
                self.xview_moveto(0.)
                self.yview_moveto(0.)
        else:
            self.insert_cells(
                i,
                axis=axis,
                N=N,
                df=df,
                sizes=sizes,
                styles=styles,
                draw=draw,
                trace=trace
            )
    
    def set_values(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            values: pl.DataFrame | str = '',
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        assert isinstance(values, (pl.DataFrame, str)), type(values)
        
        r1, c1, r2, c2 = rcs = self._set_selection(r1, c1, r2, c2)
        rr, cc = sorted([r1, r2]), sorted([c1, c2])
        r_low, r_high, c_low, c_high = map(int, (*rr, *cc))
        
        if isinstance(values, pl.DataFrame):
            df = values.cast(pl.String)  # string type
        else:
            df = df_full(
                (r_high-r_low+1, c_high-c_low+1),
                values,
                dtype=pl.String
            )
        
        old_values = self.values[r_low:r_high+1, c_low:c_high+1].clone()
        self._values = df_set_values(
            self.values, df, r_start=r_low, c_start=c_low
        )
        
        if draw:
            self.draw_cells(*rcs)
            self._reselect_cells(trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.set_values(
                    *rcs, values=df.clone(), draw=draw, trace='first'),
                backward=lambda: self.set_values(
                    *rcs, values=old_values, draw=draw, trace='first')
            )
    
    def erase_values(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        self.set_values(
            r1, c1, r2, c2, values='', draw=draw, undo=undo, trace=trace)
    
    def copy_values(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            to_cpliboard: bool = True
    ) -> pl.DataFrame:
        r1, c1, r2, c2 = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        values_to_copy = self.values[idc]
        
        if to_cpliboard:
            values_to_copy.write_clipboard(
                include_header=False, separator='\t', line_terminator='\n'
            )
        
        return values_to_copy
    
    def paste_values(
            self,
            r: int | None = None,
            c: int | None = None,
            df: pl.DataFrame | None = None,
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        if df is None:
            df = pl.read_clipboard(
                has_header=False, separator='\t', infer_schema=False
            )
        
        n_rows, n_cols = df.shape
        r1, c1, r2, c2 = self._selection_rcs
        r_start = min(r1, r2) if r is None else r
        c_start = min(c1, c2) if c is None else c
        r_end, c_end = (r_start + n_rows - 1, c_start + n_cols - 1)
        
        idc = (slice(r_start, r_end + 1), slice(c_start, c_end + 1))
        n_rows_exist, n_cols_exist = self.values[idc].shape
        r_max, c_max = self.shape  # add new cells at the end
        with self._history.add_sequence() as seq:
            # Add new rows/cols before pasting if the table to paste has a
            # larger shape
            if (n_rows_add := n_rows - n_rows_exist):
                if self._lock_number_of_rows:
                    df = df[:-n_rows_add, :]
                else:
                    self.insert_cells(
                        r_max, axis=0, N=n_rows_add, draw=False, undo=undo)
            if (n_cols_add := n_cols - n_cols_exist):
                if self._lock_number_of_cols:
                    df = df[:, :-n_cols_add]
                else:
                    self.insert_cells(
                        c_max, axis=1, N=n_cols_add, draw=False, undo=undo)
            
            # Set values
            self.set_values(
                r_start, c_start, r_end, c_end,
                values=df,
                draw=False,
                trace=None,
                undo=undo
            )
            
            if undo:
                seq["forward"].append(lambda: self.refresh(trace='first'))
                seq["backward"].insert(0, lambda: self.refresh(trace='first'))
        
        if draw:
            self.refresh(trace=trace)
    
    def set_styles(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            *,
            property_: str,
            values=None,
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        r1, c1, r2, c2 = rcs = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        styles = self._cell_styles[idc]  # an sub array of dictionaries
        
        if not isinstance(values, np.ndarray):  # broadcast
            values = np.full(styles.shape, values, dtype=object)
        assert values.shape == styles.shape, (values.shape, styles.shape)
        
        # Scale fonts
        if property_ == 'font':
            scale = self._scale.get()
            for font in values.flat:
                if font is not None:
                    font.configure(size=int(font._unscaled_size * scale))
        
        # Get the old values
        old_values = np.array([ [ d.get(property_) for d in dicts ]
                              for dicts in styles ])
        
        # Update the style collections with the new values
        for style, value in zip(styles.flat, values.flat):
            if value is None:
                style.pop(property_, None)
            else:
                style[property_] = value
        
        if draw:
            self.draw_cells(r1, c1, r2, c2)
            self._reselect_cells(trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.set_styles(
                    *rcs,
                    property_=property_,
                    values=values,
                    draw=draw,
                    trace='first'
                ),
                backward=lambda: self.set_styles(
                    *rcs,
                    property_=property_,
                    values=old_values,
                    draw=draw,
                    trace='first'
                )
            )
    
    def reset_styles(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            *,
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        r1, c1, r2, c2 = rcs = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        styles = self._cell_styles[idc]  # an sub array of dictionaries
        
        # Get the old values
        old_styles = np.array([ [ d.copy() for d in dicts ]
                                for dicts in styles ])
        
        # Clear the style collections
        for style in styles.flat:
            style.clear()
        
        if draw:
            self.draw_cells(r1, c1, r2, c2)
            self._reselect_cells(trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.reset_styles(
                    *rcs,
                    draw=draw,
                    trace='first'
                ),
                backward=lambda: self._undo_reset_styles(
                    *rcs,
                    styles=old_styles,
                    draw=draw,
                    trace='first'
                )
            )
    
    def _undo_reset_styles(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            *,
            styles: np.ndarray,
            draw: bool = True,
            trace: str | None = None
    ):
        assert isinstance(styles, np.ndarray), type(styles)
        assert styles.ndim == 2, styles.shape
        assert all([ isinstance(d, dict) for dicts in styles for d in dicts ])
        
        r1, c1, r2, c2 = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        old_styles = self._cell_styles[idc]  # an sub array of dictionaries
        assert old_styles.shape == styles.shape, [old_styles.shape, styles.shape]
        
        # Clear the style collections
        for style, new_style in zip(old_styles.flat, styles.flat):
            style.clear()
            style.update(new_style)
        
        if draw:
            self.draw_cells(r1, c1, r2, c2)
            self._reselect_cells(trace=trace)
    
    def set_fonts(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            fonts=None,
            dialog: bool = False,
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        if dialog:
            default_font = self._default_styles["cell"]["font"]
            style_topleft = self._cell_styles[min(r1, r2), min(c1, c2)]
            font_topleft = style_topleft.get("font", default_font)
            scale = self._scale.get()
            font = dialogs.Querybox.get_font(
                parent=self,
                initialvalue=font_topleft,
                scale=scale,
                position=self._center_window
            )
            if font is None:
                return
            fonts = font_topleft if font_topleft.actual() == font.actual() \
                    else font
        
        self.set_styles(
            r1, c1, r2, c2,
            property_='font',
            values=fonts,
            draw=draw,
            undo=undo,
            trace=trace
        )
        
        return fonts
    
    def _set_colors(
            self,
            r1: int | None = None,
            c1: int | None = None,
            r2: int | None = None,
            c2: int | None = None,
            field: str = 'foreground',
            colors=None,
            dialog: bool = False,
            draw: bool = True,
            trace: str | None = None,
            undo: bool = False
    ):
        assert field in ('foreground', 'background'), field
        
        if dialog:
            default_color = self._default_styles["cell"][field]["normal"]
            style_topleft = self._cell_styles[min(r1, r2), min(c1, c2)]
            color_topleft = style_topleft.get(field, default_color)
            color = dialogs.Querybox.get_color(
                parent=self,
                initialvalue=color_topleft,
                position=self._center_window
            )
            if color is None:
                return
            colors = color.hex
        
        self.set_styles(
            r1, c1, r2, c2,
            property_=field,
            values=colors,
            draw=draw,
            undo=undo,
            trace=trace
        )
        
        return colors
    
    def set_foregroundcolors(self, *args, **kwargs):
        return self._set_colors(*args, field='foreground', **kwargs)
    
    def set_backgroundcolors(self, *args, **kwargs):
        return self._set_colors(*args, field='background', **kwargs)
    
    def _selection_insert_cells(self,
                                axis: int,
                                mode: str = 'ahead',
                                dialog: bool = False,
                                undo: bool = False):
        assert axis in (0, 1), axis
        assert mode in ('ahead', 'behind'), mode
        
        r1, c1, r2, c2 = self._selection_rcs
        rcs = [(r1, r2), (c1, c2)] = [sorted([r1, r2]), sorted([c1, c2])]
        r_max, c_max = [ s - 1 for s in self.shape ]
        (_i1, _i2), max_i = rcs[axis-1], [r_max, c_max][axis-1]
        assert (_i1 == 0) and (_i2 >= max_i), (axis, (r1, c1, r2, c2), self.shape)
        
        if mode == 'ahead':
            i = rcs[axis][0]
        else:
            i = rcs[axis][1] + 1
        
        self.insert_cells(i, axis=axis, dialog=dialog, undo=undo)
    
    def _selection_delete_cells(self, undo: bool = False):
        r1, c1, r2, c2 = self._selection_rcs
        rcs = [(r1, r2), (c1, c2)] = [sorted([r1, r2]), sorted([c1, c2])]
        r_max, c_max = [ s - 1 for s in self.shape ]
        if (r1 == 0) and (r2 >= r_max):  # cols selected
            axis = 1
        elif (c1 == 0) and (c2 >= c_max):  # rows selected
            axis = 0
        else:
            raise ValueError(
                'Inserting new cells requires entire row(s)/col(s) being '
                'selected. However, the selected row(s) and col(s) indices are: '
                f'{r1} <= r <= {r2} and {c1} <= c <= {c2}'
            )
        
        i1, i2 = rcs[axis]
        self.delete_cells(i1, axis=axis, N=i2-i1+1, undo=undo)
    
    def _selection_erase_values(self, undo: bool = False):
        rcs = self._selection_rcs
        self.erase_values(*rcs, undo=undo)
    
    def _selection_copy_values(self):
        rcs = self._selection_rcs
        self.copy_values(*rcs)
    
    def _selection_paste_values(self, undo: bool = False):
        self.paste_values(undo=undo)
    
    def _selection_set_styles(
            self, property_: str, values=None, undo: bool = False):
        rcs = self._selection_rcs
        self.set_styles(*rcs, property_=property_, values=values, undo=undo)
    
    def _selection_reset_styles(self, undo: bool = False):
        rcs = self._selection_rcs
        self.reset_styles(*rcs, undo=undo)
    
    def _selection_set_fonts(self, dialog: bool = False, undo: bool = False):
        rcs = self._selection_rcs
        self.set_fonts(*rcs, dialog=dialog, undo=undo)
    
    def _selection_set_foregroundcolors(
            self, dialog: bool = False, undo: bool = False):
        rcs = self._selection_rcs
        self.set_foregroundcolors(*rcs, dialog=dialog, undo=undo)
    
    def _selection_set_backgroundcolors(
            self, dialog: bool = False, undo: bool = False):
        rcs = self._selection_rcs
        self.set_backgroundcolors(*rcs, dialog=dialog, undo=undo)
    
    def _selection_resize_cells(
            self, axis: int, dialog: bool = False, undo: bool = False):
        r1, c1, r2, c2 = self._selection_rcs
        rcs = [(r1, r2), (c1, c2)] = [sorted([r1, r2]), sorted([c1, c2])]
        r_max, c_max = [ s - 1 for s in self.shape ]
        (_i1, _i2), max_i = rcs[axis-1], [r_max, c_max][axis-1]
        assert (_i1 == 0) and (_i2 >= max_i), (axis, (r1, c1, r2, c2), self.shape)
        
        i1, i2 = rcs[axis]
        self.resize_cells(i1, axis=axis, N=i2-i1+1, dialog=dialog, undo=undo)


class Book(ttk.Frame):
    @property
    def sheets(self) -> dict[str, Sheet]:
        """
        Return a dictionary containing all `Sheet`s and their names.
        """
        return { ps["name"]: ps["sheet"] for ps in self._sheets_props.values() }
    
    @property
    def sheet(self) -> Sheet | None:
        """
        Return current `Sheet` or `None`.
        """
        return self._sheet
    
    @property
    def sidebar_hidden(self) -> bool:
        return self._panedwindow.sashpos(0) <= 0
    
    def __init__(
            self,
            master,
            scrollbar_bootstyle='round',
            sidebar_width: int = 150,
            lock_number_of_sheets: bool = False,
            dfs: dict[str, pl.DataFrame] = {
                "Sheet 1": df_full((10, 10), '', dtype=pl.String)
            },
            sheet_kw: dict = {
                "shape": (10, 10),
                "cell_width": 80,
                "cell_height": 25,
                "min_width": 20,
                "min_height": 10
            },
            **kwargs
    ):
        super().__init__(master, **kwargs)
        self._create_styles()
        self._lock_number_of_sheets: bool = bool(lock_number_of_sheets)
        
        
        # Build toolbar
        self._toolbar = ttk.Frame(self)
        self._toolbar.pack(fill='x', padx=9, pady=3)
        
        ## Sidebar button
        self._sidebar_toggle = ttk.Button(
            self._toolbar,
            style=self._toolbar_bt_style,
            text='[Sidebar]',
            command=self._toggle_sidebar,
            takefocus=False
        )
        self._sidebar_toggle.pack(side='left')
        
        ## Separator
        sep_fm = ttk.Frame(self._toolbar, width=3)
        sep_fm.pack(side='left', fill='y', padx=9, ipady=9)
        sep = ttk.Separator(sep_fm, orient='vertical', takefocus=False)
        sep.place(relx=0.5, y=0, relheight=1.)
        
        ## Undo button
        self._undo_btn = ttk.Button(
            self._toolbar,
            style=self._toolbar_bt_style,
            text='↺ Undo',
            command=lambda: self.sheet.undo(),
            takefocus=False
        )
        self._undo_btn.pack(side='left')
        
        ## Redo button
        self._redo_btn = ttk.Button(
            self._toolbar,
            style=self._toolbar_bt_style,
            text='↻ Redo',
            command=lambda: self.sheet.redo(),
            takefocus=False
        )
        self._redo_btn.pack(side='left', padx=[5, 0])
        
        ## Zooming optionmenu
        self._zoom_om = zoom_om = OptionMenu(self._toolbar, bootstyle='outline')
        self._zoom_om.pack(side='right')
        om_style = f'{id(zoom_om)}.{zoom_om["style"]}'
        self._root().style.configure(om_style, padding=[5, 0])
        self._zoom_om.configure(style=om_style)
        
        zoom_lb = ttk.Label(self._toolbar, text='x', bootstyle='primary')
        zoom_lb.pack(side='right', padx=[5, 2])
        
        # Build inputbar
        self._inputbar = ttk.Frame(self)
        self._inputbar.pack(fill='x', padx=9, pady=[9, 6])
        self._inputbar.grid_columnconfigure(0, minsize=130)
        self._inputbar.grid_columnconfigure(1, weight=1)
        
        ## Row and col labels
        self._label_fm = ttk.Frame(self._inputbar)
        self._label_fm.grid(row=0, column=0, sticky='sw')
        
        font = ('TkDefaultfont', 10)
        R_label = ttk.Label(self._label_fm, text='R', font=font)  # prefix R
        R_label.pack(side='left')
        
        self._r_label = ttk.Label(self._label_fm, font=font)
        self._r_label.pack(side='left')
        
        s_label = ttk.Label(self._label_fm, text=',  ', font=font)  # seperator
        s_label.pack(side='left')
        
        C_label = ttk.Label(self._label_fm, text='C', font=font)  # prefix C
        C_label.pack(side='left')
        
        self._c_label = ttk.Label(self._label_fm, font=font)
        self._c_label.pack(side='left')
        
        ## Entry
        self._entry = ttk.Entry(self._inputbar, style=self._entry_style)
        self._entry.grid(row=0, column=1, sticky='nesw', padx=[12, 0])
        self._entry.bind('<FocusIn>', lambda e: self.sheet._refresh_entry())
        self._entry.bind(
            '<KeyPress>', lambda e: self.sheet._on_entry_key_press(e)
        )
        
        
        # Build sidebar and sheet frame
        border_fm = ttk.Frame(self, style=self._border_fm_style, padding=1)
        border_fm.pack(fill='both', expand=True)
        
        self._panedwindow = ttk.Panedwindow(border_fm, orient='horizontal')
        self._panedwindow.pack(fill='both', expand=True)
        
        ## Sidebar
        self._sidebar_width: int = int(sidebar_width)
        self._sidebar_fm = ScrolledFrame(
            self._panedwindow,
            vbootstyle=scrollbar_bootstyle,
            propagate_geometry=False
        )
        self._panedwindow.add(self._sidebar_fm.container)
        
        ### Sheet tab container
        def _remove_selected():
            if not (selected_items := self._sidebar.selected_items):
                return
            
            selected_keys = [ key for key, props in self._sheets_props.items()
                              if props["item"] in selected_items ]
            
            # Ask if the user confirm to delete the selected sheets
            if not self._delete_sheet(selected_keys.pop(), request=True):
                return
            
            # The user confirmed it. Continue to delete the selected sheets
            for key in selected_keys:
                self._delete_sheet(key, request=False)
            self._sidebar.remove_selected()
        
        state = 'disabled' if lock_number_of_sheets else 'normal'
        self._sidebar = RearrangedDnDContainer(self._sidebar_fm)
        self._sidebar.set_rearrange_commands(
            {
                "label": 'Remove Selected...',
                "command": _remove_selected,
                "state": state
            }
        )
        self._sidebar.set_other_commands(
            {
                "label": 'New Sheet...',
                "command": lambda: self.insert_sheet(dialog=True),
                "state": state
            }
        )
        self._sidebar.pack(fill='both', expand=True)
        self._sidebar.set_dnd_end_callback(self._on_dnd_end)
        
        ## Frame to contain sheets
        self._sheet_pane = ttk.Frame(self._panedwindow, padding=[1, 1, 0, 0])
        self._panedwindow.add(self._sheet_pane)
        self._panedwindow.sashpos(0, 0)  # init the sash position
        
        ### Build the initial sheet(s)
        self._sheet_kw = dict(scrollbar_bootstyle=scrollbar_bootstyle, **sheet_kw)
        self._sheet_var = vrb.DoubleVar(self)
        self._sheet_var.trace_add('write', self._select_sheet, weak=True)
        self._sheet: Sheet | None = None
        self._sheets_props: dict[float, list] = dict()
        for i, (name, df) in enumerate(dfs.items()):
            self.insert_sheet(i, name=name, df=df)
        
        
        # Focus on current sheet if any of the frames or canvas is clicked
        for widget in [
                self, self._toolbar, self._inputbar, R_label, self._r_label,
                s_label, C_label, self._c_label, self._sidebar_fm,
                self._sidebar_fm.canvas, self._sidebar
        ]:
            widget.configure(takefocus=False)
            widget.bind('<ButtonPress-1>', self._focus_on_sheet)
        
        self.bind('<<ThemeChanged>>', self._create_styles)
    
    def _create_styles(self, event=None):
        style = ttk.Style.get_instance()
        light_theme = style.theme.type.lower() == 'light'
        colors = style.colors
        
        dummy_fm = ttk.Frame(self)
        dummy_btn = ttk.Button(self, bootstyle='link-primary')
        dummy_rdbutton = ttk.Radiobutton(self, bootstyle='toolbutton-primary')
        dummy_entry = ttk.Entry(self)
        
        self._border_fm_style = 'Book.' + dummy_fm["style"]
        self._toolbar_bt_style = 'Book.' + dummy_btn["style"]
        self._switch_rb_style = 'Book.' + dummy_rdbutton["style"]
        self._entry_style = 'Book.' + dummy_entry["style"]
        
        border = colors.border if light_theme else colors.selectbg
        style.configure(self._border_fm_style, background=border)
        style.configure(self._toolbar_bt_style, padding=1)
        style.configure(
            self._switch_rb_style,
            anchor='w',
            padding=[5, 3],
            background=colors.bg,
            foreground=colors.fg,
            borderwidth=0
        )
        style.configure(self._entry_style, padding=[5, 2])
        
        dummy_fm.destroy()
        dummy_btn.destroy()
        dummy_rdbutton.destroy()
        dummy_entry.destroy()
    
    def _on_dnd_end(self, event, initial_items):
        if initial_items != self._sidebar.dnd_items:
            self._rearrange_sheets_props()
        self._focus_on_sheet()
    
    def _center_window(self, toplevel: tk.BaseWidget):
        center_window(to_center=toplevel, center_of=self.winfo_toplevel())
    
    def _rearrange_sheets_props(self, *_):
        sheets_props = self._sheets_props.copy()
        
        new_sheets_props = dict()
        for sf in self._sidebar.dnd_items:  # new order
            for key, props in sheets_props.items():
                if props["item"] == sf:
                    break
            new_sheets_props[key] = sheets_props.pop(key)
        
        self._sheets_props = new_sheets_props
    
    def _focus_on_sheet(self, *_, **__):
        self.sheet._focus()
    
    def _toggle_sidebar(self):
        if self.sidebar_hidden:  # => show sidebar
            self.show_sidebar()
        else:  # => hide sidebar
            self.hide_sidebar()
    
    def show_sidebar(self):
        """
        Show the sidebar. Calling this method may not work before `self`
        get a layout manager. Calling this method right after `self.pack`,
        `self.grid`, or `self.place` is recommended.
        """
        if not self.sidebar_hidden:
            return
        
        self.update_idletasks()  # update all widgets
        self._panedwindow.sashpos(0, self._sidebar_width)  # set the sash position
        self.update_idletasks()  # update the sash position
    
    def hide_sidebar(self):
        if self.sidebar_hidden:
            return
        
        self._sidebar_width = self._panedwindow.sashpos(0)  # save width
        self.update_idletasks()  # update all widgets
        self._panedwindow.sashpos(0, 0)  # set the sash position
        self.update_idletasks()  # update the sash position
    
    def _get_key(self, index_or_name: int | str) -> str:
        if isinstance(index_or_name, str):  # name input
            for key, props in self._sheets_props.items():
                if props["name"] == index_or_name:
                    return key
            raise ValueError(
                "Can't find the sheet with the name: {index_or_name}")
        
        # Index input
        return list(self._sheets_props)[index_or_name]
    
    def _refresh_undo_redo_buttons(self):
        undo_state = 'normal' if self.sheet._history.backable else 'disabled'
        redo_state = 'normal' if self.sheet._history.forwardable else 'disabled'
        self._undo_btn.configure(state=undo_state)
        self._redo_btn.configure(state=redo_state)
    
    def _refresh_zoom_optionmenu(self):
        self._zoom_om._variable = self.sheet._scale
        self._zoom_om.configure(textvariable=self.sheet._scale)
        self._zoom_om.set_menu(None, *self.sheet._valid_scales)
    
    def _select_sheet(self, *_) -> Sheet:
        key = self._sheet_var.get()
        
        old_sheet = self._sheet
        self._sheet = new_sheet = self._sheets_props[key]["sheet"]
        
        if old_sheet:
            old_sheet.pack_forget()
            old_sheet._history.remove_callback()
        
        new_sheet.pack(fill='both', expand=True)
        new_sheet._focus()
        new_sheet._history.set_callback(self._refresh_undo_redo_buttons)
        self._refresh_undo_redo_buttons()
        self._refresh_zoom_optionmenu()
        
        self._r_label.configure(textvariable=new_sheet._focus_row)
        self._c_label.configure(textvariable=new_sheet._focus_col)
        self._entry.configure(textvariable=new_sheet._focus_value)
        
        return new_sheet
    
    def switch_sheet(self, index_or_name: int | str) -> Sheet:
        key = self._get_key(index_or_name)
        self._sheet_var.set(key)
        
        return self._sheet
    
    def _make_unique_name(self, name: str | None = None):
        assert isinstance(name, (str, type(None))), name
        
        # Check name
        sheets_props = self._sheets_props
        names_exist = [ props["name"] for props in sheets_props.values() ]
        if name is None:
            i, name = (1, 'Sheet 1')
            while name in names_exist:
                i += 1
                name = f'Sheet {i}'
        else:
            i, prefix = (1, name)
            while name in names_exist:
                i += 1
                name = prefix + f' ({i})'
        assert name not in names_exist, names_exist
        
        return name
    
    def insert_sheet(
            self,
            index: int | None = None,
            name: str | None = None,
            dialog: bool = False,
            **kwargs
    ):
        assert isinstance(index, (int, type(None))), index
        assert isinstance(name, (str, type(None))), name
        
        sheets_props = self._sheets_props
        name = self._make_unique_name(name)
        sheet_kw = self._sheet_kw.copy()
        sheet_kw.update(kwargs)
        
        if dialog:
            top = ttk.Toplevel(
                transient=self,
                title='Add New Sheet',
                resizable=(False, False)
            )
            top.wm_withdraw()
            
            # Body
            body = ttk.Frame(top, padding=12)
            body.pack(fill='both', expand=True)
            for c in range(3):
                body.grid_rowconfigure(c, pad=6)
            body.grid_columnconfigure(0, pad=20)
            
            ## Shape
            ttk.Label(body, text='Sheet Shape (R x C)').grid(
                row=0, column=0, sticky='w')
            sb_rows = ttk.Spinbox(body, from_=1, to=100_000, increment=1, width=8)
            sb_rows.grid(row=0, column=1)
            sb_rows.set(sheet_kw["shape"][0])
            ttk.Label(body, text='x').grid(row=0, column=2)
            sb_cols = ttk.Spinbox(body, from_=1, to=100_000, increment=1, width=8)
            sb_cols.grid(row=0, column=3)
            sb_cols.set(sheet_kw["shape"][1])
            
            ## Default size
            ttk.Label(body, text='Cell Size (W x H)').grid(
                row=1, column=0, sticky='w')
            sb_w = ttk.Spinbox(body, from_=1, to=200, increment=1, width=8)
            sb_w.grid(row=1, column=1)
            sb_w.set(sheet_kw["cell_width"])
            ttk.Label(body, text=' x ').grid(row=1, column=2)
            sb_h = ttk.Spinbox(body, from_=1, to=200, increment=1, width=8)
            sb_h.grid(row=1, column=3)
            sb_h.set(sheet_kw["cell_height"])
            
            ## Min size
            ttk.Label(body, text='Minimal Cell Size (W x H)').grid(
                row=2, column=0, sticky='w')
            sb_minw = ttk.Spinbox(body, from_=1, to=200, increment=1, width=8)
            sb_minw.grid(row=2, column=1)
            sb_minw.set(sheet_kw["min_width"])
            ttk.Label(body, text=' x ').grid(row=2, column=2)
            sb_minh = ttk.Spinbox(body, from_=1, to=200, increment=1, width=8)
            sb_minh.grid(row=2, column=3)
            sb_minh.set(sheet_kw["min_height"])
            #
            submitted = False
            def _on_submit(event=None):
                # Check if the values are valid
                for which, sb in [('number of rows', sb_rows),
                                  ('number of columns', sb_cols),
                                  ('cell width', sb_w),
                                  ('cell height', sb_h),
                                  ('minimal cell width', sb_minw),
                                  ('minimal cell height', sb_minh)]:
                    if not sb.get().isnumeric():
                        dialogs.Messagebox.show_error(
                            parent=top,
                            title='Value Error',
                            message=f'The value of "{which}" must be a positive '
                                    'integer',
                            position=self._center_window
                        )
                        return
                nonlocal submitted
                submitted = True
                sheet_kw.update({
                    "shape": (int(sb_rows.get()), int(sb_cols.get())),
                    "cell_width": int(sb_w.get()),
                    "cell_height": int(sb_h.get()),
                    "min_width": int(sb_minw.get()),
                    "min_height": int(sb_minh.get())
                })
                top.destroy()
            #
            for sb in [sb_rows, sb_cols, sb_w, sb_h, sb_minw, sb_minh]:
                sb.bind('<Return>', _on_submit)
                sb.bind('<Escape>', lambda e: top.destroy())
            
            # Separator
            ttk.Separator(top, orient='horizontal').pack(fill='x')
            
            # Buttonbox
            buttonbox = ttk.Frame(top, padding=[12, 18])
            buttonbox.pack(fill='both', expand=True)
            
            ## Submit/Cancel buttons
            ttk.Button(
                buttonbox,
                text='Submit',
                bootstyle='primary',
                command=_on_submit
            ).pack(side='right')
            ttk.Button(
                buttonbox,
                text='Cancel',
                bootstyle='secondary',
                command=top.destroy
            ).pack(side='right', padx=[0, 12])
            
            self._center_window(top)
            top.wm_deiconify()
            sb_rows.select_range(0, 'end')
            sb_rows.focus_set()
            top.grab_set()
            top.wait_window()  # don't continue until the window is destroyed
            
            if not submitted:
                return self._sheets_props
        
        if index is None:
            index = len(sheets_props)
        
        # Generate a unique key
        while (key := time.time()) in sheets_props:
            time.sleep(1e-9)
            pass
        
        # Build a new sheet widget and sidebar button
        sheet = Sheet(self._sheet_pane, **sheet_kw)
        item = OrderlyDnDItem(self._sidebar, selectbutton=True, dragbutton=True)
        switch = ttk.Radiobutton(
            item,
            style=self._switch_rb_style,
            text=name,
            value=key,
            variable=self._sheet_var,
            takefocus=False
        )
        switch.pack(fill='x', expand=True)
        switch.bind(RIGHTCLICK, lambda e: self._post_switch_menu(e, key))
        
        # Modify the sheet dict
        keys, props = (list(sheets_props.keys()), list(sheets_props.values()))
        keys.insert(index, key)
        props.insert(
            index,
            {
                "name": name,
                "sheet": sheet,
                "switch": switch,
                "item": item
            }
        )
        self._sheets_props = dict(zip(keys, props))
        
        # Remap the radio buttons
        self._refresh_sidebar()
        self._sheet_var.set(key)
        
        return self._sheets_props
    
    def _refresh_sidebar(self):
        self._sidebar.dnd_forget(destroy=False)
        self._sidebar.dnd_put(
            [ props["item"] for props in self._sheets_props.values() ],
            sticky='nwe',
            expand=(True, False),
            padding=6,
            ipadding=2
        )
    
    def _post_switch_menu(self, event, key):
        # Focus on the sheet that has been clicked
        self.after_idle(self._sheet_var.set, key)
        
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label='Rename Sheet',
            command=lambda: self._rename_sheet(key)
        )
        menu.add_command(
            label='Delete Sheet',
            command=lambda: self._delete_sheet(key, request=True)
        )
        
        menu.post(event.x_root, event.y_root)  # show the right click menu
        self.after_idle(menu.destroy)
    
    def delete_sheet(
            self,
            index_or_name: int | str,
            destroy: bool = True,
            request: bool = False
    ) -> dict:
        key = self._get_key(index_or_name)
        return self._delete_sheet(key, destroy=destroy, request=request)
    
    def _delete_sheet(self, key, destroy: bool = True, request: bool = False):
        if request:
            result = dialogs.Messagebox.okcancel(
                parent=self,
                title='Sheet Deletion',
                message="This action can't be undone. "
                        "Would you like to continue?",
                icon=Icon.warning,
                alert=True,
                position=self._center_window
            )
            if result != 'OK':
                return
        
        sheets_props = self._sheets_props
        index = list(sheets_props).index(key)
        props = sheets_props.pop(key)  # remove the sheet properties
        
        # Update GUI
        if sheets_props:  # the book is not empty after deleting the sheet
            self._refresh_sidebar()
            
            # Switch sheet
            index_focus = min(index, len(sheets_props) - 1)
            self._sheet_var.set(list(sheets_props.keys())[index_focus])
        else:  # the book is empty after deleting the sheet
            self.insert_sheet()  # add a new sheet
        
        if destroy:
            for widget in (props["sheet"], props["item"]):
                widget.destroy()
        
        self.after(500, gc.collect)
        
        return props
    
    def rename_sheet(self,
                     old_name: str,
                     new_name: str | None = None,
                     auto_rename: bool = False):
        sheets_props = self._sheets_props
        key = self._get_key(old_name)
        
        if new_name == old_name:
            return new_name
        
        if auto_rename:
            new_name = self._make_unique_name(new_name)
        
        names = [ props["name"] for props in sheets_props.values() ]
        if new_name in names:
            raise DuplicateNameError(
                f"`new_name` = {new_name}. The name already exists: {names}")
        
        # Modify the sheet dict
        props = sheets_props[key]
        props["name"] = new_name
        props["switch"].configure(text=new_name)
        
        return new_name
    
    def _rename_sheet(
            self, key, _prompt='Enter a new name for this sheet:'):
        # Ask for new name
        old_name = self._sheets_props[key]["name"]
        new_name = dialogs.Querybox.get_string(
            parent=self,
            title='Rename Sheet',
            prompt=_prompt,
            initialvalue=old_name,
            width=40,
            position=self._center_window
        )
        if new_name is None:  # cancelled
            return
        
        # Submitted
        try:
            self.rename_sheet(old_name, new_name)
        except DuplicateNameError:
            self._rename_sheet(
                key,
                _prompt='The name you entered already exists. '
                       'Please enter another name for this sheet:'
            )


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    root = ttk.Window(
        title='Book (Root)',
        themename='morph',
        position=(100, 100),
        size=(800, 500)
    )
    
    
    book = Book(root, scrollbar_bootstyle='round-light')
    book.pack(fill='both', expand=True)
    book.show_sidebar()  # show the sidebar after `self.pack`
    
    book.insert_sheet(1, name='index = 1')
    book.insert_sheet(0, name='index = 0')
    book.insert_sheet(-1, name='index = -1')
    
    book.after(3000, lambda: root.style.theme_use('minty'))
    book.after(4000, lambda: root.style.theme_use('cyborg'))
    
    
    win = ttk.Toplevel(title='Sheet', position=(100, 100), size=(800, 500))
    
    sh = Sheet(win, scrollbar_bootstyle='light-round')
    sh.pack(fill='both', expand=True)
    
    sh.set_foregroundcolors(5, 3, 5, 3, colors='#FF0000', undo=True)
    sh.set_backgroundcolors(5, 3, 5, 3, colors='#2A7AD5', undo=True)
    sh.resize_cells(5, axis=0, sizes=[80], trace=None, undo=True)
    
    def _set_value_method1():
        sh.set_values(4, 3, 4, 3, values='r4, c3 (method 1)')
    
    def _set_value_method2():
        sh.values[5, 3] = 'R5, C3 (method 2)'
        sh.draw_cells(5, 3, 5, 3)
    
    sh.after(1000, _set_value_method1)
    sh.after(2000, _set_value_method2)
    
    root.mainloop()

