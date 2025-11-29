"""
Created on Mon May 22 22:35:24 2023
@author: tungchentsai
"""
import gc
import copy
import time
import tkinter as tk
import tkinter.font as tk_font
from contextlib import contextmanager
from types import NoneType
from typing import Any, TypeGuard, Literal, Generator, cast, overload
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
import ttkbootstrap as tb
from ttkbootstrap.icons import Icon
from ttkbootstrap.style import Colors
from ttkbootstrap.colorutils import color_to_hsl

from tkinter_extensions._constants import (
    MLEFTPRESS, MRIGHTPRESS, MLEFTMOTION, MLEFTRELEASE, MDLEFTPRESS, MSCROLL,
    MODIFIERS, MODIFIER_MASKS, COMMAND, SHIFT, LOCK,
    Int, _Int, Float, _Float, NpInt, _NpInt, NpFloat, _NpFloat
)
from tkinter_extensions.utils import get_modifiers, center_window, modify_hsl
from tkinter_extensions import dialogs
from tkinter_extensions import variables as vrb
from tkinter_extensions.widgets.scrolled import AutoHiddenScrollbar, ScrolledFrame
from tkinter_extensions.widgets._others import OptionMenu
from tkinter_extensions.widgets.dnd import (
    DnDItem, OrderlyDnDItem, RearrangedDnDContainer
)

_StringDType = np.dtypes.StringDType(coerce=False)
type StringArray = np.ndarray[Any, np.dtypes.StringDType]
 # `NDArray` only accept a scalar type but `StringDType` hasn't have one.
# =============================================================================
# MARK: Functions
# =============================================================================
def is_string_array(data: Any) -> TypeGuard[StringArray]:
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"`data` must be a `np.ndarray` of `{_StringDType}` dtype "
            f"but got a `{type(data)}` type."
        )
    if not np.issubdtype(data.dtype, _StringDType):
        raise TypeError(
            f"`data` must be a `np.ndarray` of `{_StringDType}` dtype "
            f"but got a `{data.dtype}` dtype."
        )
    
    return True


def array_to_string(array: StringArray) -> str:
    assert is_string_array(array)
    return '\n'.join( '\t'.join(row) for row in array )


def string_to_array(string: str) -> StringArray:
    assert isinstance(string, str), type(string)
    
    # Count the leading and trailing newline characters
    stripped = string.strip('\n')
    n_leading = i_nonempty = string.index(stripped)
    n_trailing = len(string) - len(stripped) - n_leading
    
    # Extend the rows having only 1 column of empty string.
    # This is needed to prevent the resultant array from being non-rectangular
    lists = [ row.split('\t') for row in string.split('\n') ]
    empty_row = [''] * len(lists[i_nonempty])
    for i in [*range(n_leading), *range(-n_trailing, 0)]:
        lists[i] = empty_row
    
    return cast(StringArray, np.array(lists, dtype=_StringDType))


# =============================================================================
# MARK: Classes
# =============================================================================
class DuplicateNameError(ValueError):
    pass


class _History:  #TODO: used `dataclass`es for `self._stack` and `self._sequence`
    @property
    def step(self) -> Int:
        return self._step
    
    @property
    def backable(self) -> bool:
        return bool(self._step >= 0)
    
    @property
    def forwardable(self) -> bool:
        if self._stack["forward"]:
            return bool(self._step < len(self._stack["forward"]) - 1)
        return False
    
    def __init__(
        self,
        max_height: Int,
        callback: Callable[[], Any] | None = None  #TODO: `callback_on_update(action)`
    ):
        assert callback is None or callable(callback), callback
        
        self._callback: Callable[[], Any] | None = callback
        self._sequence: dict[str, list[Callable[[], Any]]] | None = None
        self._stack: dict[str, list[Callable[[], Any]]] = {
            "forward": [], "backward": []
        }
        self._step: Int = -1
        self._max_height: Int
        self.set_max_height(max_height)
    
    def set_max_height(self, height: Int) -> None:
        assert isinstance(height, _Int), height
        assert height >= 1, height
        
        self._max_height = height
    
    def clear(self) -> None:
        self.__init__(max_height=self._max_height, callback=self._callback)
    
    def add(
        self,
        forward: Callable[[], Any],
        backward: Callable[[], Any]
    ) -> None:
        assert callable(forward) and callable(backward), (forward, backward)
        
        if self._sequence is None:  # add to the stack
            forwards = self._stack["forward"][:self._step+1] + [forward]
            backwards = self._stack["backward"][:self._step+1] + [backward]
            forwards = forwards[-self._max_height:]
            backwards = backwards[-self._max_height:]
            self._step = len(forwards) - 1
            self._stack["forward"] = forwards
            self._stack.update(forward=forwards, backward=backwards)
        else:  # add to the sequence
            self._sequence["forward"].append(forward)
            self._sequence["backward"].append(backward)
        
        if self._callback:
            self._callback()
    
    @contextmanager
    def add_sequence(self):
        yield self.start_sequence()
        self.stop_sequence()
    
    def start_sequence(self) -> dict[str, list[Callable[[], Any]]]:
        assert self._sequence is None, self._sequence
        self._sequence = {"forward": [], "backward": []}
        
        return self._sequence
    
    def stop_sequence(self) -> None:
        assert isinstance(self._sequence, dict), self._sequence
        
        forward, backward = self._sequence["forward"], self._sequence["backward"]
        
        self._sequence = None
        if forward or backward:
            self.add(
                forward=lambda: [ func() for func in forward ],
                backward=lambda: [ func() for func in backward[::-1] ]
            )
    
    def pop(self) -> dict[str, Callable[[], Any]]:
        assert self._step >= 0, self._step
        
        drop = {
            "forward": self._stack["forward"][self._step],
            "backward": self._stack["backward"][self._step]
        }
        self._stack.update(
            forward=self._stack["forward"][:self._step],
            backward=self._stack["backward"][:self._step]
        )
        self._step -= 1
        
        if self._callback:
            self._callback()
        
        return drop
    
    def back(self) -> Int:
        assert self.backable, (self._step, self._stack)
        
        self._stack["backward"][self._step]()
        self._step -= 1
        
        if self._callback:
            self._callback()
        
        return self._step
    
    def forward(self) -> Int:
        assert self.forwardable, (self._step, self._stack)
        
        self._step += 1
        self._stack["forward"][self._step]()
        
        if self._callback:
            self._callback()
        
        return self._step
    
    def set_callback(self, func: Callable[[], Any]) -> None:
        assert callable(func), func
        
        self._callback = func
    
    def get_callback(self) -> Callable[[], Any] | None:
        return self._callback
    
    def pop_callback(self) -> Callable[[], Any] | None:
        callback = self.get_callback()
        
        self._callback = None
        
        return callback


class Sheet(tb.Frame):  #TODO: allow all `ScreenUnits`
    _root: Callable[[], tb.Window]
    _valid_header_states = ('normal', 'hover', 'selected')
    _valid_cell_states = ('normal', 'readonly')
    _valid_scales = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0)
    
    @property
    def canvas(self):  # cell canvas
        return self._canvas
    
    @property
    def cornercanvas(self):
        return self._cornercanvas
    
    @property
    def rowcanvas(self):
        return self._rowcanvas
    
    @property
    def colcanvas(self):
        return self._colcanvas
    
    @property
    def hbar(self):
        return self._hbar
    
    @property
    def vbar(self):
        return self._vbar
    
    @property
    def values(self):
        return self._values
    
    @property
    def shape(self) -> tuple[Int, Int]:
        return self._values.shape
    
    @property
    def _cell_heights(self):
        return self._cell_sizes[0]
    
    @property
    def _cell_widths(self):
        return self._cell_sizes[1]
    
    def __init__(
        self,
        master: tk.Misc,
        data: StringArray | None = None,
        shape: tuple[Int, Int] | None = None,
        cell_width: Int = 80,
        cell_height: Int = 25,
        min_width: Int = 20,
        min_height: Int = 10,
        scale: Float = 1.0,
        get_style: Callable[[], dict[str, Any]] | None = None,
        max_undo: Int = 20,
        lock_number_of_rows: bool = False,
        lock_number_of_cols: bool = False,
        autohide_scrollbars: bool = True,
        scrollbar_bootstyle: str | tuple[str, ...] | None ='round',
        scroll_sensitivities:
            Float
            | tuple[Float, Float]
            = (1., 1.),
        **canvas_kw
    ):
        super().__init__(master)
        
        # Stacking order: CornerCanvas > ColCanvas > RowCanvas > CoverFrame
        # > Entry > CellCanvas
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        top_left = {"row": 0, "column": 0, "cnf": None}  # add "cnf" to workaround a type-checking bug
        self._canvas: tk.Canvas = tk.Canvas(self, **canvas_kw)  # cell canvas
        self._canvas.grid(**top_left, sticky='nesw', rowspan=2, columnspan=2)
        self._rowcanvas: tk.Canvas = tk.Canvas(self, **canvas_kw)
        self._rowcanvas.grid(**top_left, rowspan=2, sticky='nesw')
        self._colcanvas: tk.Canvas = tk.Canvas(self, **canvas_kw)
        self._colcanvas.grid(**top_left, columnspan=2, sticky='nesw')
        self._cornercanvas: tk.Canvas = tk.Canvas(self, **canvas_kw)
        self._cornercanvas.grid(**top_left, sticky='nesw')
        canvas, cornercanvas = self._canvas, self._cornercanvas
        rowcanvas, colcanvas = self._rowcanvas, self._colcanvas
        
        self._hbar: AutoHiddenScrollbar = AutoHiddenScrollbar(
            master=self,
            autohide=autohide_scrollbars,
            command=self.xview,
            bootstyle=scrollbar_bootstyle,
            orient='horizontal',
        )
        self._hbar.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(1, 0))
        self._vbar: AutoHiddenScrollbar = AutoHiddenScrollbar(
            master=self,
            autohide=autohide_scrollbars,
            command=self.yview,
            bootstyle=scrollbar_bootstyle,
            orient='vertical',
        )
        self._vbar.grid(row=0, column=2, rowspan=2, sticky='ns', padx=(1, 0))
        self._cover: tb.Frame = tb.Frame(self)  # covers the entry widget
        self._cover.grid(row=2, column=2, sticky='nesw')  # right bottom corner
        self._cover.lift()
        
        self._scroll_sensitivities: tuple[Float, Float]
        if isinstance(scroll_sensitivities, tuple):
            self.set_scroll_sensitivities(*scroll_sensitivities)
        else:
            self.set_scroll_sensitivities(
                scroll_sensitivities, scroll_sensitivities
            )
        
        # Create an invisible background (lowest item) which makes this sheet
        # become the focus if it is clicked
        for _canvas in [canvas, rowcanvas, colcanvas]:
            oid = _canvas.create_rectangle(
                0, 0, 0, 0, width=0, tags='invisible-bg'
            )
            _canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
            _canvas.tag_bind('invisible-bg', MLEFTPRESS, self._focus)
        
        # Create the selection frame
        oid = canvas.create_rectangle(0, 0, 0, 0, fill='', tags='selection-frame')
        canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
        
        # Init the backend states
        self._history: _History
        self._lock_number_of_rows: bool
        self._lock_number_of_cols: bool
        self._values: StringArray
        self._cell_sizes: tuple[NDArray[NpFloat], NDArray[NpFloat]]  # scaled sizes
        self._cell_styles: NDArray[np.object_]  # an array of dictionaries
        self._focus_old_value: str | None
        self._focus_row: vrb.IntVar
        self._focus_col: vrb.IntVar
        self._focus_value: vrb.StringVar
        self._prev_scale: Float
        self._scale: vrb.DoubleVar
        self._get_style: Callable[[], dict[str, Any]] | None
        self._default_styles: dict[str, Any]
        self._default_cell_shape: tuple[Int, Int]
        self._default_cell_sizes: tuple[Int, Int]
        self._min_sizes: tuple[Int, Int]
        self._canvas_size: tuple[Int, Int]
        self._content_size: tuple[Int, Int]
        self._view: tuple[tuple[Int, Int], tuple[Int, Int]]  # x, y views in px
        self._visible_xys: tuple[tuple[Int, Int], tuple[Int, Int]]
        self._visible_rcs: tuple[tuple[Int, Int], tuple[Int, Int]]
        self._gy2s_gx2s: tuple[NDArray[NpInt], NDArray[NpInt]]
        self._selection_rcs: tuple[Int, Int, Int, Int]
        self._resize_start: dict[str, Any] | None
        self._hover: dict[str, Any] | None
        self._motion_select_id: str | None
        
        if not ((data is None) ^ (shape is None)):
            raise TypeError(
                "If `shape` is `None`, `data` must be a numpy array. "
                "If `shape` is a 2-item tuple or list, `data` must be `None`. "
                f"But got {data} and {shape}."
            )
        if shape is None:
            assert is_string_array(data)
            shape = data.shape
        assert shape is not None, shape
        
        self._set_states(
            shape=shape,
            cell_width=cell_width,
            cell_height=cell_height,
            min_width=min_width,
            min_height=min_height,
            scale=scale,
            get_style=get_style,
            max_undo=max_undo,
            lock_number_of_rows=lock_number_of_rows,
            lock_number_of_cols=lock_number_of_cols
        )
        
        # Create an entry widget
        self._entry: tk.Entry = tk.Entry(
            self, textvariable=self._focus_value, takefocus=False
        )
        entry = self._entry
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
            for scrollseq in MSCROLL:
                widget.bind(scrollseq, self._on_mousewheel_scroll)
        
        for _canvas in [canvas, cornercanvas, rowcanvas, colcanvas]:
            _canvas.bind(MLEFTPRESS, self._on_leftbutton_press)
            _canvas.bind(MLEFTMOTION, self._on_leftbutton_motion)
            _canvas.bind(MLEFTRELEASE, self._on_leftbutton_release)
        canvas.bind(MDLEFTPRESS, self._on_double_leftclick)
        
        # Update values
        if data is not None:
            selection_rcs = self._selection_rcs  # backup selection rcs
            n_rows, n_cols = data.shape
            self.set_values(0, 0, n_rows-1, n_cols-1, data=data, draw=False)
            self._selection_rcs = selection_rcs  # restore the selection rcs
        
        # Refresh the canvases and scrollbars
        self.refresh(trace='first')
        self.focus_set()
    
    def _set_states(
        self,
        shape: tuple[Int, Int],
        cell_width: Int,
        cell_height: Int,
        min_width: Int,
        min_height: Int,
        scale: Float,
        get_style: Callable | None,
        max_undo: Int,
        lock_number_of_rows: bool,
        lock_number_of_cols: bool
    ) -> None:
        """
        Reset the cell sizes, table shape, contents, styles, and lock numbers.
        """
        assert len(shape) == 2, shape
        assert all( isinstance(s, _Int) and s > 0 for s in shape ), shape
        assert isinstance(cell_width, _Int) and cell_width >= 1, cell_width
        assert isinstance(cell_height, _Int) and cell_height >= 1, cell_height
        assert isinstance(min_width, _Int) and min_width >= 1, min_width
        assert isinstance(min_height, _Int) and min_height >= 1, min_height
        assert get_style is None or callable(get_style), get_style
        assert isinstance(lock_number_of_rows, bool), lock_number_of_rows
        assert isinstance(lock_number_of_cols, bool), lock_number_of_cols
        
        # Init the backend states
        if hasattr(self, '_history'):
            self._history.set_max_height(max_undo)
        else:
            self._history = _History(max_height=max_undo)
        self._lock_number_of_rows = lock_number_of_rows
        self._lock_number_of_cols = lock_number_of_cols
        
        self._values = np.full(shape, '', dtype=_StringDType)
        self._cell_sizes = (
            np.full(shape[0] + 1, cell_height, dtype=_NpFloat),
            np.full(shape[1] + 1, cell_width, dtype=_NpFloat)
        )
        self._cell_styles = np.array(
            [ [ {} for _c in range(shape[1]) ] for _r in range(shape[0]) ],
            dtype=dict
        )  # an array of dictionaries
        
        self._focus_old_value = None
        if not hasattr(self, '_focus_row'):
            self._focus_row = vrb.IntVar(self)
        if not hasattr(self, '_focus_col'):
            self._focus_col = vrb.IntVar(self)
        if not hasattr(self, '_focus_value'):
            self._focus_value = vrb.StringVar(self)
        self._prev_scale = 1.0
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
        self._update_content_size()
        self._view = ((0, 0), (0, 0))  # x view and y view in pixel
        self._update_visible_and_p2s()
        
        self._selection_rcs = (0, 0, 0, 0)
        self._resize_starte = None
        self._hover = None
        self._motion_select_id = None
    
    def set_scroll_sensitivities(
        self, horizontal: Float | None = None, vertical: Float | None = None
    ):
        assert isinstance(horizontal, (_Float, NoneType)), horizontal
        assert isinstance(vertical, (_Float, NoneType)), vertical
        
        self._scroll_sensitivities = (
            self._scroll_sensitivities[0] if horizontal is None else horizontal,
            self._scroll_sensitivities[1] if vertical is None else vertical
        )
        
        return self._scroll_sensitivities
    
    def set_autohide_scrollbars(
        self, enable: bool | None = None
    ) -> tuple[bool, bool]:
        self.hbar.autohide = enable
        self.vbar.autohide = enable
        
        return (self.hbar.autohide, self.vbar.autohide)
    
    def show_scrollbars(
        self, after_ms: Int = -1, autohide: bool | None = None
    ) -> None:
        self.hbar.show(after_ms, autohide=autohide)
        self.vbar.show(after_ms, autohide=autohide)
    
    def hide_scrollbars(
        self, after_ms: Int = -1, autohide: bool | None = None
    ) -> None:
        self.hbar.hide(after_ms, autohide=autohide)
        self.vbar.hide(after_ms, autohide=autohide)
    
    def lock_number_of_rows(self, lock: bool | None = None):
        assert isinstance(lock, (bool, NoneType)), lock
        
        if lock is not None:
            self._lock_number_of_rows = lock
        
        return self._lock_number_of_rows
    
    def lock_number_of_cols(self, lock: bool | None = None):
        assert isinstance(lock, (bool, NoneType)), lock
        
        if lock is not None:
            self._lock_number_of_cols = lock
        
        return self._lock_number_of_cols
    
    def __view(self, axis: Int, *args) -> tuple[Float, Float] | None:
        """Update the view from the canvas
        """
        assert axis in (0, 1), axis
        
        if not args:
            start, stop = self._view[axis]
            f1, f2 = self.__to_fractions(axis, start, stop)
            return max(f1, 0.), min(f2, 1.)
        
        action, args = args[0], args[1:]
        if action == 'moveto':
            return self.__view_moveto(axis, args[0])
        elif action == 'scroll':
            return self.__view_scroll(axis, args[0], args[1])
        raise ValueError(
            "The first argument must be 'moveto' or 'scroll' but got: "
            f"{repr(args[0])}"
        )
    
    xview = lambda self, *args: self.__view(0, *args)
    yview = lambda self, *args: self.__view(1, *args)
    
    def __view_moveto(self, axis: Int, fraction: Float) -> None:
        """Move the view from the canvas
        """
        start, = self.__to_pixels(axis, fraction)
        
        # Update the canvas and scrollbar
        self.__update_content_and_scrollbar(axis, start)
    
    xview_moveto = lambda self, *args, **kw: self.__view_moveto(0, *args, **kw)
    yview_moveto = lambda self, *args, **kw: self.__view_moveto(1, *args, **kw)
    
    #TODO: use `typing.overload` to support `ScreenUnits`
    def __view_scroll(
        self, axis: Int, number: Int, what: Literal['units', 'pages', 'pixels']
    ) -> None:
        """
        Scroll the view from the canvas.
        
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
    
    def __to_fractions(self, axis: Int, *pixels: Int) -> tuple[Float, ...]:
        assert axis in (0, 1), axis
        
        complete = self._content_size[axis]
        
        return tuple( pixel / complete for pixel in pixels )
    
    def __to_pixels(self, axis: Int, *fractions: Float) -> tuple[Int, ...]:
        assert axis in (0, 1), axis
        
        complete = self._content_size[axis]
        
        return tuple( round(fraction * complete) for fraction in fractions )
    
    def __confine_region(self, axis: Int, start: Int) -> tuple[Int, Int]:
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
    
    def __update_content_and_scrollbar(self, axis: Int, start: Int) -> None:
        new_start, new_stop = self.__confine_region(axis, start)
        old_start, _old_stop = self._view[axis]
        (r1_old, r2_old), (c1_old, c2_old) = self._visible_rcs
        view = list(self._view)
        view[axis] = (new_start, new_stop)
        self._view = cast(tuple, tuple(view))
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
        first, last = self.__to_fractions(axis, new_start, new_stop)
        (self.hbar, self.vbar)[axis].set(float(first), float(last))
    
    def _update_default_styles(self):
        def _scale_fonts(_default_styles: dict[str, Any]) -> None:
            scale = self._scale.get()
            for font in self._collect_fonts(_default_styles):
                unscaled_size = font.actual('size')
                setattr(font, '_unscaled_size', unscaled_size)
                font.configure(size=int(unscaled_size * scale))
        #
        
        if self._get_style:
            self._default_styles = default_styles = dict(self._get_style())
            assert isinstance(default_styles["header"]["font"], tk_font.Font)
            assert isinstance(default_styles["cell"]["font"], tk_font.Font)
            
            _scale_fonts(default_styles)
            
            return self._default_styles
        
        style = self._root().style
        
        # Create some dummy widgets and get the ttk style name from them
        header = tb.Checkbutton(
            self,
            bootstyle='primary-outline-toolbutton'
        )
        header_style = header["style"]
        
        cell = tb.Entry(
            self,
            bootstyle='secondary'
        )
        cell_style = cell["style"]
        
        selection = tb.Frame(
            self,
            bootstyle='primary'
        )
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
                        header_style, "background", ('hover', '!disabled')
                    ),
                    "selected": modify_hsl(
                        style.lookup(
                            header_style,
                            "background",
                            ('selected', '!disabled')
                        ),
                        func=lambda h, s, l: (h, s, min(l+header_bg_l, 200) // 2)
                    ),
                },
                "foreground": {
                    "normal": style.lookup(header_style, "foreground"),
                    "hover": style.lookup(
                        header_style, "foreground", ('hover', '!disabled')
                    ),
                    "selected": style.lookup(
                        header_style, "foreground", ('selected', '!disabled')
                    )
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
                "font": tk_font.nametofont('TkDefaultFont').copy(),
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
                "font": tk_font.nametofont('TkDefaultFont').copy(),
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
    
    def _update_content_size(self):
        self._content_size = (
            np.sum(self._cell_widths, dtype=_NpInt),
            np.sum(self._cell_heights, dtype=_NpInt)
        )
        
        return self._content_size
    
    def _update_visible_and_p2s(self):
        heights, widths = self._cell_sizes
        (gx1_view, gx2_view), (gy1_view, gy2_view) = self._view
        gx1_vis, gy1_vis = (gx1_view + widths[0], gy1_view + heights[0])
        r12: list[Any] = [None, None]
        c12: list[Any] = [None, None]
        gy2s_gx2s: list[Any] = [None, None]
        for axis, [(gp1_vis, gp2_vis), i12] in enumerate(
            zip(
                [(gy1_vis, gy2_view), (gx1_vis, gx2_view)],
                [r12, c12]
            )
        ):
            gy2s_gx2s[axis] = np.cumsum(self._cell_sizes[axis], dtype=_NpInt)
            gp2s = gy2s_gx2s[axis][1:]
            visible = (gp1_vis <= gp2s) & (gp2s <= gp2_vis)
            i12[0] = i1 = 0 if visible.all() else visible.argmax()
            i12[1] = (
                len(visible) - 1 if (tail := visible[i1:]).all()
                else tail.argmin() + i1
            )
        
        self._gy2s_gx2s = tuple(gy2s_gx2s)  # (y2s_headers, x2s_headers)
        self._visible_xys = cast(tuple, ((gx1_vis, gx2_view), (gy1_vis, gy2_view)))
        self._visible_rcs = cast(tuple, (tuple(r12), tuple(c12)))
         # ((r1, r2), (c1, c2))
        
        return self._gy2s_gx2s, self._visible_xys, self._visible_rcs
    
    def _collect_fonts(self, dictionary: dict[str, Any]) -> list[tk_font.Font]:
        fonts = []
        for value in dictionary.values():
            if isinstance(value, tk_font.Font):
                fonts.append(value)
            elif isinstance(value, dict):
                fonts.extend(self._collect_fonts(value))
        
        return fonts
    
    def _canvasx(
        self,
        xs: NDArray[NpInt] | Sequence[Int] | Sequence[NDArray[NpInt]]
    ) -> NDArray[NpInt]:
        header_width = _NpInt(self._cell_widths[0])
        (gx1, _gx2), (_gy1, _gy2) = self._visible_xys
        
        return np.asarray(xs) - gx1 + header_width  # => to canvas coordinates
    
    def _canvasy(
        self,
        ys: NDArray[NpInt] | Sequence[Int] | Sequence[NDArray[NpInt]]
    ) -> NDArray[NpInt]:
        header_height = _NpInt(self._cell_heights[0])
        (_gx1, _gx2), (gy1, _gy2) = self._visible_xys
        
        return np.asarray(ys) - gy1 + header_height  # => to canvas coordinates
    
    def _fit_size(
        self, text: str, font: tk_font.Font, width: Int, height: Int
    ) -> str:
        width, height = (max(width, 0), max(height, 0))
        canvas = self.canvas
        lines = text.split('\n')
        oid = canvas.create_text(0, 0, anchor='se', text=text, font=font)
        canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
        x1, y1, x2, y2 = canvas.bbox(oid)
        canvas.delete(oid)
        longest_line = sorted(lines, key=lambda t: len(t))[-1]
        n_chars = int( len(longest_line) / (x2 - x1) * width )
        n_lines = int( len(lines) / (y2 - y1) * height )
        
        return '\n'.join( t[:n_chars] for t in lines[:n_lines] )
    
    def _focus(self, *_) -> None:
        self._focus_out_cell()
        self.focus_set()
    
    def _center_window(self, toplevel: tk.Misc) -> None:
        center_window(to_center=toplevel, center_of=self.winfo_toplevel())
    
    @overload
    def _make_tags(
        self,
        oid: int | None = None,
        type_: str | None = None,
        subtype: str | None = None,
        row: Int | None = None,
        col: Int | None = None,
        others: tuple[str, ...] = (),
        *,
        raw: bool = True,
        to_tuple: Literal[False] = False
    ) -> dict[str, Any]: ...
    @overload
    def _make_tags(
        self,
        oid: int | None = None,
        type_: str | None = None,
        subtype: str | None = None,
        row: Int | None = None,
        col: Int | None = None,
        others: tuple[str, ...] = (),
        *,
        raw: bool = True,
        to_tuple: Literal[True]
    ) -> tuple[str, ...]: ...
    def _make_tags(
        self,
        oid=None,
        type_=None,
        subtype=None,
        row=None,
        col=None,
        others=(),
        *,
        raw=True,
        to_tuple=False
    ):
        assert isinstance(others, tuple), others
        
        tagdict: dict[str, Any] = {
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
    
    @overload
    def _make_tag(
        self,
        key: str,
        *,
        oid: int | None = None,
        type_: str | None = None,
        subtype: str | None = None,
        row: Int | None = None,
        col: Int | None = None,
        others: None = None
    ) -> str: ...
    @overload
    def _make_tag(
        self,
        key: str,
        *,
        oid: int | None = None,
        type_: str | None = None,
        subtype: str | None = None,
        row: Int | None = None,
        col: Int | None = None,
        others: tuple[str, ...]
    ) -> tuple[str, ...]: ...
    def _make_tag(
        self,
        key,
        oid=None,
        type_=None,
        subtype=None,
        row=None,
        col=None,
        others=None
    ):
        assert oid is None or isinstance(oid, int), oid
        assert type_ is None or isinstance(type_, str), type_
        assert subtype is None or isinstance(subtype, str), subtype
        assert row is None or isinstance(row, _Int), row
        assert col is None or isinstance(col, _Int), col
        assert others is None or isinstance(others, tuple), others
        
        match key:
            case 'oid':
                return f'oid={oid}'
            case 'type':
                return f'type={type_}'
            case 'subtype':
                return f'subtype={subtype}'
            case 'row':
                return f'row={row}'
            case 'col':
                return f'col={col}'
            case 'row:col':
                return f'row:col={row}:{col}'
            case 'type:row':
                return f'type:row={type_}:{row}'
            case 'type:col':
                return f'type:col={type_}:{col}'
            case 'type:row:col':
                return f'type:row:col={type_}:{row}:{col}'
            case 'type:subtype':
                return f'type:subtype={type_}:{subtype}'
            case 'type:subtype:row':
                return f'type:subtype:row={type_}:{subtype}:{row}'
            case 'type:subtype:col':
                return f'type:subtype:col={type_}:{subtype}:{col}'
            case 'type:subtype:row:col':
                return f'type:subtype:row:col={type_}:{subtype}:{row}:{col}'
            case 'others':
                return others
            case _:
                raise ValueError(f"Invalid `key`: {key}.")
    
    @overload
    def _get_tags(
        self,
        oid: int | str,
        raw: bool = False,
        *,
        to_tuple: Literal[False] = False,
        canvas: tk.Canvas | None = None,
        _key: str | None = None
    ) -> dict[str, Any]: ...
    @overload
    def _get_tags(
        self,
        oid: int | str,
        raw: bool = False,
        *,
        to_tuple: Literal[True],
        canvas: tk.Canvas | None = None,
        _key: str | None = None
    ) -> tuple[str, ...]: ...
    def _get_tags(
        self, oid, raw=False, *, to_tuple=False, canvas=None, _key=None
    ):
        assert isinstance(oid, (int, str)), oid
        assert isinstance(canvas, (NoneType, tk.Canvas)), canvas
        
        canvas = canvas or self.canvas
        
        others = ()
        tagdict: dict[str, Any] = dict.fromkeys(
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
                key, _value = tag.split('=')
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
        self, key: str, oid: int | str, canvas: tk.Canvas | None = None
    ) -> str | tuple[str, ...] | None:
        return self._get_tags(oid=oid, canvas=canvas, _key=key)[key]
    
    def _parse_raw_tag(self, key: str, tag: str | None) -> int | str | None:
        if tag is None or (_tag := tag.split('=', 1)[1]) == 'None':
            return None
        elif key in ('oid', 'row', 'col'):
            return int(_tag)
        return _tag
    
    @overload
    def _get_rc(
        self,
        oid_or_tagdict: int | str | dict,
        to_tuple: Literal[False] = False,
        canvas: tk.Canvas | None = None
    ) -> dict[str, int]: ...
    @overload
    def _get_rc(
        self,
        oid_or_tagdict: int | str | dict,
        to_tuple: Literal[True],
        canvas: tk.Canvas | None = None
    ) -> tuple[int, int]: ...
    def _get_rc(self, oid_or_tagdict, to_tuple=False, canvas=None):
        assert isinstance(oid_or_tagdict, (int, str, dict)), oid_or_tagdict
        
        if isinstance(oid_or_tagdict, dict):
            tagdict = oid_or_tagdict
        else:
            tagdict = self._get_tags(oid_or_tagdict, canvas=canvas)
        
        row, col = cast(tuple, (tagdict["row"], tagdict["col"]))
        
        if to_tuple:
            return (row, col)
        return {"row": row, "col": col}
    
    def _get_rcs(self, tag: str) -> dict[str, set[int]]:  # currently not used
        return dict(zip(
            ["rows", "cols"],
            map(
                set,
                zip(*[
                    self._get_rc(oid, to_tuple=True)
                    for oid in self.canvas.find_withtag(tag)
                ])
            )
        ))
    
    def _build_general_rightclick_menu(
        self, menu: tk.Menu | None = None
    ) -> tk.Menu:
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
            command=lambda: self._selection_set_colors(
                field='foreground', dialog=True, undo=True
            )
        )
        menu_textcolor.add_command(
            label='Reset Color(s)',
            command=lambda: self._selection_set_colors(
                field='foreground', undo=True
            )
        )
        menu.add_cascade(label='Text Color(s)', menu=menu_textcolor)
        
        # Change background colors
        menu_bgcolor = tk.Menu(menu, tearoff=0)
        menu_bgcolor.add_command(
            label='Choose Color...',
            command=lambda: self._selection_set_colors(
                field='background', dialog=True, undo=True
            )
        )
        menu_bgcolor.add_command(
            label='Reset Color(s)',
            command=lambda: self._selection_set_colors(
                field='background', undo=True)
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
                "alignx", 'w', undo=True
            )
        )
        menu_align.add_command(
            label='→ Right',
            command=lambda: self._selection_set_styles(
                "alignx", 'e', undo=True
            )
        )
        menu_align.add_command(
            label='⏐ Center',
            command=lambda: self._selection_set_styles(
                "alignx", 'center', undo=True
            )
        )
        menu_align.add_command(
            label='⤬ Reset',
            command=lambda: self._selection_set_styles(
                "alignx", None, undo=True
            )
        )
        menu_align.add_separator()
        menu_align.add_command(
            label='↑ Top',
            command=lambda: self._selection_set_styles(
                "aligny", 'n', undo=True
            )
        )
        menu_align.add_command(
            label='↓ Bottom',
            command=lambda: self._selection_set_styles(
                "aligny", 's', undo=True
            )
        )
        menu_align.add_command(
            label='⎯ Center',
            command=lambda: self._selection_set_styles(
                "aligny", 'center', undo=True
            )
        )
        menu_align.add_command(
            label='⤬ Reset',
            command=lambda: self._selection_set_styles(
                "aligny", None, undo=True
            )
        )
        menu.add_cascade(label='Align', menu=menu_align)
        
        # Reset styles
        menu.add_command(
            label='Reset Style(s)',
            command=lambda: self._selection_reset_styles(undo=True)
        )
        
        return menu
    
    def _redirect_widget_event(  # currently not used
        self, event: tk.Event
    ) -> tk.Event:
        widget, canvas = (event.widget, self.canvas)
        event.x += widget.winfo_x() - canvas.winfo_x()
        event.y += widget.winfo_y() - canvas.winfo_y()
        event.widget = canvas
        
        return event
    
    def _on_theme_changed(self, event: tk.Event | None = None) -> None:
        self._update_default_styles()
        self.refresh()
    
    def _on_canvas_configured(self, event: tk.Event) -> None:
        self._canvas_size = canvas_size = (event.width, event.height)
        self.canvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.rowcanvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.colcanvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.xview_scroll(0, 'units')
        self.yview_scroll(0, 'units')
    
    def _on_mousewheel_scroll(self, event: tk.Event) -> None:
        """
        Callback for when the mouse wheel is scrolled.
        Modified from: `ttkbootstrap.scrolled.ScrolledFrame._on_mousewheel`
        """
        assert isinstance(event_state := event.state, int), event_state
        
        if event.num == 4:  # Linux
            delta = 10.
        elif event.num == 5:  # Linux
            delta = -10.
        elif self._windowingsystem == "win32":  # Windows
            delta = event.delta / 120.
        else:  # macOS
            delta = event.delta
        
        x_direction = (
            (event_state & MODIFIER_MASKS["Shift"]) == MODIFIER_MASKS["Shift"]
        )
        sensitivity = self._scroll_sensitivities[0 if x_direction else 1]
        number = -round(delta * sensitivity * 2.)
        
        if x_direction:
            self.xview_scroll(number, 'units')
        else:
            self.yview_scroll(number, 'units')
    
    def _on_select_all(self, event: tk.Event | None = None) -> None:
        self.select_cells()
    
    def _on_copy(self, event: tk.Event | None = None) -> None:
        self._selection_copy_values()
    
    def _on_paste(self, event: tk.Event | None = None) -> None:
        self._selection_paste_values(undo=True)
    
    def _on_entry_key_press(self, event: tk.Event) -> Literal['break'] | None:
        assert isinstance(event_state := event.state, int), event_state
        
        keysym = event.keysym
        modifiers = get_modifiers(event_state)
        
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
    
    def _on_key_press(self, event: tk.Event) -> Literal['break'] | None:
        assert isinstance(event_state := event.state, int), event_state
        
        keysym, char = event.keysym, event.char
        modifiers = get_modifiers(event_state)
        
        if self._on_entry_key_press(event):  # same as entry events
            return 'break'
        
        elif keysym in ('Up', 'Down', 'Left', 'Right'):  # move selection
            direction = cast(
                Literal['up', 'down', 'left', 'right'], keysym.lower()
            )
            area = 'paragraph' if COMMAND in modifiers else None
            expand = SHIFT in modifiers
            self._move_selections(direction, area=area, expand=expand)
            return 'break'
        
        elif keysym in ('Home', 'End', 'Prior', 'Next'):  # move selection
            direction = cast(
                Literal['up', 'down', 'left', 'right'],
                {
                    "Home": 'left',
                    "End": 'right',
                    "Prior": 'up',
                    "Next": 'down'
                }[keysym]
            )
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
        self, x: Int, y: Int, canvas: tk.Canvas, expand: bool, dry: bool = False
    ) -> tuple[int | None, int | None, int | None, int | None]:
        gy2s, gx2s = self._gy2s_gx2s
        x2s, y2s = (self._canvasx(gx2s[1:]), self._canvasy(gy2s[1:]))
        (r1_vis, r2_vis), (c1_vis, c2_vis) = self._visible_rcs
        
        if canvas in (self.cornercanvas, self.colcanvas):
            r2 = None
        else:
            r2 = int(
                np.clip(
                    above.argmax() if (above := y <= y2s).any() else y2s.size - 1,
                    r1_vis,
                    r2_vis,
                    dtype=_NpInt
                )
            )
        if canvas in (self.cornercanvas, self.rowcanvas):
            c2 = None
        else:
            c2 = int(
                np.clip(
                    left.argmax() if (left := x <= x2s).any() else x2s.size - 1,
                    c1_vis,
                    c2_vis,
                    dtype=_NpInt
                )
            )
        
        if expand:
            r1, c1, *_ = self._selection_rcs
            r1, c1 = int(r1), int(c1)
        else:
            r1, c1 = (r2, c2)
        
        if not dry:
            self.select_cells(r1, c1, r2, c2)
        
        return (r1, c1, r2, c2)
    
    def _on_leftbutton_press(self, event: tk.Event, select: bool = True):
        assert isinstance(canvas := event.widget, tk.Canvas), canvas
        
        self._focus()
        x, y = (event.x, event.y)
        
        return self.__mouse_select(
            x, y, canvas=canvas, expand=False, dry=not select
        )
    
    def _on_leftbutton_motion(
        self, event: tk.Event, _dxdy: tuple[int, int] | None = None
    ) -> None:
        # Move the viewing window if the mouse cursor is moving outside the 
        # canvas
        assert isinstance(canvas := event.widget, tk.Canvas), canvas
        
        x, y = (event.x, event.y)
        
        if _dxdy is None:
            heights, widths = self._cell_sizes
            top_bd, left_bd = (heights[0], widths[0])  # headers' bottom/right
            right_bd, bottom_bd = self._canvas_size
            dx = int(x - left_bd if x < left_bd else max(x - right_bd, 0)) // 2
            dy = int(y - top_bd if y < top_bd else max(y - bottom_bd, 0)) // 8
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
    
    def _on_leftbutton_release(self, event: tk.Event | None = None) -> None:
        # Remove the autoscroll function loop
        if (funcid := self._motion_select_id) is not None:
            self.after_cancel(funcid)
            self._motion_select_id = None
    
    def _on_double_leftclick(self, event: tk.Event) -> None:
        assert event.widget == self.canvas, event.widget
        self._focus_in_cell()
    
    def _index_generator(
        self, start: Int, last: Int
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
        assert isinstance(start, _Int), (type(start), start)
        assert isinstance(last, _Int), (type(last), last)
        assert 0 <= start <= last, (start, last)
        
        n_half = (start + last) // 2
        upper_idc = list(range(start, n_half+1, +1))
        lower_idc = list(range(last, n_half, -1))
        indices = [0] * (len(upper_idc) + len(lower_idc))
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
    
    def draw_cornerheader(self, skip_existing: bool = False) -> None:
        assert isinstance(skip_existing, bool), skip_existing
        
        type_ = 'cornerheader'
        x1, y1, x2, y2 = (
            0, 0, int(self._cell_widths[0]), int(self._cell_heights[0])
        )
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
        
        tag = self._make_tag('type', type_=type_)
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
        canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
        
        ## Handles (invisible)
        tags = self._make_tags(subtype='hhandle', **kw)
        oid_hh = canvas.create_line(
            x1, y2, x2, y2, width=handle["width"], fill='', tags=tags)
        canvas.addtag_withtag(self._make_tag('oid', oid=oid_hh), oid_hh)
        
        tags = self._make_tags(subtype='vhandle', **kw)
        oid_vh = canvas.create_line(
            x2, y1, x2, y2, width=handle["width"], fill='', tags=tags)
        canvas.addtag_withtag(self._make_tag('oid', oid=oid_vh), oid_vh)
        
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
        canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
        
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
        canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
        
        ## Handles > separators > background
        canvas.tag_raise(oid_hh)
        canvas.tag_raise(oid_vh)
        
        # Bindings
        tag_cornerheader = self._make_tag('type', type_=type_)
        canvas.tag_bind(tag_cornerheader, '<Enter>', self._on_header_enter)
        canvas.tag_bind(tag_cornerheader, '<Leave>', self._on_header_leave)
        canvas.tag_bind(
            tag_cornerheader, MRIGHTPRESS, self._on_header_rightbutton_press
        )
        
        for handle in ['hhandle', 'vhandle']:
            tag_cornerhandle = self._make_tag(
                'type:subtype', type_=type_, subtype=handle
            )
            canvas.tag_bind(
                tag_cornerhandle,
                MLEFTPRESS,
                getattr(self, f'_on_{handle}_leftbutton_press')
            )
    
    def draw_headers(
        self,
        i1: Int | None = None,
        i2: Int | None = None,
        *,
        axis: Int,
        skip_existing: bool = False
    ) -> None:
        assert (i1 is not None) or (i2 is None), (i1, i2)
        assert axis in (0, 1), axis
        assert isinstance(skip_existing, bool), skip_existing
        
        r12_vis, c12_vis = self._visible_rcs
        i1_vis, i2_vis = r12_vis if axis == 0 else c12_vis
        
        if i1 is None:
            i1, i2 = (i1_vis, i2_vis)
        elif i2 is None:
            i2 = i2_vis
        i1, i2 = sorted([ int(np.clip(i, i1_vis, i2_vis)) for i in (i1, i2) ])
        assert i1 is not None and i2 is not None, (i1, i2)
        
        max_i = self.shape[axis] - 1
        assert 0 <= i1 <= i2 <= max_i, (i1, i2, max_i)
        
        style = self._default_styles["header"]
        background, foreground = style["background"], style["foreground"]
        bordercolor, font = style["bordercolor"], style["font"]
        w_hand = style["handle"]["width"]
        separator = style["separator"]
        dw_sep = separator["width"] // 2
        
        heights, widths = self._cell_sizes
        heights, widths = heights.astype(_NpInt), widths.astype(_NpInt)
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
            tag = self._make_tag('type:row:col', **kw)
            
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
            canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
            
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
            canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
            
            ### Handle (invisible)
            tags = self._make_tags(subtype=handle, **kw)
            oid = canvas.create_line(
                *xys_handle, width=w_hand, fill='', tags=tags)
            canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
            
            try:
                i = i_generator.send(False)  # get next `i`
            except StopIteration:
                break  # stop this iteration
        
        # Separator (always redrawn)
        kw.pop("to_tuple", None)
        kw.pop("row" if axis == 0 else "col")
        tag = self._make_tag('type:subtype', subtype='separator', **kw)
        canvas.delete(tag)
        tags = self._make_tags(subtype='separator', to_tuple=True, **kw)
        oid_sep = canvas.create_line(
            *xys_sep,
            width=separator["width"],
            fill=separator["color"],
            tags=tags
        )
        canvas.addtag_withtag(self._make_tag('oid', oid=oid_sep), oid_sep)
        
        # Stacking order: CornerHeader > Row/ColSeparator > Row/ColHeaderHandle
        # > Row/ColHeader
        tag_bg = self._make_tag('type:subtype', type_=type_, subtype='background')
        tag_text = self._make_tag('type:subtype', type_=type_, subtype='text')
        tag_handle = self._make_tag('type:subtype', type_=type_, subtype=handle)
        canvas.tag_raise(tag_bg)
        canvas.tag_raise(tag_text)
        canvas.tag_raise(tag_handle)  # second from the front
        canvas.tag_raise(oid_sep)  # frontmost
        
        # Bindings
        tag_header = self._make_tag('type', type_=type_)
        canvas.tag_bind(tag_header, '<Enter>', self._on_header_enter)
        canvas.tag_bind(tag_header, '<Leave>', self._on_header_leave)
        canvas.tag_bind(
            tag_header, MRIGHTPRESS, self._on_header_rightbutton_press
        )
        canvas.tag_bind(
            tag_handle,
            MLEFTPRESS,
            getattr(self, f'_on_{handle}_leftbutton_press')
        )
    
    def _set_header_state(
        self,
        tagdict: dict[str, Any],
        state: Literal['normal', 'hover', 'selected']
    ) -> None:
        assert state is None or state in self._valid_header_states, state
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), type_
        
        if type_ == 'cornerheader':
            canvas = self.cornercanvas
            tag = self._make_tag('type', type_=type_)
            tag_bg = self._make_tag(
                'type:subtype', type_=type_, subtype='background'
            )
            tag_text = None
        else:
            if type_ == 'rowheader':
                col_or_row = 'row'
                canvas = self.rowcanvas
            else:
                col_or_row = 'col'
                canvas = self.colcanvas
            
            kw = {"type_": type_, col_or_row: tagdict[col_or_row]}
            tag = self._make_tag(f'type:{col_or_row}', **kw)
            tag_bg = self._make_tag(
                f'type:subtype:{col_or_row}', subtype='background', **kw
            )
            tag_text = self._make_tag(
                f'type:subtype:{col_or_row}', subtype='text', **kw
            )
        
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
            assert tag_text is not None, tag_text
            
            # Update the stacking order
            ## Front to back: separator > handlers > texts > backgrounds
            ## > invisible-bg
            if state in ('selected', 'hover'):
                tag_all_text = self._make_tag(
                    'type:subtype', type_=type_, subtype='text'
                )
                canvas.tag_lower(tag_bg, tag_all_text)  # frontmost background
            else:
                try:
                    # Lower than the selected
                    canvas.tag_lower(tag_bg, 'selected')
                except tk.TclError:
                    pass  # the selected not in the view
            
            # Update the text color
            canvas.itemconfigure(tag_text, fill=style["foreground"][state])
    
    def __on_header_enter_leave(
        self, event: tk.Event, enter_or_leave: Literal['enter', 'leave']
    ) -> None:
        assert enter_or_leave in ('enter', 'leave'), enter_or_leave
        assert isinstance(canvas := event.widget, tk.Canvas), canvas
        
        # Set header state
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
        event, 'enter'
    )
    _on_header_leave = lambda self, event: self.__on_header_enter_leave(
        event, 'leave'
    )
    
    def _on_header_rightbutton_press(self, event: tk.Event) -> None:
        assert (tagdict := self._hover) is not None, tagdict
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), self._hover
        
        # Select the current row/col if it is not selected
        _r1, _c1, _r2, _c2 = self._selection_rcs
        (r_low, r_high), (c_low, c_high) = sorted([_r1, _r2]), sorted([_c1, _c2])
        r1, c1, r2, c2 = self._on_leftbutton_press(event, select=False)
        r_max, c_max = [ s - 1 for s in self.shape ]
        
        if type_ == 'rowheader':
            assert r1 is not None, r1
            
            if (c_low, c_high) != (0, c_max) or not r_low <= r1 <= r_high:
                self.select_cells(r1, c1, r2, c2)
            
            axis_name, axis = ('Row', 0)
            modifiable_nheaders = (
                'disabled' if self._lock_number_of_rows else 'active'
            )
        elif type_ == 'colheader':
            assert c1 is not None, c1
            
            if (r_low, r_high) != (0, r_max) or not c_low <= c1 <= c_high:
                self.select_cells(r1, c1, r2, c2)
            
            axis_name, axis = ('Column', 1)
            modifiable_nheaders = (
                'disabled' if self._lock_number_of_cols else 'active'
            )
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
    
    def __on_handle_leftbutton_press(  # resize starts
        self, event: tk.Event, axis: Int
    ) -> None:
        assert (tagdict := self._hover) is not None, tagdict
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), self._hover
        assert axis in (0, 1), axis
        
        r, c = self._get_rc(tagdict, to_tuple=True)
        
        # Bind (overwrite) the event functions with the handle callbacks
        canvas = event.widget
        old_leftmotion = canvas.bind(MLEFTMOTION)
        old_leftrelease = canvas.bind(MLEFTRELEASE)
        if axis == 0:
            i = r
            rcs = (r, None, r, None)
            canvas.bind(MLEFTMOTION, self._on_hhandle_leftbutton_motion)
            canvas.bind(MLEFTRELEASE, self._on_handle_leftbutton_release)
        else:
            i = c
            rcs = (None, c, None, c)
            canvas.bind(MLEFTMOTION, self._on_vhandle_leftbutton_motion)
            canvas.bind(MLEFTRELEASE, self._on_handle_leftbutton_release)
        self.after_idle(self.select_cells, *rcs)
        
        _i = i + 1
        self._resize_start = {
            "x": event.x,
            "y": event.y,
            "i": i,
            "size": self._cell_sizes[axis][_i],  # scaled size
            "scale": self._scale.get(),
            "step": self._history.step,
            "leftmotion": old_leftmotion,
            "leftrelease": old_leftrelease
        }
    
    _on_hhandle_leftbutton_press = lambda self, event: (
        self.__on_handle_leftbutton_press(event, axis=0)
    )
    _on_vhandle_leftbutton_press = lambda self, event: (
        self.__on_handle_leftbutton_press(event, axis=1)
    )
    
    def __on_handle_leftbutton_motion(  # resizing
        self, event: tk.Event, axis: Int
    ) -> None:
        assert (start := self._resize_start) is not None, start
        
        scale = start["scale"]
        if axis == 0:
            new_size = start["size"] + event.y - start["y"]
        else:
            new_size = start["size"] + event.x - start["x"]
        
        # Scaled size => unscaled size
        unscaled_start_size = round(start["size"] / scale)
        unscaled_new_size = round(new_size / scale)
        
        self.resize_cells(
            start["i"], axis=axis, N=1, sizes=[unscaled_new_size], undo=False
        )
        
        history = self._history
        if history.step > start["step"]:
            history.pop()
        history.add(
            forward=lambda: self.resize_cells(
                start["i"],
                axis=axis,
                N=1,
                sizes=[unscaled_new_size],
                trace='first'
            ),
            backward=lambda: self.resize_cells(
                start["i"], axis=axis, sizes=[unscaled_start_size], trace='first'
            )
        )
    
    _on_hhandle_leftbutton_motion = lambda self, event: (
        self.__on_handle_leftbutton_motion(event, axis=0))
    _on_vhandle_leftbutton_motion = lambda self, event: (
        self.__on_handle_leftbutton_motion(event, axis=1))
    
    def _on_handle_leftbutton_release(  # resize ends
        self, event: tk.Event
    ) -> None:
        # Overwrite the handle callbacks with the originals
        assert (start := self._resize_start) is not None, start
        
        event.widget.bind(MLEFTMOTION, start["leftmotion"])
        event.widget.bind(MLEFTRELEASE, start["leftrelease"])
        self._resize_start = None
    
    def draw_cells(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        skip_existing: Literal['row', 'col'] | None = None,
    ) -> None:
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
        r1, r2 = sorted([ int(np.clip(r, r1_vis, r2_vis)) for r in (r1, r2) ])
        c1, c2 = sorted([ int(np.clip(c, c1_vis, c2_vis)) for c in (c1, c2) ])
        r_max, c_max = [ s - 1 for s in self.shape ]
        assert 0 <= r1 <= r2 <= r_max, (r1, r2, r_max)
        assert 0 <= c1 <= c2 <= c_max, (c1, c2, c_max)
        
        heights, widths = self._cell_sizes
        heights, widths = heights.astype(_NpInt), widths.astype(_NpInt)
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
                kw = {
                    "row": r, "col": c,
                    "others": ('xscroll', 'yscroll', 'temp')
                }
                tag = self._make_tag('type:row:col', type_=type_, **kw)
                
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
                canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
                
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
                    'right' if 'e' in anchor else 'center'
                )
                anchor = (
                    'center' if anchor[0] == anchor[1] else ''.join(anchor)[::-1]
                )
                assert anchor in (
                    'nw', 'n', 'ne', 'w', 'center', 'e', 'sw', 's', 'se'
                ), anchor
                
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
                canvas.addtag_withtag(self._make_tag('oid', oid=oid), oid)
                try:
                    c = c_generator.send(False)  # get next `c`
                except StopIteration:
                    break  # stop this column iteration
            
            try:
                r = r_generator.send(skip_this_row)  # get next `r`
            except StopIteration:
                break  # stop this row iteration
        
        # Bindings
        tag_cell = self._make_tag('type', type_=type_)
        canvas.tag_bind(tag_cell, MRIGHTPRESS, self._on_cell_rightbutton_press)
        
        # Keep the selection frame on the top
        canvas.tag_raise('selection-frame')
    
    def _on_cell_rightbutton_press(self, event: tk.Event) -> None:
        tagdict = self._get_tags('current')
        r, c = self._get_rc(tagdict, to_tuple=True)
        r1, c1, r2, c2 = self._selection_rcs
        (r_low, r_high), (c_low, c_high) = sorted([r1, r2]), sorted([c1, c2])
        
        if not ((r_low <= r <= r_high) and (c_low <= c <= c_high)):
            self.select_cells(r, c, r, c)
        
        menu = self._build_general_rightclick_menu()
        menu.post(event.x_root, event.y_root)
        self.after_idle(menu.destroy)
    
    def _refresh_entry(
        self, r: Int | None = None, c: Int | None = None
    ) -> None:
        assert (r is None) == (c is None), (r, c)
        
        if (r is None) and (c is None):
            r1, c1, r2, c2 = self._selection_rcs
            r, c = min(r1, r2), min(c1, c2)
        assert r is not None and c is not None, (r, c)
        
        heights, widths = self._cell_sizes
        heights, widths = heights.astype(_NpInt), widths.astype(_NpInt)
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
        
        match cell_style["alignx"]:
            case 'w':
                justify = 'left'
            case 'e':
                justify = 'right'
            case _:
                justify = 'center'
        
        en = self._entry
        en.configure(
            background=cell_style["background"],
            foreground=cell_style["foreground"],
            highlightcolor=cell_style["bordercolor"],
            font=cell_style["font"],
            justify=justify
        )
        en.place(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)
        en.lift(self.canvas)
        self._focus_old_value = old_text
    
    def _focus_in_cell(
        self, r: Int | None = None, c: Int | None = None
    ) -> None:
        self._focus_out_cell()
        self._refresh_entry(r, c)
        self._entry.focus_set()
    
    def _focus_out_cell(self, discard: bool = False) -> None:
        if self._focus_old_value is not None:
            r, c = self._focus_row.get(), self._focus_col.get()
            old_value = self._focus_old_value
            rcs = (r, c, r, c)
            if (not discard) and (new_value := self._entry.get()) != old_value:
                # Apply the new value
                self.set_values(*rcs, data=new_value, draw=False)
                self.draw_cells(*rcs)
                 # put draw here to avoid recursive function calls
                self._history.add(
                    forward=lambda: self.set_values(
                        *rcs, data=new_value, trace='first'
                    ),
                    backward=lambda: self.set_values(
                        *rcs, data=old_value, trace='first'
                    )
                )
            else:  # Restore the old value
                self._entry.delete(0, 'end')
                self._entry.insert('end', old_value)
            self._focus_old_value = None
        
        self._entry.lower()
        self.focus_set()
    
    def _set_selection(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None
    ):
        assert (r1 is not None) or (r2 is None), (r1, r2)
        assert (c1 is not None) or (c2 is None), (c1, c2)
        
        r_max, c_max = [ s - 1 for s in self.shape ]
        r_max, c_max = int(r_max), int(c_max)
        r1 = 0 if r1 is None else int(np.clip(r1, 0, r_max))
        c1 = 0 if c1 is None else int(np.clip(c1, 0, c_max))
        r2 = r_max if r2 is None else int(np.clip(r2, 0, r_max))
        c2 = c_max if c2 is None else int(np.clip(c2, 0, c_max))
        
        self._selection_rcs = (r1, c1, r2, c2)
        
        return self._selection_rcs
    
    def _update_selection_tags(self) -> tuple[set[int], set[int]]:
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
            tag = self._make_tag('type', type_='cornerheader')
            cornercanvas.addtag_withtag('selected', tag)
        
        (r1_vis, r2_vis), (c1_vis, c2_vis) = self._visible_rcs
        rows_on = set(range(r_low, r_high+1)) & set(range(r1_vis, r2_vis+1))
        cols_on = set(range(c_low, c_high+1)) & set(range(c1_vis, c2_vis+1))
        
        for r in rows_on:
            tag = self._make_tag('type:row', type_='rowheader', row=r)
            rowcanvas.addtag_withtag('selected', tag)
        for c in cols_on:
            tag = self._make_tag('type:col', type_='colheader', col=c)
            colcanvas.addtag_withtag('selected', tag)
        
        return (rows_on, cols_on)
    
    def select_cells(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        trace: Literal['first', 'last'] | None = None
    ):
        assert trace in ('first', 'last', None), trace
        
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
            heights, widths = heights.astype(_NpInt), widths.astype(_NpInt)
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
        *self._selection_rcs, *args, **kw
    )
    
    def _move_selections(
        self,
        direction: Literal['up', 'down', 'left', 'right'],
        area: Literal['paragraph', 'all'] | None = None,
        expand: bool = False
    ):
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
                values = values.ravel()[::-1]  # flip
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
                values = values.ravel()
            
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
        skip_existing: Literal['row', 'col'] | None = None,
        trace: Literal['first', 'last'] | None = None
    ) -> None:
        assert skip_existing in ('row', 'col', None), skip_existing
        
        if update_visible_rcs:
            self._update_visible_and_p2s()
        self.draw_cornerheader(skip_existing=skip_existing is not None)
        self.draw_headers(axis=0, skip_existing=skip_existing == 'row')
        self.draw_headers(axis=1, skip_existing=skip_existing == 'col')
        self.draw_cells(skip_existing=skip_existing)
        self._reselect_cells(trace=trace)
    
    def refresh(self, scrollbar: str = 'both', trace: Literal['first', 'last'] | None = None):
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
    
    def zoom(self, scale: Float = 1.) -> None:
        """
        Zoom in when `scale` > 1.0 or zoom out when `scale` < 1.0.
        """
        assert scale in self._valid_scales, [self._valid_scales, scale]
        
        self._scale.set(float(scale))
    
    def _zoom(self, *_) -> None:
        # Update scale state
        scale = self._scale.get()
        ratio = scale / self._prev_scale
        self._prev_scale = scale
        
        # Update cell sizes
        self._cell_sizes = (self._cell_heights * ratio, self._cell_widths * ratio)
        self._update_content_size()
        
        # Update text sizes
        ## Default fonts
        for font in self._collect_fonts(self._default_styles):
            unscaled_size = getattr(font, '_unscaled_size')
            font.configure(size=int(unscaled_size * scale))
        
        ## Cell fonts
        for row_styles in self._cell_styles:
            for style in row_styles:
                if font := style.get("font"):
                    unscaled_size = getattr(font, '_unscaled_size')
                    font.configure(size=int(unscaled_size * scale))
        
        # Refresh canvases
        self.refresh()
    
    def _zoom_to_next_scale_level(
        self, direction: Literal['up', 'down'] = 'up'
    ) -> None:
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
        i: Int,
        axis: Int,
        N: Int = 1,
        sizes: NDArray[NpInt] | Sequence[Int]| None = None,  # unscaled sizes
        dialog: bool = False,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> NDArray[NpInt] | None:
        assert axis in (0, 1), axis
        max_i = self.shape[axis] - 1
        assert -1 <= i <= max_i + 1, (i, max_i)
        assert N >= 1, N
        
        scale = self._scale.get()
        _idc = np.arange(i+1, i+N+1)
        old_sizes = self._cell_sizes[axis][_idc]  # scaled
        old_sizes_unscaled = np.round(old_sizes / scale).astype(_NpInt)
        
        # Check for the new sizes
        if dialog:  # ask for new size
            dimension = ('height', 'width')[axis]
            size = dialogs.Querybox.get_integer(
                parent=self,
                prompt=f'Enter the new {dimension}:',
                initialvalue=old_sizes_unscaled[0],
                width=40,
                position=self._center_window
            )
            if not isinstance(size, int):  # cancelled
                return
            new_sizes_unscaled = sizes = np.full(N, size, dtype=_NpInt)
        elif sizes is None:  # reset the rows or cols sizes
            new_sizes_unscaled = np.full(
                N, self._default_cell_sizes[axis], dtype=_NpInt
            )
        else:  # new size
            new_sizes_unscaled = np.asarray(sizes, dtype=_NpInt)
            assert np.shape(new_sizes_unscaled) == (N,), (sizes, N)
        new_sizes = new_sizes_unscaled * scale  # scaled
        
        # Update the status of the resized rows or cols
        key = ("row", "col")[axis]
        min_size = self._min_sizes[axis] * scale
        deltas = np.maximum(new_sizes - old_sizes, min_size - old_sizes)
        self._cell_sizes[axis][_idc] += deltas
        self._update_content_size()
        
        # Move the bottom rows or right cols
        (_r1_vis, r2_vis), (_c1_vis, c2_vis) = self._visible_rcs
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
                    i, axis=axis, N=N, sizes=copy.copy(sizes), trace='first'
                ),
                backward=lambda: self.resize_cells(
                    i,
                    axis=axis,
                    N=N,
                    sizes=old_sizes_unscaled.copy(),
                    trace='first'
                )
            )
        
        return new_sizes_unscaled
    
    def insert_cells(
        self,
        i: Int | None = None,
        *,
        axis: Int,
        N: Int = 1,
        data: StringArray | None = None,
        sizes: NDArray[NpInt] | Sequence[NDArray[NpInt]] | None = None,  # unscaled
        styles=None,
        dialog: bool = False,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        assert axis in (0, 1), axis
        old_df, old_shape = self.values, self.shape
        max_i = old_shape[axis] - 1
        i = max_i + 1 if i is None else i
        assert i is not None, i
        assert 0 <= i <= max_i + 1, (i, max_i)
        
        if dialog:
            # Ask for number of rows/cols to insert
            axis_name = ('rows', 'columns')[axis]
            _N = dialogs.Querybox.get_integer(
                parent=self,
                title='Rename Sheet',
                prompt=f'Enter the number of {axis_name} to insert:',
                initialvalue=1,
                minvalue=1,
                maxvalue=100000,
                width=40,
                position=self._center_window
            )
            if not isinstance(_N, int):
                return
            N = _N
        assert N >= 1, N
        
        new_shape = list(old_shape)
        new_shape[axis] = N
        new_shape = tuple(new_shape)
        
        # Create a list of new sizes (a 1-D list)
        if sizes is None:
            unscaled_new_size = self._default_cell_sizes[axis]
            new_sizes_unscaled = np.array(
                [ unscaled_new_size for _ in range(N) ], dtype=_NpInt
            )
        else:
            assert np.shape(sizes) == (N,), (sizes, N)
            new_sizes_unscaled = np.asarray(sizes, dtype=_NpInt)
        new_sizes = new_sizes_unscaled * self._scale.get()  # scaled
        
        # Create a list of new styles
        if styles is None:
            new_styles = np.array(
                [
                    [ {} for _c in range(new_shape[1]) ]
                    for _r in range(new_shape[0])
                ]
            )
        else:
            assert np.shape(styles) == new_shape, styles
            new_styles = np.asarray(styles)
        
        # Create a dataframe containing the new values (a 2-D dataframe)
        if data is None:
            inserted_data = np.full(new_shape, '', dtype=_StringDType)
        else:
            assert is_string_array(data)
            inserted_data = data
        
        # Extract the leading and trailing partitions
        if axis == 0:  # new rows
            leading, trailing = old_df[:i, :], old_df[i:, :]
        else:  # new columns
            leading, trailing = old_df[:, :i], old_df[:, i:]
        
        # Insert the new values
        self._values = cast(
            StringArray,
            np.concat(
                [leading, inserted_data, trailing],
                axis=axis,
                dtype=_StringDType
            )
        )
        
        # Insert the new sizes
        idc: list = [i+1] * N  # add 1 to skip the header size
        cell_sizes = list(self._cell_sizes)
        cell_sizes[axis] = np.insert(self._cell_sizes[axis], idc, new_sizes)
        self._cell_sizes = cast(tuple, tuple(cell_sizes))
        self._update_content_size()
        
        # Insert the new styles
        idc: list = [i] * N
        self._cell_styles = np.insert(
            self._cell_styles, idc, new_styles, axis=axis
        )
        
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
                    data=None if data is None else data.copy(),
                    sizes=None if sizes is None else copy.copy(sizes),
                    styles=None if styles is None else np.array(
                        [ [ d.copy() for d in dicts] for dicts in styles ]
                    ),
                    draw=draw,
                    trace='first'
                ),
                backward=lambda: self.delete_cells(
                    i, axis=axis, N=N, draw=draw, trace='first'
                )
            )
    
    def delete_cells(
        self,
        i: Int,
        axis: Int,
        N: Int = 1,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        assert axis in (0, 1), axis
        max_i = self.shape[axis] - 1
        assert 0 <= i <= max_i, (i, max_i)
        assert N >= 1, N
        
        # Delete the values
        old_df = self.values
        idc = np.arange(i, i+N)
        if axis == 0:
            idc_2d = (idc, slice(None))
            leading, trailing = old_df[:i, :], old_df[i+N:, :]
        else:
            idc_2d = (slice(None), idc)
            leading, trailing = old_df[:, :i], old_df[:, i+N:]
        deleted_data = self.values[idc_2d].copy()
        self._values = cast(
            StringArray,
            np.concat(
                [leading, trailing],
                axis=axis,
                dtype=_StringDType
            )
        )
        
        # Delete the sizes
        _idc = idc + 1  # add 1 to skip the header size
        scale = self._scale.get()
        all_sizes_unscaled = (
            np.round(self._cell_heights.copy() / scale).astype(_NpInt),
            np.round(self._cell_widths.copy() / scale).astype(_NpInt)
        )
        deleted_sizes_unscaled = all_sizes_unscaled[axis][_idc].copy()
        cell_sizes = list(self._cell_sizes)
        cell_sizes[axis] = np.delete(self._cell_sizes[axis], _idc)
        self._cell_sizes = cast(tuple, tuple(cell_sizes))
        self._update_content_size()
        
        # Delete the style dictionaries
        deleted_styles = np.array(
            [ [ d.copy() for d in dicts ] for dicts in self._cell_styles[idc_2d] ]
        )
        self._cell_styles = np.delete(self._cell_styles, idc, axis=axis)
        
        # Select cells
        if axis == 0:
            selection_kw = dict(r1=i, r2=i+N-1)
        else:
            selection_kw = dict(c1=i, c2=i+N-1)
        self._set_selection(**selection_kw)
        
        # Redraw
        was_reset = False
        if self._values.size == 0:  # reset the Sheet if no cells exist
            self._set_states(
                shape=self._default_cell_shape,
                cell_width=self._default_cell_sizes[1],
                cell_height=self._default_cell_sizes[0],
                min_width=self._min_sizes[1],
                min_height=self._min_sizes[0],
                get_style=self._get_style,
                max_undo=self._history._max_height,
                lock_number_of_rows=self._lock_number_of_rows,
                lock_number_of_cols=self._lock_number_of_cols,
                scale=self._scale.get()
            )
            deleted_sizes_unscaled = all_sizes_unscaled
            was_reset = True
            if draw:
                self.refresh(trace='first')
        elif draw:
            self.refresh(trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.delete_cells(
                    i, axis=axis, N=N, draw=draw, trace='first'
                ),
                backward=lambda: self._undo_delete_cells(
                    i,
                    axis=axis,
                    N=N,
                    data=deleted_data,
                    sizes=deleted_sizes_unscaled,
                    styles=deleted_styles,
                    was_reset=was_reset,
                    draw=draw,
                    trace='first'
                )
            )
    
    def _undo_delete_cells(#TODO: probably need to pass in `scale`
        self,
        i: Int,
        axis: Int,
        N: Int,
        data: StringArray,
        sizes: NDArray[NpInt] | Sequence[NDArray[NpInt]],  # unscaled sizes
        styles: NDArray[np.object_],
        was_reset: bool,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None
    ) -> None:
        assert is_string_array(data)
        assert isinstance(styles, np.ndarray), styles
        assert isinstance(sizes, (np.ndarray, Sequence)), sizes
        assert styles.shape == data.shape, (styles.shape, data.shape)
        
        if was_reset:
            n_rows, n_cols = data.shape
            assert isinstance(sizes, Sequence), type(sizes)
            assert len(sizes) == 2, sizes
            assert all( isinstance(ss, np.ndarray) for ss in sizes ), sizes
            assert sizes[0].shape == (n_rows+1,), (sizes[0].shape, data.shape)
            assert sizes[1].shape == (n_cols+1,), (sizes[1].shape, data.shape)
            
            scale = self._scale.get()
            self._values = data.copy()
            self._cell_styles = np.array(
                [ [ d.copy() for d in dicts ] for dicts in styles ]
            )
            self._cell_sizes = (
                (sizes[0] * scale).astype(_NpFloat),
                (sizes[1] * scale).astype(_NpFloat)
            )
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
                data=data,
                sizes=sizes,
                styles=styles,
                draw=draw,
                trace=trace
            )
    
    def set_values(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        data: StringArray | str = '',
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        assert isinstance(data, str) or is_string_array(data), type(data)
        
        r1, c1, r2, c2 = rcs = self._set_selection(r1, c1, r2, c2)
        rr, cc = sorted([r1, r2]), sorted([c1, c2])
        r_low, r_high, c_low, c_high = map(int, (*rr, *cc))
        
        if isinstance(data, str):
            shape = (r_high-r_low+1, c_high-c_low+1)
            _data = np.full(shape, data, dtype=_StringDType)
        else:  # `StringArray`
            _data = data
        
        old_data = self.values[r_low:r_high+1, c_low:c_high+1].copy()
        self._values[r_low:r_high+1, c_low:c_high+1] = _data
        
        if draw:
            self.draw_cells(*rcs)
            self._reselect_cells(trace=trace)
        
        if undo:
            data = data if isinstance(data, str) else data.copy()
            self._history.add(
                forward=lambda: self.set_values(
                    *rcs, data=data, draw=draw, trace='first'
                ),
                backward=lambda: self.set_values(
                    *rcs, data=old_data, draw=draw, trace='first'
                )
            )
    
    def erase_values(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        self.set_values(
            r1, c1, r2, c2, data='', draw=draw, undo=undo, trace=trace
        )
    
    def copy_values(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        to_cpliboard: bool = True
    ) -> StringArray:
        r1, c1, r2, c2 = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        data = self.values[idc]
        
        if to_cpliboard:
            string = array_to_string(data)
            self.clipboard_clear()
            self.clipboard_append(string)
        
        return data
    
    def paste_values(
        self,
        r: Int | None = None,
        c: Int | None = None,
        data: StringArray | None = None,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        if data is None:
            try:
                string = self.clipboard_get()
                data = string_to_array(string)
            except tk.TclError:
                return
        
        assert is_string_array(data)
        
        n_rows, n_cols = data.shape
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
                    data = data[:-n_rows_add, :]
                else:
                    self.insert_cells(
                        r_max, axis=0, N=n_rows_add, draw=False, undo=undo
                    )
            if (n_cols_add := n_cols - n_cols_exist):
                if self._lock_number_of_cols:
                    data = data[:, :-n_cols_add]
                else:
                    self.insert_cells(
                        c_max, axis=1, N=n_cols_add, draw=False, undo=undo
                    )
            
            # Set values
            self.set_values(
                r_start, c_start, r_end, c_end,
                data=data,
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
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        *,
        property_: str,
        values: Any = None,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        r1, c1, r2, c2 = rcs = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        styles = self._cell_styles[idc]  # an sub array of dictionaries
        
        if not isinstance(values, np.ndarray):
            values = np.full(styles.shape, values)  # broadcast
        assert values.shape == styles.shape, (values.shape, styles.shape)
        
        # Scale fonts
        if property_ == 'font':
            scale = self._scale.get()
            for font in values.flat:
                if font is not None:
                    font.configure(size=int(font._unscaled_size * scale))
        
        # Get the old values
        old_values = np.array([
            [ d.get(property_) for d in dicts ] for dicts in styles
        ])
        
        # Update the style collections with the new values
        for style, value in zip(styles.flat, values.flat):
            assert isinstance(style, dict), style
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
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        *,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        r1, c1, r2, c2 = rcs = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        styles = self._cell_styles[idc]  # an sub array of dictionaries
        
        # Get the old values
        old_styles = np.array(
            [ [ d.copy() for d in dicts ] for dicts in styles ]
        )
        
        # Clear the style collections
        for style in styles.flat:
            assert isinstance(style, dict), style
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
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        *,
        styles: NDArray[np.object_],
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None
    ) -> None:
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
            assert isinstance(style, dict), style
            style.clear()
            style.update(new_style)
        
        if draw:
            self.draw_cells(r1, c1, r2, c2)
            self._reselect_cells(trace=trace)
    
    def _get_topleft(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None
    ) -> tuple[Int, Int]:
        match (r1, r2):
            case (None, None):
                raise ValueError('`r1` and `r2` should not both be `None`.')
            case (r, None):
                pass
            case (None, r):
                pass
            case _:
                r = min(r1, r2)
        match (c1, c2):
            case (None, None):
                raise ValueError('`c1` and `c2` should not both be `None`.')
            case (c, None):
                pass
            case (None, c):
                pass
            case _:
                c = min(c1, c2)
        
        return (r, c)
    
    def set_fonts(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        fonts: tk_font.Font | NDArray[np.object_] | None = None,
        dialog: bool = False,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        if dialog:
            default_font = self._default_styles["cell"]["font"]
            r, c = self._get_topleft(r1, c1, r2, c2)
            topleft_style = self._cell_styles[r, c]
            assert isinstance(topleft_style, dict), topleft_style
            init_font = topleft_style.get('font', default_font)
            scale = self._scale.get()
            font = dialogs.Querybox.get_font(
                parent=self,
                initialvalue=init_font,
                scale=scale,
                position=self._center_window
            )
            if font is None:
                return
            
            fonts = init_font if init_font.actual() == font.actual() else font
        
        self.set_styles(
            r1, c1, r2, c2,
            property_='font',
            values=fonts,
            draw=draw,
            undo=undo,
            trace=trace
        )
    
    def set_colors(
        self,
        r1: Int | None = None,
        c1: Int | None = None,
        r2: Int | None = None,
        c2: Int | None = None,
        field: Literal['foreground', 'background'] = 'foreground',
        colors: str | None = None,
        dialog: bool = False,
        draw: bool = True,
        trace: Literal['first', 'last'] | None = None,
        undo: bool = False
    ) -> None:
        assert field in ('foreground', 'background'), field
        
        if dialog:
            default_color = self._default_styles["cell"][field]["normal"]
            r, c = self._get_topleft(r1, c1, r2, c2)
            topleft_style = self._cell_styles[r, c]
            topleft_color = topleft_style.get(field, default_color)
            color = dialogs.Querybox.get_color(
                parent=self,
                initialvalue=topleft_color,
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
    
    def _selection_insert_cells(
        self,
        axis: Int,
        mode: Literal['ahead', 'behind'] = 'ahead',
        dialog: bool = False,
        undo: bool = False
    ) -> None:
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
    
    def _selection_delete_cells(self, undo: bool = False) -> None:
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
    
    def _selection_erase_values(self, undo: bool = False) -> None:
        rcs = self._selection_rcs
        self.erase_values(*rcs, undo=undo)
    
    def _selection_copy_values(self) -> None:
        rcs = self._selection_rcs
        self.copy_values(*rcs)
    
    def _selection_paste_values(self, undo: bool = False) -> None:
        self.paste_values(undo=undo)
    
    def _selection_set_styles(
        self, property_: str, values: Any = None, undo: bool = False
    ) -> None:
        rcs = self._selection_rcs
        self.set_styles(*rcs, property_=property_, values=values, undo=undo)
    
    def _selection_reset_styles(self, undo: bool = False) -> None:
        rcs = self._selection_rcs
        self.reset_styles(*rcs, undo=undo)
    
    def _selection_set_fonts(
        self, dialog: bool = False, undo: bool = False
    ) -> None:
        rcs = self._selection_rcs
        self.set_fonts(*rcs, dialog=dialog, undo=undo)
    
    def _selection_set_colors(
        self,
        field: Literal['foreground', 'background'] = 'foreground',
        dialog: bool = False,
        undo: bool = False
    ) -> None:
        rcs = self._selection_rcs
        self.set_colors(*rcs, field=field, dialog=dialog, undo=undo)
    
    def _selection_resize_cells(
        self, axis: Int, dialog: bool = False, undo: bool = False
    ) -> None:
        r1, c1, r2, c2 = self._selection_rcs
        rcs = [(r1, r2), (c1, c2)] = [sorted([r1, r2]), sorted([c1, c2])]
        r_max, c_max = [ s - 1 for s in self.shape ]
        (_i1, _i2), max_i = rcs[axis-1], [r_max, c_max][axis-1]
        assert (_i1 == 0) and (_i2 >= max_i), (axis, (r1, c1, r2, c2), self.shape)
        
        i1, i2 = rcs[axis]
        self.resize_cells(i1, axis=axis, N=i2-i1+1, dialog=dialog, undo=undo)


class Book(tb.Frame):
    _root: Callable[[], tb.Window]
    
    @property
    def sheets(self) -> dict[str, Sheet]:
        """
        Return a dictionary containing all `Sheet`s and their names.
        """
        return { ps["name"]: ps["sheet"] for ps in self._sheets_props.values() }
    
    @property
    def sheet(self):
        """
        Return current `Sheet` or `None`.
        """
        return self._sheet
    
    @property
    def hidden_sidebar(self) -> bool:
        return self._panedwindow.sashpos(0) <= 0
    
    def __init__(
        self,
        master: tk.Misc,
        scrollbar_bootstyle: str | tuple[str, ...] | None = 'round',
        sidebar_width: Int = 180,
        lock_number_of_sheets: bool = False,
        data: dict[str, StringArray] = {
            "Sheet 0": np.full((10, 10), '', dtype=_StringDType)
        },
        sheet_kw: dict[str, Any] = {
            "shape": (10, 10),
            "cell_width": 80,
            "cell_height": 25,
            "min_width": 20,
            "min_height": 10
        },
        **kwargs
    ):
        assert isinstance(lock_number_of_sheets, bool), lock_number_of_sheets
        assert isinstance(data, dict), data
        assert isinstance(sheet_kw, dict), sheet_kw
        
        super().__init__(master, **kwargs)
        self._create_styles()
        self._lock_number_of_sheets: bool = lock_number_of_sheets
        
        
        # Build toolbar
        self._toolbar: tb.Frame = tb.Frame(self)
        self._toolbar.pack(fill='x', padx=9, pady=3)
        
        ## Sidebar button
        self._sidebar_toggle: tb.Button = tb.Button(
            self._toolbar,
            style=self._toolbar_bt_style,
            text='[Sidebar]',
            command=self._toggle_sidebar,
            takefocus=False
        )
        self._sidebar_toggle.pack(side='left')
        
        ## Separator
        sep_fm = tb.Frame(self._toolbar, width=3)
        sep_fm.pack(side='left', fill='y', padx=9, ipady=9)
        sep = tb.Separator(sep_fm, orient='vertical', takefocus=False)
        sep.place(relx=0.5, y=0, relheight=1.)
        
        ## Undo button
        self._undo_btn: tb.Button = tb.Button(
            self._toolbar,
            style=self._toolbar_bt_style,
            text='↺ Undo',
            command=lambda: getattr(self.sheet, 'undo')(),
            takefocus=False
        )
        self._undo_btn.pack(side='left')
        
        ## Redo button
        self._redo_btn: tb.Button = tb.Button(
            self._toolbar,
            style=self._toolbar_bt_style,
            text='↻ Redo',
            command=lambda: getattr(self.sheet, 'redo')(),
            takefocus=False
        )
        self._redo_btn.pack(side='left', padx=(5, 0))
        
        ## Zooming optionmenu
        self._zoom_om: OptionMenu = OptionMenu(self._toolbar, bootstyle='outline')
        self._zoom_om.pack(side='right')
        om_style = f'{id(self._zoom_om)}.{self._zoom_om["style"]}'
        self._root().style.configure(om_style, padding=(5, 0))
        self._zoom_om.configure(style=om_style)
        
        zoom_lb = tb.Label(
            self._toolbar,
            text='x',
            bootstyle='primary'
        )
        zoom_lb.pack(side='right', padx=(5, 2))
        
        # Build inputbar
        self._inputbar: tb.Frame = tb.Frame(self)
        self._inputbar.pack(fill='x', padx=9, pady=(9, 6))
        self._inputbar.grid_columnconfigure(0, minsize=130)
        self._inputbar.grid_columnconfigure(1, weight=1)
        
        ## Row and col labels
        self._label_fm: tb.Frame = tb.Frame(self._inputbar)
        self._label_fm.grid(row=0, column=0, sticky='sw')
        
        font = ('TkDefaultfont', 10)
        R_label = tb.Label(self._label_fm, text='R', font=font)  # prefix R
        R_label.pack(side='left')
        
        self._r_label: tb.Label = tb.Label(self._label_fm, font=font)
        self._r_label.pack(side='left')
        
        s_label = tb.Label(self._label_fm, text=',  ', font=font)  # seperator
        s_label.pack(side='left')
        
        C_label = tb.Label(self._label_fm, text='C', font=font)  # prefix C
        C_label.pack(side='left')
        
        self._c_label: tb.Label = tb.Label(self._label_fm, font=font)
        self._c_label.pack(side='left')
        
        ## Entry
        self._entry: tb.Entry = tb.Entry(self._inputbar, style=self._entry_style)
        self._entry.grid(row=0, column=1, sticky='nesw', padx=(12, 0))
        self._entry.bind(
            '<FocusIn>', lambda e: getattr(self.sheet, '_refresh_entry')()
        )
        self._entry.bind(
            '<KeyPress>', lambda e: getattr(self.sheet, '_on_entry_key_press')(e)
        )
        
        
        # Build sidebar and sheet frame
        border_fm = tb.Frame(self, style=self._border_fm_style, padding=1)
        border_fm.pack(fill='both', expand=True)
        
        self._panedwindow: tb.Panedwindow = tb.Panedwindow(
            border_fm, orient='horizontal'
        )
        self._panedwindow.pack(fill='both', expand=True)
        
        ## Sidebar
        self._sidebar_width: Int = sidebar_width
        self._sidebar_fm: ScrolledFrame = ScrolledFrame(
            self._panedwindow,
            vbootstyle=scrollbar_bootstyle,
            propagate_geometry=False
        )
        self._panedwindow.add(self._sidebar_fm.container)
        
        ### Sheet tab container
        def _remove_selected() -> None:
            if not (selected_items := self._sidebar.selected_items):
                return
            
            selected_keys = [
                key for key, props in self._sheets_props.items()
                if props["item"] in selected_items
            ]
            
            # Ask if the user confirm to delete the selected sheets
            if not self._delete_sheet(selected_keys.pop(), request=True):
                return
            
            # The user confirmed it. Continue to delete the selected sheets
            for key in selected_keys:
                self._delete_sheet(key, request=False)
            self._sidebar.remove_selected()
        
        state = 'disabled' if lock_number_of_sheets else 'normal'
        self._sidebar: RearrangedDnDContainer = RearrangedDnDContainer(
            self._sidebar_fm
        )
        self._sidebar.set_rearrange_commands({
            "label": 'Remove Selected...',
            "command": _remove_selected,
            "state": state
        })
        self._sidebar.set_other_commands({
            "label": 'New Sheet...',
            "command": lambda: self.insert_sheet(dialog=True),
            "state": state
        })
        self._sidebar.pack(fill='both', expand=True)
        self._sidebar.set_dnd_end_callback(self._on_dnd_end)
        
        ## Frame to contain sheets
        self._sheet_pane: tb.Frame = tb.Frame(
            self._panedwindow, padding=(1, 1, 0, 0)
        )
        self._panedwindow.add(self._sheet_pane)
        self._panedwindow.sashpos(0, 0)  # init the sash position
        
        ### Build the initial sheet(s)
        self._sheet_kw: dict[str, Any] = {
            "scrollbar_bootstyle": scrollbar_bootstyle,
            **sheet_kw
        }
        self._sheet_var: vrb.DoubleVar = vrb.DoubleVar(self)
        self._sheet_var.trace_add('write', self._select_sheet, weak=True)
        self._sheet: Sheet | None = None
        self._sheets_props: dict[float, dict[str, Any]] = {}
        if data:
            for i, (name, array) in enumerate(data.items()):
                self.insert_sheet(i, name=name, data=array, shape=None)
            self.switch_sheet(0)
        else:
            self.insert_sheet(0)
        
        # Focus on current sheet if any of the frames or canvas is clicked
        for widget in [
            self, self._toolbar, self._inputbar, R_label, self._r_label,
            s_label, C_label, self._c_label, self._sidebar_fm,
            self._sidebar_fm.canvas, self._sidebar
        ]:
            widget.configure(takefocus=False)
            widget.bind(MLEFTPRESS, self._focus_on_sheet)
        
        # Styles
        self._border_fm_style: str
        self._toolbar_bt_style: str
        self._switch_rb_style: str
        self._entry_style: str
        self.bind('<<ThemeChanged>>', self._create_styles)
    
    def _create_styles(self, event: tk.Event | None = None) -> None:
        style = self._root().style
        assert (theme := style.theme), theme
        light_theme = theme.type.lower() == 'light'
        assert isinstance(colors := style.colors, Colors), colors
        
        dummy_fm = tb.Frame(self)
        dummy_btn = tb.Button(
            self,
            bootstyle='link-primary'
        )
        dummy_rdbutton = tb.Radiobutton(
            self,
            bootstyle='toolbutton-primary'
        )
        dummy_entry = tb.Entry(self)
        
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
    
    def _on_dnd_end(
        self, event: tk.Event | None, initial_items: list[DnDItem]
    ) -> None:
        if initial_items != self._sidebar.dnd_items:
            self._rearrange_sheets_props()
        self._focus_on_sheet()
    
    def _center_window(self, toplevel: tk.Misc) -> None:
        center_window(to_center=toplevel, center_of=self.winfo_toplevel())
    
    def _rearrange_sheets_props(self, *_) -> None:
        sheets_props = self._sheets_props.copy()
        
        new_sheets_props = {}
        for item in self._sidebar.dnd_items:  # new order
            for key, props in sheets_props.items():
                if props["item"] == item:
                    new_sheets_props[key] = sheets_props.pop(key)
                    break
            else:
                raise RuntimeError(
                    f'DnD item {item!r} not found in `self._sheets_props` '
                    f'{sheets_props}.'
                )
        
        self._sheets_props = new_sheets_props
    
    def _focus_on_sheet(self, *_) -> None:
        assert self.sheet, self.sheet
        self.sheet._focus()
    
    def _toggle_sidebar(self) -> None:
        if self.hidden_sidebar:  # => show sidebar
            self.show_sidebar()
        else:  # => hide sidebar
            self.hide_sidebar()
    
    def show_sidebar(self) -> None:
        """
        Show the sidebar. Calling this method may not work before `self`
        get a layout manager. Calling this method right after `self.pack`,
        `self.grid`, or `self.place` is recommended.
        """
        if not self.hidden_sidebar:
            return
        
        self.update_idletasks()  # update all widgets
        self._panedwindow.sashpos(0, self._sidebar_width)  # set the sash position
        self.update_idletasks()  # update the sash position
    
    def hide_sidebar(self) -> None:
        if self.hidden_sidebar:
            return
        
        self._sidebar_width = self._panedwindow.sashpos(0)  # save width
        self.update_idletasks()  # update all widgets
        self._panedwindow.sashpos(0, 0)  # set the sash position
        self.update_idletasks()  # update the sash position
    
    def _get_key(self, index_or_name: Int | str) -> float:
        if isinstance(index_or_name, str):  # name input
            for key, props in self._sheets_props.items():
                if props["name"] == index_or_name:
                    return key
            raise ValueError(
                "Can't find the sheet with the name: {index_or_name}."
            )
        
        # Index input
        return list(self._sheets_props)[index_or_name]
    
    def _refresh_undo_redo_buttons(self) -> None:
        assert self.sheet, self.sheet
        
        undo_state = 'normal' if self.sheet._history.backable else 'disabled'
        redo_state = 'normal' if self.sheet._history.forwardable else 'disabled'
        self._undo_btn.configure(state=undo_state)
        self._redo_btn.configure(state=redo_state)
    
    def _refresh_zoom_optionmenu(self) -> None:
        assert self.sheet, self.sheet
        
        self._zoom_om._variable = self.sheet._scale
        self._zoom_om.configure(textvariable=self.sheet._scale)
        self._zoom_om.set_menu(None, *map(str, self.sheet._valid_scales))
    
    def _select_sheet(self, *_) -> Sheet:
        key = self._sheet_var.get()
        
        old_sheet = self._sheet
        self._sheet = new_sheet = self._sheets_props[key]["sheet"]
        
        if old_sheet:
            old_sheet.pack_forget()
            old_sheet._history.pop_callback()
        
        new_sheet.pack(fill='both', expand=True)
        new_sheet._focus()
        new_sheet._history.set_callback(self._refresh_undo_redo_buttons)
        self._refresh_undo_redo_buttons()
        self._refresh_zoom_optionmenu()
        
        self._r_label.configure(textvariable=new_sheet._focus_row)
        self._c_label.configure(textvariable=new_sheet._focus_col)
        self._entry.configure(textvariable=new_sheet._focus_value)
        
        return new_sheet
    
    def switch_sheet(self, index_or_name: Int | str) -> Sheet:
        key = self._get_key(index_or_name)
        self._sheet_var.set(key)
        self.update_idletasks()
        
        assert self.sheet, self.sheet
        
        return self.sheet
    
    def _make_unique_name(self, name: str | None = None) -> str:
        assert isinstance(name, (str, NoneType)), name
        
        # Check name
        sheets_props = self._sheets_props
        names_exist = [ props["name"] for props in sheets_props.values() ]
        if name is None:
            i, name = (0, 'Sheet 0')
            while name in names_exist:
                i += 1
                name = f'Sheet {i}'
        else:
            i, prefix = (0, name)
            while name in names_exist:
                i += 1
                name = prefix + f' ({i})'
        assert name not in names_exist, names_exist
        
        return name
    
    def insert_sheet(
        self,
        index: Int | None = None,
        name: str | None = None,
        dialog: bool = False,
        **kwargs
    ) -> Sheet | None:
        assert isinstance(index, (_Int, NoneType)), index
        assert isinstance(name, (str, NoneType)), name
        
        sheets_props = self._sheets_props
        name = self._make_unique_name(name)
        sheet_kw = self._sheet_kw.copy()
        sheet_kw.update(kwargs)
        
        if dialog:
            top = tb.Toplevel(
                transient=self,
                title='Add New Sheet',
                resizable=(False, False)
            )
            top.wm_withdraw()
            
            # Body
            body = tb.Frame(top, padding=12)
            body.pack(fill='both', expand=True)
            for c in range(3):
                body.grid_rowconfigure(c, pad=6)
            body.grid_columnconfigure(0, pad=20)
            
            ## Shape
            tb.Label(body, text='Sheet Shape (R x C)').grid(
                row=0, column=0, sticky='w')
            sb_rows = tb.Spinbox(body, from_=1, to=100_000, increment=1, width=8)
            sb_rows.grid(row=0, column=1)
            sb_rows.set(sheet_kw["shape"][0])
            tb.Label(body, text='x').grid(row=0, column=2)
            sb_cols = tb.Spinbox(body, from_=1, to=100_000, increment=1, width=8)
            sb_cols.grid(row=0, column=3)
            sb_cols.set(sheet_kw["shape"][1])
            
            ## Default size
            tb.Label(body, text='Cell Size (W x H)').grid(
                row=1, column=0, sticky='w')
            sb_w = tb.Spinbox(body, from_=1, to=200, increment=1, width=8)
            sb_w.grid(row=1, column=1)
            sb_w.set(sheet_kw["cell_width"])
            tb.Label(body, text=' x ').grid(row=1, column=2)
            sb_h = tb.Spinbox(body, from_=1, to=200, increment=1, width=8)
            sb_h.grid(row=1, column=3)
            sb_h.set(sheet_kw["cell_height"])
            
            ## Min size
            tb.Label(body, text='Minimal Cell Size (W x H)').grid(
                row=2, column=0, sticky='w')
            sb_minw = tb.Spinbox(body, from_=1, to=200, increment=1, width=8)
            sb_minw.grid(row=2, column=1)
            sb_minw.set(sheet_kw["min_width"])
            tb.Label(body, text=' x ').grid(row=2, column=2)
            sb_minh = tb.Spinbox(body, from_=1, to=200, increment=1, width=8)
            sb_minh.grid(row=2, column=3)
            sb_minh.set(sheet_kw["min_height"])
            #
            submitted = False
            def _on_submit(event=None):
                # Check if the values are valid
                for which, sb in [
                    ('number of rows', sb_rows),
                    ('number of columns', sb_cols),
                    ('cell width', sb_w),
                    ('cell height', sb_h),
                    ('minimal cell width', sb_minw),
                    ('minimal cell height', sb_minh)
                ]:
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
            tb.Separator(top, orient='horizontal').pack(fill='x')
            
            # Buttonbox
            buttonbox = tb.Frame(top, padding=(12, 18))
            buttonbox.pack(fill='both', expand=True)
            
            ## Submit/Cancel buttons
            tb.Button(
                buttonbox,
                text='Submit',
                bootstyle='primary',
                command=_on_submit
            ).pack(side='right')
            tb.Button(
                buttonbox,
                text='Cancel',
                bootstyle='secondary',
                command=top.destroy
            ).pack(side='right', padx=(0, 12))
            
            self._center_window(top)
            top.wm_deiconify()
            sb_rows.select_range(0, 'end')
            sb_rows.focus_set()
            top.grab_set()
            top.wait_window()  # don't continue until the window is destroyed
            
            if not submitted:
                return
        
        if index is None:
            index = len(sheets_props)
        
        # Generate a unique key
        while (key := time.time()) in sheets_props:
            time.sleep(1e-9)
            pass
        
        # Build a new sheet widget and sidebar button
        sheet = Sheet(self._sheet_pane, **sheet_kw)
        item = OrderlyDnDItem(self._sidebar, selectbutton=True, dragbutton=True)
        switch = tb.Radiobutton(
            item,
            style=self._switch_rb_style,
            text=name,
            value=key,
            variable=self._sheet_var,
            takefocus=False
        )
        switch.pack(fill='x', expand=True)
        switch.bind(MRIGHTPRESS, lambda e: self._post_switch_menu(e, key))
        
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
        
        return sheet
    
    def _refresh_sidebar(self) -> None:
        self._sidebar.dnd_forget(destroy=False)
        self._sidebar.dnd_put(
            [ props["item"] for props in self._sheets_props.values() ],
            sticky='nwe',
            expand=(True, False),
            padding=6,
            ipadding=2
        )
    
    def _post_switch_menu(self, event: tk.Event, key: float) -> None:
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
        index_or_name: Int | str,
        destroy: bool = True,
        request: bool = False
    ):
        key = self._get_key(index_or_name)
        return self._delete_sheet(key, destroy=destroy, request=request)
    
    def _delete_sheet(
        self, key: float, destroy: bool = True, request: bool = False
    ) -> dict[str, Any] | None:
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
    
    def rename_sheet(
        self,
        old_name: str,
        new_name: str | None = None,
    ) -> str:
        assert isinstance(old_name, str), old_name
        assert isinstance(new_name, (str, NoneType)), new_name
        
        sheets_props = self._sheets_props
        key = self._get_key(old_name)
        
        if new_name is None:
            new_name = self._make_unique_name(new_name)
        elif new_name == old_name:
            return new_name
        
        names = [ props["name"] for props in sheets_props.values() ]
        if new_name in names:
            raise DuplicateNameError(
                f'`new_name` = {new_name}. The name already exists: {names}.'
            )
        
        # Modify the sheet dict
        props = sheets_props[key]
        props["name"] = new_name
        props["switch"].configure(text=new_name)
        
        return new_name
    
    def _rename_sheet(
        self, key: float, _prompt: str = 'Enter a new name for this sheet:'
    ) -> None:
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

