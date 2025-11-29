"""
Created on Mon May 22 22:35:24 2023
@author: tungchentsai
"""
import time
from types import NoneType
from typing import Protocol, Any, Literal, overload
from collections.abc import Callable
import tkinter as tk
from tkinter.font import nametofont

import ttkbootstrap as tb

from tkinter_extensions._constants import (
    OS, MODIFIER_MASKS, COMMAND, MSCROLL, ScreenUnits, _ScreenUnits
)
from tkinter_extensions.utils import (
    Int,
    DropObject,
    mixin_base,
    to_pixels,
    defer,
    bind_recursively,
    unbind_recursively,
    redirect_layout_managers
)
# =============================================================================
# MARK: Scrollbar Widget
# =============================================================================
class AutoHiddenScrollbar(tb.Scrollbar):  # hide if all visible
    _root: Callable[[], tb.Window]
    
    def __init__(
        self,
        master: tk.Misc,
        autohide: bool = True,
        autohide_ms: int = 300,
        command: Callable | str = '',
        bootstyle: str | tuple[str, ...] | None = 'round',
        **kwargs
    ):
        super().__init__(
            master=master,
            command=command,
            bootstyle=bootstyle,
            **kwargs
        )
        
        self._autohide: bool = bool(autohide)
        self._autohide_ms: int = int(autohide_ms)
        self._manager: str | None = None
        self._last_func: dict[str, Any] = {"name": 'show', "id": None}
    
    @property
    def autohide(self):
        return self._autohide
    
    @autohide.setter
    def autohide(self, enabled: bool | None) -> None:
        if enabled is not None:
            self._autohide = bool(enabled)
    
    @property
    def hidden(self) -> bool:
        return self._last_func["name"] == 'hide'
    
    @property
    def all_visible(self) -> bool:
        return tuple( float(v) for v in self.get() ) == (0., 1.)
    
    def set(self, first: float | str, last: float | str) -> None:
        if self._last_func["id"] is None:  # init
            self.show()
        
        if self.autohide and self.hidden and (not self.all_visible):
            self.show()
        
        super().set(first, last)
        
        if self.autohide and (not self.hidden) and self.all_visible:
            self.hide(after_ms=self._autohide_ms)
    
    def show(self, after_ms: Int = -1, autohide: bool | None = None) -> None:
        if self._manager is None:
            self._manager = self.winfo_manager()
        assert self._manager == 'grid', self._manager
        
        self.autohide = autohide
        id_ = time.monotonic()
        self._last_func.update(name='show', id=id_)
        
        if after_ms < 0:
            self._show(id_)
        else:
            self.after(int(after_ms), self._show, id_)
    
    def hide(self, after_ms: Int = -1, autohide: bool | None = None) -> None:
        if self._manager is None:
            self._manager = self.winfo_manager()
        assert self._manager == 'grid', self._manager
        
        self.autohide = autohide
        id_ = time.monotonic()
        self._last_func.update(name='hide', id=id_)
        
        if after_ms < 0:
            self._hide(id_)
        else:
            self.after(int(after_ms), self._hide, id_)
    
    def _show(self, id_) -> None:
        if self._last_func == {"name": 'show', "id": id_}:
            self.grid()
    
    def _hide(self, id_) -> None:
        if self._last_func == {"name": 'hide', "id": id_}:
            self.grid_remove()


# =============================================================================
# MARK: Patches
# =============================================================================
class _GeneralView:
    _root: Callable[[], tb.Window]
    
    def __init__(
        self,
        widget: '_GeneralXYView',
        orient: Literal['horizontal', 'vertical'],
        sensitivity: float = 1.
    ):
        assert isinstance(widget, _Scrolled), widget  # types' intersection
        assert orient in ('horizontal', 'vertical'), orient
        
        self._widget: _GeneralXYView = widget
        self._orient: Literal['horizontal', 'vertical'] = orient
        self._sensitivity: float = sensitivity
        self.start: float = 0.  # pixel location
    
    @property
    def sensitivity(self):
        return self._sensitivity
    
    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._sensitivity = float(value)
    
    @property
    def stop(self) -> float:  # pixel location
        content_size = self._get_content_size()
        showing = self._get_showing_size(content_size=content_size)
        return self.start + showing
    
    @property
    def step(self) -> float:
        style = self._root().style
        font_name = style.lookup(self._widget.winfo_class(), 'font')
        font = nametofont(font_name or 'TkDefaultFont')
        linespace = font.metrics()["linespace"]
        
        if self._orient == 'vertical':
            return self._sensitivity * linespace * 1.  # 2-linespace height
        return self._sensitivity * linespace * 2.  # 4-linespace width
    
    @overload
    def view(self) -> tuple[float, float]: ...
    @overload
    def view(self, action, /, *values) -> None: ...
    def view(self, *args):
        """
        Update the vertical position of the inner widget within the outer
        frame.
        """
        if not args:
            return self._to_fractions(self.start, self.stop)
        
        action, args = args[0], args[1:]
        if action == 'moveto':
            return self.view_moveto(float(args[0]))
        elif action == 'scroll':
            return self.view_scroll(int(args[0]), args[1])
        raise ValueError(
            f"The first argument must be 'moveto' or 'scroll' but got: {action}"
        )
    
    def view_moveto(self, fraction: float) -> None:
        """
        Update the position of the inner widget within the outer frame.
        """
        # Check the start and stop locations are valid
        content_size = self._get_content_size()
        showing = self._get_showing_size(content_size=content_size)
        self.start = self._to_pixels(fraction, content_size=content_size)
        self.start, stop = self._confine_region(
            self.start, content_size, showing
        )
        
        # Update widgets
        self._move_content_and_scrollbar(self.start, stop, content_size)
    
    @overload
    def view_scroll(
        self, number: int, what: Literal['units', 'pages']
    ) -> None: ...
    @overload
    def view_scroll(
        self, number: ScreenUnits, what: Literal['pixels']
    ) -> None: ...
    def view_scroll(self, number, what) -> None:
        """
        Update the position of the inner widget within the outer frame.
        Note: If `what == 'units'` and `number == 1`, the content will be 
        scrolled down 1 line (y orientation). If `what == 'pages'`, the content 
        will be scrolled down five times the amount of the aforementioned lines 
        (y orientation).
        """
        assert isinstance(self, tk.Widget), self
        if what == 'pages':
            assert isinstance(number, int), (number, what)
            pixels = number * self.step * 5
        else:
            assert isinstance(number, _ScreenUnits), (number, what)
            pixels = to_pixels(self, number) * self.step
        
        # Check the start and stop locations are valid
        content_size = self._get_content_size()
        showing = self._get_showing_size(content_size=content_size)
        self.start += pixels
        self.start, stop = self._confine_region(
            self.start, content_size, showing
        )
        
        # Update widgets
        self._move_content_and_scrollbar(self.start, stop, content_size)
    
    def _confine_region(
        self, start: float, content_size: int, showing: int
    ) -> tuple[float, float]:
        stop = start + showing
        
        if start < 0:
            start = 0
            stop = showing
        elif stop > content_size:
            stop = content_size
            start = stop - showing
        
        return float(start), float(stop)
    
    def _move_content_and_scrollbar(
        self, start: float, stop: float, content_size: int
    ) -> None:
        assert isinstance(self._widget, _Scrolled), self._widget  # types' intersection
        
        first, last = self._to_fractions(start, stop, content_size=content_size)
        
        if self._orient == 'horizontal':  # X orientation
            self._widget.content_place(x=-start)
            if self._widget._set_xscrollbar:
                self._widget._set_xscrollbar(first, last)
            self._widget.update_idletasks()
            return
        
        # Y orientation
        self._widget.content_place(y=-start)
        if self._widget._set_yscrollbar:
            self._widget._set_yscrollbar(first, last)
        self._widget.update_idletasks()
    
    def _to_fractions(
        self, *pixels: float, content_size: int | None = None
    ) -> tuple[float, ...]:
        if content_size is None:
            content_size = self._get_content_size()
        return tuple( pixel / content_size for pixel in pixels )
    
    def _to_pixels(
        self, fraction: float, content_size: int | None = None
    ) -> int:
        if content_size is None:
            content_size = self._get_content_size()
        
        return round(fraction * content_size)
    
    def _get_showing_size(self, content_size: int | None = None) -> int:
        assert isinstance(self._widget, _Scrolled), self._widget  # types' intersection
        assert (cropper := self._widget.cropper) is not None, cropper
        
        self._widget.update_idletasks()
        
        if self._orient == 'horizontal':
            showing = cropper.winfo_width()
        else:
            showing = cropper.winfo_height()
        
        if content_size is None:
            content_size = self._get_content_size()
        
        return min(showing, content_size)
    
    def _get_content_size(self) -> int:
        self._widget.update_idletasks()
        
        if self._orient == 'horizontal':
            return self._widget.winfo_width()
        return self._widget.winfo_height()


class _GeneralXYView(mixin_base(tk.Widget), metaclass=DropObject):
    """
    This class is a workaround to mimic the `tkinter.XView` and `tkinter.YView`
    behaviors.
    
    This class is designed to be used with multiple inheritance and must be 
    the 2nd parent class. This means that the 1st parent class must call this 
    class' `__init__` function. After that this class will automatically call 
    the 3rd parent class' `__init__` function
    """
    
    def __init__(
        self,
        *args,
        xscrollcommand: Callable[[float | str, float | str], None] | None = None,
        yscrollcommand: Callable[[float | str, float | str], None] | None = None,
        **kwargs
    ):
        # Init the 2nd parent class
        self._set_xscrollbar: Callable[[float | str, float | str], None] | None \
            = None
        self._set_yscrollbar: Callable[[float | str, float | str], None] | None \
            = None
        self._configure_scrollcommands(
            xscrollcommand=xscrollcommand, yscrollcommand=yscrollcommand
        )
        super().__init__(*args, **kwargs)
        
        # Init x and y GeneralViews
        self._xview: _GeneralView = _GeneralView(
            widget=self, orient='horizontal', sensitivity=0.5
        )
        self._yview: _GeneralView = _GeneralView(widget=self, orient='vertical')
    
    @overload
    def xview(self) -> tuple[float, float]: ...
    @overload
    def xview(self, action, /, *values) -> None: ...
    def xview(self, *args) -> tuple[float, float] | None:
        return self._xview.view(*args)
    
    def xview_moveto(self, fraction: float):
        return self._xview.view_moveto(fraction)
    
    @overload
    def xview_scroll(
        self, number: int, what: Literal['units', 'pages']
    ) -> None: ...
    @overload
    def xview_scroll(
        self, number: ScreenUnits, what: Literal['pixels']
    ) -> None: ...
    def xview_scroll(self, number, what):
        self._xview.view_scroll(number, what)
    
    @overload
    def yview(self) -> tuple[float, float]: ...
    @overload
    def yview(self, action, /, *values) -> None: ...
    def yview(self, *args) -> tuple[float, float] | None:
        return self._yview.view(*args)
    
    def yview_moveto(self, fraction: float):
        return self._yview.view_moveto(fraction)
    
    @overload
    def yview_scroll(
        self, number: int, what: Literal['units', 'pages']
    ) -> None: ...
    @overload
    def yview_scroll(
        self, number: ScreenUnits, what: Literal['pixels']
    ) -> None: ...
    def yview_scroll(self, number, what):
        self._yview.view_scroll(number, what)
    
    def configure(
        self,
        *args,
        xscrollcommand: Callable[[float | str, float | str], None] | None = None,
        yscrollcommand: Callable[[float | str, float | str], None] | None = None,
        **kwargs
    ) -> Any:
        self._configure_scrollcommands(
            xscrollcommand=xscrollcommand, yscrollcommand=yscrollcommand
        )
        return super().configure(*args, **kwargs)
    
    def _configure_scrollcommands(
        self,
        xscrollcommand: Callable[[float | str, float | str], None] | None = None,
        yscrollcommand: Callable[[float | str, float | str], None] | None = None
    ) -> None:
        if xscrollcommand:
            self._set_xscrollbar = xscrollcommand
        if yscrollcommand:
            self._set_yscrollbar = yscrollcommand
    
    def _refresh(self) -> None:
        self.xview_scroll(0, 'units')
        self.yview_scroll(0, 'units')


class _Scrolled(mixin_base(tk.Widget), metaclass=DropObject):
    """
    This class is designed to be used with multiple inheritance and must be 
    the 1st parent class. It will automatically call the 2nd parent class'
    `__init__` function.
    """
    
    _scrollbar_padding: tuple[int, int] = (0, 1)
    
    def __init__(
        self,
        master: tk.Misc,
        scroll_orient: Literal['horizontal', 'vertical', 'both'] = 'vertical',
        autohide: bool = True,
        hbootstyle: str | tuple[str, ...] | None = 'round',
        vbootstyle: str | tuple[str, ...] | None = 'round',
        scroll_sensitivities:
            float
            | tuple[float, float]
            = 1.,
        builtin_method: bool = False,
        propagate_geometry: bool = True,
        bind_mousewheel_with_add: bool = True,
        **kwargs
    ):
        valid_orients = ('horizontal', 'vertical', 'both')
        assert isinstance(self, tk.Widget), self
        assert scroll_orient in valid_orients, (valid_orients, scroll_orient)
        assert isinstance(builtin_method, bool), builtin_method
        assert isinstance(propagate_geometry, bool), propagate_geometry
        assert isinstance(bind_mousewheel_with_add, bool), bind_mousewheel_with_add
        self._builtin_method: bool = builtin_method
        self._bind_mousewheel_with_add: bool = bind_mousewheel_with_add
        if isinstance(scroll_sensitivities, tuple):
            self.set_scroll_sensitivities(*scroll_sensitivities)
        else:
            self.set_scroll_sensitivities(
                scroll_sensitivities, scroll_sensitivities
            )
        
        # Outer frame (container)
        container = tk.Frame(master=master, relief='flat', borderwidth=0)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.grid_propagate(propagate_geometry)
        self._container: tk.Frame = container
        
        self._cropper: tk.Frame | None
        if builtin_method:
            assert isinstance(self, tk.XView), self  # types' intersection
            assert isinstance(self, tk.YView), self  # types' intersection
            
            class _ScrolledConfigure(Protocol):
                def __call__(
                    self,
                    *args,
                    xscrollcommand:
                        Callable[[float | str, float | str], None]
                        | None
                        = None,
                    yscrollcommand:
                        Callable[[float | str, float | str], None]
                        | None
                        = None,
                    **kwargs
                ) -> Any: ...
            self.configure: _ScrolledConfigure  # pyright: ignore [reportIncompatibleMethodOverride]
            
            self._cropper = None
            
            # Main widget
            super().__init__(master=container, **kwargs)
            self.grid(row=0, column=0, sticky='nesw')
        else:
            assert isinstance(self, _GeneralXYView), self  # types' intersection
            
            # Inner frame (cropper)
            self._cropper = tk.Frame(
                master=container, relief='flat', borderwidth=0
            )
            self._cropper.grid(row=0, column=0, sticky='nesw')
            
            # Main widget
            super().__init__(master=self._cropper, **kwargs)
            self.place(x=0, y=0)
        
        # Scrollbars
        self._hbar: AutoHiddenScrollbar | None = None
        self._vbar: AutoHiddenScrollbar | None = None
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
        
        # Bind events
        self.bind('<Map>', self._on_map, add=True)
        self.bind('<<MapChild>>', self._on_map_child, add=True)
        self.bind('<Configure>', self._on_configure, add=True)
        self.bind('<Enter>', lambda e: self.rebind_mousewheel(), add=True)
        
        # Redirect layout manager methods
        redirect_layout_managers(self, container, orig_prefix='content_')
        self.content_pack_configure: Callable
        self.content_pack: Callable
        self.content_pack_forget: Callable
        self.content_pack_info: Callable
        self.content_pack_propagate: Callable
        self.content_pack_slaves: Callable
        self.content_place_configure: Callable
        self.content_place: Callable
        self.content_place_forget: Callable
        self.content_place_info: Callable
        self.content_place_slaves: Callable
        self.content_grid_configure: Callable
        self.content_grid: Callable
        self.content_grid_forget: Callable
        self.content_grid_remove: Callable
        self.content_grid_info: Callable
        self.content_grid_propagate: Callable
        self.content_grid_slaves: Callable
        self.content_grid_bbox: Callable
        self.content_grid_location: Callable
        self.content_grid_size: Callable
        self.content_grid_anchor: Callable
    
    @property
    def container(self):
        return self._container  # outer frame
    
    @property
    def cropper(self):
        return self._cropper  # inner frame
    
    @property
    def hbar(self):
        return self._hbar
    
    @property
    def vbar(self):
        return self._vbar
    
    def set_scroll_sensitivities(
        self, horizontal: float | None = None, vertical: float | None = None
    ) -> tuple[float, float]:
        assert isinstance(horizontal, (float, NoneType)), horizontal
        assert isinstance(vertical, (float, NoneType)), vertical
        
        self._scroll_sensitivities = (
            self._scroll_sensitivities[0] if horizontal is None else horizontal,
            self._scroll_sensitivities[1] if vertical is None else vertical
        )
        
        return self._scroll_sensitivities
    
    def set_autohide_scrollbars(
        self, enable: bool | None = None
    ) -> tuple[bool, bool]:
        
        states: list = [None, None]
        if self.hbar:
            self.hbar.autohide = enable
            states[0] = self.hbar.autohide
        
        if self.vbar:
            self.vbar.autohide = enable
            states[1] = self.vbar.autohide
        
        return tuple(states)
    
    def show_scrollbars(
        self, after_ms: int = -1, autohide: bool | None = None
    ) -> None:
        if self.hbar:
            self.hbar.show(after_ms, autohide=autohide)
        if self.vbar:
            self.vbar.show(after_ms, autohide=autohide)
    
    def hide_scrollbars(
        self, after_ms: int = -1, autohide: bool | None = None
    ) -> None:
        if self.hbar:
            self.hbar.hide(after_ms, autohide=autohide)
        if self.vbar:
            self.vbar.hide(after_ms, autohide=autohide)
    
    def rebind_mousewheel(self, add: bool | None = None) -> None:
        assert isinstance(self, tk.Widget), self
        assert isinstance(add, (bool, NoneType)), add
        
        self.unbind_mousewheel()
        
        # Bind mousewheel
        funcs = [self._mousewheel_scroll] * len(MSCROLL)
        bind_recursively(
            self, MSCROLL, funcs,
            add=self._bind_mousewheel_with_add if add is None else add,
            key='scrolled-wheel',
            skip_toplevel=True
        )
    
    def unbind_mousewheel(self) -> None:
        assert isinstance(self, tk.Widget), self
        unbind_recursively(self, key='scrolled-wheel')
    
    def _on_configure(self, event: tk.Event | None = None) -> None:
        if not self._builtin_method:
            assert isinstance(self, _GeneralXYView), self
            self._refresh()
    
    def _on_map(self, event: tk.Event | None = None) -> None:
        assert isinstance(self, tk.Widget), self
        
        if not self._builtin_method:
            assert isinstance(self._cropper, tk.Widget), self
            
            if not self.hbar:
                self._cropper.configure(width=self.winfo_reqwidth())
            if not self.vbar:
                self._cropper.configure(height=self.winfo_reqheight())
    
    def _on_map_child(self, event: tk.Event) -> None:
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
        
        if self._container.winfo_ismapped():
            self._on_map(event)
    
    def _mousewheel_scroll(self, event: tk.Event) -> Literal['break']:
        """
        Callback for when the mouse wheel is scrolled.
        Modified from: `ttkbootstrap.scrolled.ScrolledFrame._on_mousewheel`
        """
        if self._builtin_method:
            assert isinstance(self, tk.XView), self  # types' intersection
            assert isinstance(self, tk.YView), self  # types' intersection
        else:
            assert isinstance(self, _GeneralXYView), self  # types' intersection
        assert isinstance(event.state, int), event
        
        if event.num == 4:  # Linux
            delta = 10.
        elif event.num == 5:  # Linux
            delta = -10.
        elif OS == "Windows":  # Windows
            delta = event.delta / 120.
        else:  # Mac
            delta = event.delta
        
        x_direction = (
            (event.state & MODIFIER_MASKS["Shift"])
            == MODIFIER_MASKS["Shift"]
        )
        sensitivity = self._scroll_sensitivities[0 if x_direction else 1]
        number = -round(delta * sensitivity)
        
        if x_direction:
            if self.hbar:
                self.xview('scroll', number, 'units')
        elif self.vbar:
            self.yview('scroll', number, 'units')
        
        return 'break'
    
    def content_size(
        self, hbar: bool = False, vbar: bool = False
    ) -> tuple[int, int]:
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
        
        return (
            content_width + vbar_width + pad_width,
            content_height + hbar_height + pad_height
        )
    
    def set_size(
        self, width: int | None = None, height: int | None = None
    ) -> None:
        if self._builtin_method:
            raise TypeError("This function does not support built-in methods.")
        
        assert self._cropper is not None, self._cropper
        
        self._cropper.configure(width=width, height=height)  # pyright: ignore [reportArgumentType] (`None` should also be acceptable)
        self._container.configure(width=width, height=height)  # pyright: ignore [reportArgumentType] (`None` should also be acceptable)


# =============================================================================
# MARK: Scrolled Widgets with `GeneralXYView`
# =============================================================================
def create_scrolledwidget(widget: type[tk.Widget] = tb.Frame) -> type[_Scrolled]:
    """
    Structure:
    <. Container (outer frame) >
        <.1 Cropper (inner frame) >
            <.1.1 wrapped widget >
        <.2 horizontal scrollbar >
        <.3 vertical scrollbar >
    """
    assert issubclass(widget, tk.Widget), widget
    
    class _ScrolledWidget(_Scrolled, _GeneralXYView, widget):
        _root: Callable[[], tb.Window]
    
    return _ScrolledWidget


def ScrolledWidget(
    master: tk.Misc,
    widget: type[tk.Widget] = tb.Frame,
    **kwargs
) -> tk.Widget:
    """
    A convenience function working like a class instance init function.
    """
    return create_scrolledwidget(widget)(
        master=master, builtin_method=False, **kwargs
    )


# =============================================================================
# MARK: Scrolled Widgets with the Builtin Method
# =============================================================================
class ScrolledTreeview(_Scrolled, tb.Treeview):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, builtin_method=True, **kwargs)


class ScrolledText(_Scrolled, tb.Text):
    def __init__(
        self,
        *args,
        readonly: bool = False,
        bind_select_all: bool = True,
        **kwargs
    ):
        super().__init__(*args, builtin_method=True, **kwargs)
        
        if readonly:
            self.bind('<KeyPress>', self._prevent_modification, add=True)
        
        if bind_select_all:
            self.bind(f'<{COMMAND}-A>', self._select_all, add=True)
            self.bind(f'<{COMMAND}-a>', self._select_all, add=True)
    
    def _prevent_modification(self, event) -> Literal['break'] | None:
        command_mask = MODIFIER_MASKS[COMMAND]
        if (
            (event.keysym.lower() in ('c', 'a'))
            and ((event.state & command_mask) == command_mask)
        ):
            return
        return 'break'
    
    def _select_all(self, event: tk.Event | None = None) -> Literal['break']:
        self.event_generate('<<SelectAll>>')
        return 'break'


class ScrolledCanvas(_Scrolled, tk.Canvas):
    def __init__(self, *args, fill: bool = False, **kwargs):
        assert isinstance(fill, bool), (type(fill), fill)
        
        self._on_map_child = defer(200)(self._on_map_child)
        super().__init__(*args, builtin_method=True, **kwargs)
        self._fill: bool = fill
    
    @overload
    def xview(self) -> tuple[float, float]: ...
    @overload
    def xview(self, *args): ...
    def xview(self, *args) -> tuple[float, float] | None:
        if not args:
            return super().xview()
        elif super().xview() != (0.0, 1.0):  # prevent from over scrolling
            return super().xview(*args)
    
    @overload
    def yview(self) -> tuple[float, float]: ...
    @overload
    def yview(self, *args): ...
    def yview(self, *args) -> tuple[float, float] | None:
        if not args:
            return super().yview()
        elif super().yview() != (0.0, 1.0):  # prevent from over scrolling
            super().yview(*args)
    
    def _update_scrollregion(self) -> tuple[int, int, int, int]:
        self.update_idletasks()
        if bbox := self.bbox('all'):
            _, _, x2, y2 = bbox
        else:
            x2 = y2 = 0
        x2, y2 = max(x2, 0), max(y2, 0)
        scrollregion = (0, 0, x2, y2)
        self.configure(scrollregion=scrollregion)
        
        return scrollregion
    
    def _on_configure(self, event: tk.Event | None = None) -> None:
        # Fill the space with content in the non-scrollable direction
        if self._fill:
            self.update_idletasks()
            if bbox := self.bbox('all'):
                _, _, x2, y2 = bbox
            else:
                x2 = y2 = 0.
            xscale = 1.0 if self.hbar or x2 == 0. else self.winfo_width() / x2
            yscale = 1.0 if self.vbar or y2 == 0. else self.winfo_height() / y2
            
            ## Scale objects
            self.scale('all', 0, 0, xscale, yscale)
        
        self._update_scrollregion()
    
    def _on_map(self, event: tk.Event | None = None) -> None:
        self.refresh()
    
    def refresh(self) -> None:
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
            _, _, x2, y2 = bbox
            self.configure(width=x2, height=y2)
    
    def content_size(
        self, hbar: bool = False, vbar: bool = False
    ) -> tuple[int, int]:
        self.update_idletasks()
        if bbox := self.bbox('all'):
            _, _, content_width, content_height = bbox
        else:
            content_width = content_height = 0
        
        pad_width = pad_height = sum(self._scrollbar_padding)
        if vbar and self.vbar:
            vbar_width = self.vbar.winfo_reqwidth()
        else:
            vbar_width = pad_width = 0
        
        if hbar and self.hbar:
            hbar_height = self.hbar.winfo_reqheight()
        else:
            hbar_height = pad_height = 0
        
        return (
            content_width + vbar_width + pad_width,
            content_height + hbar_height + pad_height
        )
    
    def set_size(
        self, width: int | None = None, height: int | None = None
    ) -> None:
        self.configure(width=width, height=height)
        self._on_configure()


class _CanvasBasedScrolled(mixin_base(tk.Widget), metaclass=DropObject):
    def __init__(
        self,
        master: tk.Misc,
        scroll_orient: Literal['horizontal', 'vertical', 'both'] = 'vertical',
        autohide: bool = True,
        hbootstyle: str | tuple[str, ...] | None ='round',
        vbootstyle: str | tuple[str, ...] | None = 'round',
        scroll_sensitivities:
            float
            | tuple[float, float]
            = 1.,
        propagate_geometry: bool = True,
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
        
        # Layout: [master [ScrolledCanvas [self]]]
        self._canvas: ScrolledCanvas = ScrolledCanvas(
            master, fill=True, **canvas_kw
        )
        super().__init__(self._canvas, **kwargs)
        self._id: int = self._canvas.create_window(0, 0, anchor='nw', window=self)
        self.bind('<<MapChild>>', self._canvas._on_map_child, add=True)
        
        redirect_layout_managers(self, self._canvas, orig_prefix='content_')
        self.content_pack_configure: Callable
        self.content_pack: Callable
        self.content_pack_forget: Callable
        self.content_pack_info: Callable
        self.content_pack_propagate: Callable
        self.content_pack_slaves: Callable
        self.content_place_configure: Callable
        self.content_place: Callable
        self.content_place_forget: Callable
        self.content_place_info: Callable
        self.content_place_slaves: Callable
        self.content_grid_configure: Callable
        self.content_grid: Callable
        self.content_grid_forget: Callable
        self.content_grid_remove: Callable
        self.content_grid_info: Callable
        self.content_grid_propagate: Callable
        self.content_grid_slaves: Callable
        self.content_grid_bbox: Callable
        self.content_grid_location: Callable
        self.content_grid_size: Callable
        self.content_grid_anchor: Callable
    
    @property
    def container(self):
        return self._canvas._container  # outer container
    
    @property
    def canvas(self):
        return self._canvas  # scrollable canvas
    
    @property
    def hbar(self):
        return self._canvas.hbar
    
    @property
    def vbar(self):
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


class ScrolledFrame(_CanvasBasedScrolled, tb.Frame): pass
class ScrolledTkFrame(_CanvasBasedScrolled, tk.Frame): pass
class ScrolledLabelframe(_CanvasBasedScrolled, tb.Labelframe): pass

