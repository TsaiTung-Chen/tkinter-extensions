#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import time
from typing import Optional
import tkinter as tk
from tkinter.font import nametofont

import ttkbootstrap as ttk

from ..constants import MODIFIER_MASKS
from ..utils import (bind_recursively,
                     unbind_recursively,
                     redirect_layout_managers)
# =============================================================================
# ---- Views
# =============================================================================
class _GeneralView:
    def __init__(self, widget, orient, sensitivity:float=1.):
        assert orient in ('x', 'y'), orient
        self._widget = widget
        self._orient = orient
        self.sensitivity = sensitivity
        self.start = 0  # pixel location
    
    @property
    def sensitivity(self):
        return self._sensitivity
    
    @sensitivity.setter
    def sensitivity(self, value:float):
        self._sensitivity = float(value)
    
    @property
    def stop(self):  # pixel location
        complete = self._get_complete_size()
        showing = self._get_showing_size(complete=complete)
        return self.start + showing
    
    @property
    def step(self):
        style = ttk.Style.get_instance()
        font_name = style.lookup(self._widget.winfo_class(), 'font')
        font = nametofont(font_name or 'TkDefaultFont')
        linespace = font.metrics()["linespace"]
        if self._orient == 'y':
            return self._sensitivity * linespace * 2  # 2-linespace height
        return self._sensitivity * linespace * 4  # 4-linespace width
    
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
    
    def view_moveto(self, fraction:float):
        """Update the position of the inner widget within the outer frame.
        """
        # Check the start and stop locations are valid
        complete = self._get_complete_size()
        showing = self._get_showing_size(complete=complete)
        self.start = self._to_pixel(fraction, complete=complete)
        self.start, stop = self._confine_region(self.start, complete, showing)
        
        # Update widgets
        self._move_content_and_scrollbar(self.start, stop, complete)
    
    def view_scroll(self, number:int, what:str):
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
        complete = self._get_complete_size()
        showing = self._get_showing_size(complete=complete)
        self.start += pixel
        self.start, stop = self._confine_region(self.start, complete, showing)
        
        # Update widgets
        self._move_content_and_scrollbar(self.start, stop, complete)
    
    def _confine_region(self, start, complete, showing):
        stop = start + showing
        if start < 0:
            start = 0
            stop = showing
        elif stop > complete:
            stop = complete
            start = stop - showing
        return start, stop
    
    def _move_content_and_scrollbar(self, start, stop, complete):
        first, last = self._to_fraction(start, stop, complete=complete)
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
    
    def _to_fraction(self, *pixels, complete=None):
        complete = complete or self._get_complete_size()
        return tuple( pixel / complete for pixel in pixels )
    
    def _to_pixel(self, *fractions, complete=None):
        complete = complete or self._get_complete_size()
        numbers = tuple( round(fraction * complete) for fraction in fractions )
        if len(numbers) == 1:
            return numbers[0]
        return numbers
    
    def _get_showing_size(self, complete=None):
        self._widget.update_idletasks()
        if self._orient == 'x':
            showing = self._widget.cropper.winfo_width()
        else:
            showing = self._widget.cropper.winfo_height()
        return min(showing, complete or self._get_complete_size())
    
    def _get_complete_size(self):
        self._widget.update_idletasks()
        if self._orient == 'x':
            return self._widget.winfo_width()
        return self._widget.winfo_height()


class GeneralXYView_Mixin:
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
        self._xview = _GeneralView(widget=self, orient='x')
        self._yview = _GeneralView(widget=self, orient='y')
    
    def get_scrolling_sensitivity(self, x:bool=True, y:bool=True):
        assert x or y, (x, y)
        if x and y:
            return [self._xview.sensitivity, self._yview.sensitivity]
        elif x:
            return self._xview.sensitivity
        return self._yview.sensitivity
    
    def set_scrolling_sensitivity(
            self, x:Optional[float]=None, y:Optional[float]=None):
        assert not (x is None and y is None), (x, y)
        
        if x is not None:
            self._xview.sensitivity = x
        if y is not None:
            self._yview.sensitivity = y
    
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


# =============================================================================
# ---- Widgets
# =============================================================================
class AutoHiddenScrollbar(ttk.Scrollbar):  # hide if all visible
    def __init__(self,
                 master=None,
                 autohide=True,
                 autohide_ms:int=300,
                 command=None,
                 **kwargs):
        super().__init__(master, command=command, **kwargs)
        self.autohide = autohide
        self._autohide_ms = autohide_ms
        self._manager = None
        self._last_func = dict()
    
    @property
    def hidden(self):
        return self._last_func.get("name", None) == 'hide'
    
    @property
    def all_visible(self):
        return [ float(v) for v in self.get() ] == [0., 1.]
    
    def set(self, first, last):
        if self.autohide and self.hidden and (not self.all_visible):
            self.show()
        
        super().set(first, last)
        
        if self.autohide and (not self.hidden) and self.all_visible:
            self.hide(after_ms=self._autohide_ms)
    
    def show(self, after_ms:int=0, autohide=None):
        assert self._manager, self._manager
        
        if autohide is not None:
            self.autohide = autohide
        
        id_ = time.time()
        self._last_func = {"name": 'show', "id": id_}
        
        if after_ms > 0:
            self.after(after_ms, self._show, id_)
        else:
            self._show(id_)
    
    def hide(self, after_ms:int=0, autohide=None):
        if autohide is not None:
            self.autohide = autohide
        
        self._manager = manager = self._manager or self.winfo_manager()
        if not manager:
            return
        
        assert manager == 'grid', (self, manager)
        
        id_ = time.time()
        
        self._last_func = {"name": 'hide', "id": id_}
        if after_ms > 0:
            self.after(after_ms, self._hide, id_)
        else:
            self._hide(id_)
    
    def _show(self, id_):
        if self._last_func.get("id", None) == id_:
            self.grid()
    
    def _hide(self, id_):
        if self._last_func.get("id", None) == id_:
            self.grid_remove()


class _Scrolled_Mixin:
    """This class is designed to be used with multiple inheritance and must be 
    the 1st parent class. This means that it will automatically call the 2nd 
    parent class' `__init__` function
    """
    mousewheel_sensitivity = 1.
    
    def __init__(self,
                 master=None,
                 scroll_orient:Optional[str]='both',
                 autohide=True,
                 hbootstyle='round',
                 vbootstyle='round',
                 builtin_method=False,
                 **kwargs):
        valid_orients = ('vertical', 'horizontal', 'both', None)
        assert scroll_orient in valid_orients, (valid_orients, scroll_orient)
        self._scroll_orient = scroll_orient
        self._builtin_method = builtin_method
        
        # Outer frame (container)
        self._container = container = ttk.Frame(
            master=master, relief='flat', borderwidth=0)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
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
        hbar, vbar = None, None
        pad = (1, 2)
        if scroll_orient in ('horizontal', 'both'):
            self._hbar = hbar = AutoHiddenScrollbar(
                master=container,
                autohide=autohide,
                command=self.xview,
                bootstyle=hbootstyle,
                orient='horizontal',
            )
            hbar.grid(row=1, column=0, sticky='ew', pady=pad)
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
            vbar.grid(row=0, column=1, sticky='ns', padx=pad)
            self.configure(yscrollcommand=vbar.set)
        elif not builtin_method:
            self.place(relheight=1.)
        
        redirect_layout_managers(self, container, orig_prefix='content_')
        
        if not builtin_method:
            cropper.bind('<Configure>', self._on_configure, add='+')
        container.bind('<Map>', self._on_map, add='+')
        container.bind('<<MapChild>>', self._on_map_child, add='+')
        self.bind('<<MapChild>>', self._on_map_child, add='+')
        self._os = self.tk.call('tk', 'windowingsystem')
    
    @property
    def container(self): return self._container  # outer frame
    
    @property
    def cropper(self): return self._cropper  # inner frame
    
    @property
    def hbar(self): return self._hbar
    
    @property
    def vbar(self): return self._vbar
    
    def show_scrollbars(self, after_ms:int=0):
        if self.hbar:
            self.hbar.show(after_ms)
        if self.vbar:
            self.vbar.show(after_ms)
    
    def hide_scrollbars(self, after_ms:int=0):
        if self.hbar:
            self.hbar.hide(after_ms)
        if self.vbar:
            self.vbar.hide(after_ms)
    
    def rebind_mousewheel(self):
        self.unbind_mousewheel()
        self.bind_mousewheel()
    
    def bind_mousewheel(self):
        if self._os == 'x11':  # Linux
            seqs = ['<ButtonPress-4>', '<ButtonPress-5>']
        else:
            seqs = ['<MouseWheel>']
        funcs = [self._mousewheel_scroll] * len(seqs)
        bind_recursively(self, seqs, funcs, add='+', key='scroll')
    
    def unbind_mousewheel(self):
        unbind_recursively(self, key='scroll')
    
    def _on_configure(self, event=None):
        self._refresh()
    
    def _on_map(self, event=None):
        self.rebind_mousewheel()
        
        if not self._builtin_method:
            if self._scroll_orient in ('vertical', None):
                self.cropper.configure(width=self.winfo_reqwidth())
            if self._scroll_orient in ('horizontal', None):
                self.cropper.configure(height=self.winfo_reqheight())
    
    def _on_map_child(self, event=None):
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
        number = -round(delta * self.mousewheel_sensitivity)
        
        if ((event.state & MODIFIER_MASKS["Shift"]) == MODIFIER_MASKS["Shift"]):
            if self._builtin_method:
                number *= 20
            self.xview_scroll(number, 'units')
        else:
            self.yview_scroll(number, 'units')


# =============================================================================
# ---- Scrolled Widgets
# =============================================================================
def ScrolledWidget(master=None,
                   widget:tk.BaseWidget=ttk.Frame,
                   scroll_orient='both',
                   autohide=True,
                   hbootstyle='round',
                   vbootstyle='round',
                   **kwargs):
    """Structure:
    <. Container (outer frame) >
        <.1 Cropper (inner frame) >
            <.1.1 wrapped widget >
        <.2 horizontal scrollbar >
        <.3 vertical scrollbar >
    """
    assert issubclass(widget, tk.BaseWidget), widget
    class _ScrolledWidget(_Scrolled_Mixin, GeneralXYView_Mixin, widget): pass
    return _ScrolledWidget(master=master,
                           scroll_orient=scroll_orient,
                           autohide=autohide,
                           hbootstyle=hbootstyle,
                           vbootstyle=vbootstyle,
                           builtin_method=False,
                           **kwargs)


class ScrolledText(_Scrolled_Mixin, ttk.Text):
    def __init__(self, *args, readonly=False, **kwargs):
        super().__init__(*args, builtin_method=True, **kwargs)
        if readonly:
            self.bind('<KeyPress>', self._prevent_modification, add='+')
        self.bind('<Control-a>', self._select_all, add='+')
        self.bind('<Control-A>', self._select_all, add='+')
    
    def _prevent_modification(self, event):
        control_mask = MODIFIER_MASKS["Control"]
        if (event.keysym.lower() in ('c', 'a')) and (
                (event.state & control_mask) == control_mask):
            return
        return 'break'
    
    def _select_all(self, event=None):
        self.event_generate('<<SelectAll>>')
        return 'break'


class ScrolledTreeview(_Scrolled_Mixin, ttk.Treeview):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, builtin_method=True, **kwargs)


def ScrolledFrame(master=None,
                  scroll_orient='both',
                  autohide=True,
                  hbootstyle='round',
                  vbootstyle='round',
                  **kwargs):
    return ScrolledWidget(master=master,
                          widget=ttk.Frame,
                          scroll_orient=scroll_orient,
                          autohide=autohide,
                          hbootstyle=hbootstyle,
                          vbootstyle=vbootstyle,
                          **kwargs)


def ScrolledLabelframe(master=None,
                       scroll_orient='both',
                       autohide=True,
                       hbootstyle='round',
                       vbootstyle='round',
                       **kwargs):
    return ScrolledWidget(master=master,
                          widget=ttk.Labelframe,
                          scroll_orient=scroll_orient,
                          autohide=autohide,
                          hbootstyle=hbootstyle,
                          vbootstyle=vbootstyle,
                          **kwargs)


# =============================================================================
# %% Main
# =============================================================================
if __name__ == '__main__':
    from ..utils import quit_if_all_closed
    
    
    root = ttk.Window(themename='cyborg')
    root.withdraw()
    
    win1 = ttk.Toplevel(title='ScrolledText')
    win1.lift()
    st = ScrolledText(win1, scroll_orient='both', autohide=True, wrap='none')
    st.insert('end', ttk.tk.__doc__)
    st.pack(fill='both', expand=1)
    
    win2 = ttk.Toplevel(title='ScrolledFrame')
    win2.lift()
    sf = ScrolledFrame(win2, autohide=False, scroll_orient='vertical')
    sf.pack(fill='both', expand=1)
    for i in range(30):
        text = str(i)+'_'.join(str(i) for i in range(500))
        ttk.Button(sf, text=text).pack(fill='both', expand=1)
    
    win3 = ttk.Toplevel(title='ScrolledCanvas', width=1500, height=1000)
    win3.lift()
    sc = ScrolledWidget(win3, widget=ttk.Canvas, vbootstyle='round-light')
    sc.pack(fill='both', expand=1)
    sc.create_polygon((10, 100), (600, 300), (900, 600), (300, 600), (300, 600),
                      outline='red', stipple='gray25', smooth=1)
    x1, y1, x2, y2 = sc.bbox('all')
    sc.configure(bg='gray', width=x2, height=y2)
    
    for win in [win1, win2, win3]:
        win.place_window_center()
        win.protocol('WM_DELETE_WINDOW', quit_if_all_closed(win))
    root.mainloop()

