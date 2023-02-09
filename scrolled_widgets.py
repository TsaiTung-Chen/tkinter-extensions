# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:14:42 2023

@author: Jeff_Tsai
"""

from tkinter.font import nametofont
from tkinter import Pack, Grid, Place, Widget

import ttkbootstrap as ttk

DEFAULT_FONT = 'TkDefaultFont'
MODIFIER_MASKS = {
    "Shift": int('0x0001', base=16),
    "Caps Lock": int('0x0002', base=16),
    "Control": int('0x0004', base=16),
    "Left-hand Alt": int('0x0008', 16),
    "Num Lock": int('0x0010', base=16),
    "Right-hand Alt": int('0x0080', base=16),
    "Mouse button 1": int('0x0100', base=16),
    "Mouse button 2": int('0x0200', base=16),
    "Mouse button 3": int('0x0400', base=16)
}
# =============================================================================
# ---- Functions
# =============================================================================
def quit_if_all_closed(window):
    def _wrapped(event=None):
        root = window._root()
        if len(root.children) > 1:
            window.destroy()
        else:
            root.quit()
    return _wrapped


def _bind_recursively(widget, seqs, funcs, add=''):
    add = '+' if add else ''
    if isinstance(seqs, str):
        seqs = [seqs]
    if not isinstance(funcs, (list, tuple)):
        funcs = [funcs]
    
    widget._bound = getattr(widget, "_bound", dict())
    for seq, func in zip(seqs, funcs):
        assert seq.startswith('<') and seq.endswith('>'), seq
        if seq in widget._bound:
            continue
        widget._bound[seq] = widget.bind(seq, func, add)  # func id
    
    # Propagate
    for child in widget.winfo_children():
        _bind_recursively(child, seqs, funcs)


def _unbind_recursively(widget, seqs=None):
    if isinstance(seqs, str):
        seqs = [seqs]
    
    if getattr(widget, "_bound", None):
        for seq, func_id in list(widget._bound.items()):
            if (seqs is not None) and (seq not in seqs):
                continue
            widget.unbind(seq, func_id)
            del widget._bound[seq]
    
    # Propagate
    for child in widget.winfo_children():
        _unbind_recursively(child)


# =============================================================================
# ---- Views
# =============================================================================
class _GeneralView:
    def __init__(self, widget, orient):
        assert orient in ('x', 'y'), orient
        self._widget = widget
        self._orient = orient
        self.start = 0  # pixel location
    
    @property
    def stop(self):  # pixel location
        complete = self._get_complete_size()
        showing = self._get_showing_size(complete=complete)
        return self.start + showing
    
    @property
    def step(self):
        style = ttk.Style.get_instance()
        font_name = style.lookup(self._widget.winfo_class(), 'font')
        font = nametofont(font_name or DEFAULT_FONT)
        linespace = font.metrics()["linespace"]
        if self._orient == 'y':
            return linespace  # 1-linespace height
        return linespace * 2  # 2-linespace width
    
    def view(self, *args):
        """Update the vertical position of the inner widget within the outer
        frame.
        """
        if not args:
            return self._to_fraction(self.start, self.stop)
        elif args[0] == 'moveto':
            return self.view_moveto(fraction=float(args[1]))
        elif args[0] == 'scroll':
            return self.view_scroll(int(args[1]), args[2])
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
        scrolled down 1 line. If `what == 'pages'`, the content will be 
        scrolled down some pixels, which is determined by the values of 
        scrollbar's `first` and `last`
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
            return
        # Y orientation
        self._widget.content_place(y=-start)
        if self._widget._set_yscrollbar:
            self._widget._set_yscrollbar(first, last)
    
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
    """This class is a workaround to mimic the behavior of `tkinter.XView` and 
    `tkinter.YView`.
    
    This class is designed to be used with multiple inheritance and must be 
    the 1st parent class. This means that it will automatically call the 2nd 
    parent class' __ini__ function
    """
    def __init__(self, *args, mousewheel_sens=2., **kwargs):
        # Check the method resolution order (mro)
        mro = type(self).mro()
        assert mro[1] is GeneralXYView_Mixin, mro
        
        # Init the 2nd parent class
        self._set_xscrollbar = None
        self._set_yscrollbar = None
        kwargs = self._configure_scrollcommands(kwargs)
        super().__init__(*args, **kwargs)
        
        # Init x and y GeneralViews
        self._xview = _GeneralView(widget=self, orient='x')
        self._yview = _GeneralView(widget=self, orient='y')
        
        self._mousewheel_sens = mousewheel_sens
        self._os = self.tk.call('tk', 'windowingsystem')
    
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
        number = -round(delta * self._mousewheel_sens)
        
        if ((event.state & MODIFIER_MASKS["Shift"]) == MODIFIER_MASKS["Shift"]):
            self.xview_scroll(number, 'units')
        else:
            self.yview_scroll(number, 'units')
    
    def _refresh(self):
        self.xview_scroll(0, 'unit')
        self.yview_scroll(0, 'unit')


# =============================================================================
# ---- Widgets
# =============================================================================
class AutoHiddenScrollbar(ttk.Scrollbar):  # hide if all visible
    def __init__(self, master=None, autohide=True, command=None, **kwargs):
        super().__init__(master, command=command, **kwargs)
        self._autohide = autohide
        self._manager = None
        self._last_func = dict()
    
    @property
    def hidden(self):
        return self._last_func.get("name", None) == 'hide'
    
    @property
    def all_visible(self):
        return [ float(v) for v in self.get() ] == [0., 1.]
    
    def set(self, first, last):
        if self.hidden and (not self.all_visible):
            self.show()
        
        super().set(first, last)
        
        if (not self.hidden) and self.all_visible:
            self.hide(after_ms=1000)
    
    def show(self, after_ms:int=0):
        assert self.hidden and self._manager, (self.hidden, self._manager)
        
        self._cancel_last_action()
        self._last_func = {
            "name": 'show',
            "id": self.after(after_ms, getattr(self, self._manager))
        }
    
    def hide(self, after_ms:int=0, _=None):
        self._manager = manager = self._manager or self.winfo_manager()
        if not manager:
            return
        
        self._cancel_last_action()
        self._last_func = {
            "name": 'hide',
            "id": self.after(after_ms, getattr(self, manager + '_remove'))
        }
    
    def _cancel_last_action(self):
        if self._last_func:
            self.after_cancel(self._last_func["id"])
            self._last_func = None


class _ContainerFrame(ttk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, relief='flat', borderwidth=0, **kwargs)
        self._os = self.tk.call('tk', 'windowingsystem')
    
    def rebind_mousewheel(self, func):
        self.unbind_mousewheel()
        self.bind_mousewheel(func)
    
    def bind_mousewheel(self, func):
        if self._os == 'x11':  # Linux
            seqs = ['<ButtonPress-4>', '<ButtonPress-5>',
                    '<Shift-ButtonPress-4>', '<Shift-ButtonPress-5>']
        else:
            seqs = ['<MouseWheel>', '<Shift-MouseWheel>']
        funcs = [func] * len(seqs)
        _bind_recursively(self, seqs[0], funcs[0], add='+')
    
    def unbind_mousewheel(self):
        _unbind_recursively(self)


class _CropperFrame(ttk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, relief='flat', borderwidth=0, **kwargs)


# =============================================================================
# ---- Scrolled Widgets
# =============================================================================
def ScrolledWidget(master=None,
                   widget:Widget=ttk.Frame,
                   orient='both',
                   autohide=True,
                   hbootstyle='round-light',
                   vbootstyle='round-light',
                   **kwargs):
    """Structure:
    <. Container (outer frame) >
        <.1 Cropper (inner frame) >
            <.1.1 wrapped widget >
        <.2 horizontal scrollbar >
        <.3 vertical scrollbar >
    """
    class _ScrolledWidget(GeneralXYView_Mixin, widget):
        @property
        def container(self): return container  # outer frame
        
        @property
        def cropper(self): return cropper  # inner frame
        
        @property
        def hbar(self): return hbar
        
        @property
        def vbar(self): return vbar
        
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
        
        def _on_configure(self, event):
            scrolledwidget._refresh()
        
        def _on_map(self, event):
            self.container.rebind_mousewheel(self._mousewheel_scroll)
        
        def _on_map_child(self, event):
            if self.container.winfo_ismapped():
                self.container.rebind_mousewheel(self._mousewheel_scroll)
    
    #
    #
    # Outer frame (container)
    container = _ContainerFrame(master=master)
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)
    
    # Inner frame (cropper)
    assert issubclass(widget, Widget), widget
    cropper = _CropperFrame(master=container)
    cropper.grid(row=0, column=0, sticky='nesw')
    hbar, vbar = None, None
    
    # Main widget
    scrolledwidget = _ScrolledWidget(master=cropper, **kwargs)
    scrolledwidget.place(x=0, y=0)
    
    # Scrollbars
    if orient in ('horizontal', 'both'):
        hbar = AutoHiddenScrollbar(
            master=container,
            autohide=autohide,
            command=scrolledwidget.xview,
            bootstyle=hbootstyle,
            orient='horizontal',
        )
        hbar.grid(row=1, column=0, sticky='esw')
        scrolledwidget.configure(xscrollcommand=hbar.set)
        container.grid_rowconfigure(1, minsize=hbar.winfo_reqheight())
    
    if orient in ('vertical', 'both'):
        vbar = AutoHiddenScrollbar(
            master=container,
            autohide=autohide,
            command=scrolledwidget.yview,
            bootstyle=vbootstyle,
            orient='vertical',
        )
        vbar.grid(row=0, column=1, sticky='nes')
        scrolledwidget.configure(yscrollcommand=vbar.set)
        container.grid_columnconfigure(1, minsize=vbar.winfo_reqwidth())
    
    cropper.bind('<Configure>', scrolledwidget._on_configure)
    container.bind('<Map>', scrolledwidget._on_map)
    scrolledwidget.bind('<<MapChild>>', scrolledwidget._on_map_child)
    
    # Redirect layout manager to the outer frame's layout manager
    layout_methods = vars(Pack).keys() | vars(Grid).keys() | vars(Place).keys()
    is_layout = lambda name: (
        (not name.startswith('_')) and
        (name not in ['configure', 'config']) and
        (getattr(_ContainerFrame, name) is getattr(_ScrolledWidget, name))
    )
    
    for name in filter(is_layout, layout_methods):
        setattr(scrolledwidget, 'content_'+name, getattr(scrolledwidget, name))
        setattr(scrolledwidget, name, getattr(container, name))
    
    return scrolledwidget


def ScrolledText(master=None,
                 orient='both',
                 autohide=True,
                 hbootstyle='round-light',
                 vbootstyle='round-light',
                 **kwargs):
    return ScrolledWidget(master=master,
                          widget=ttk.Text,
                          orient=orient,
                          autohide=autohide,
                          hbootstyle='round-light',
                          vbootstyle='round-light',
                          **kwargs)


def ScrolledFrame(master=None,
                  orient='both',
                  autohide=True,
                  hbootstyle='round-light',
                  vbootstyle='round-light',
                  **kwargs):
    return ScrolledWidget(master=master,
                          widget=ttk.Frame,
                          orient=orient,
                          autohide=autohide,
                          hbootstyle='round-light',
                          vbootstyle='round-light',
                          **kwargs)


# =============================================================================
# %% Main
# =============================================================================
if __name__ == '__main__':
    root = ttk.Window(themename='cyborg')
    root.withdraw()
    
    win1 = ttk.Toplevel(title='ScrolledText')
    win1.lift()
    st = ScrolledText(win1, orient='both', autohide=True, wrap='none')
    st.insert('end', ttk.tk.__doc__)
    st.pack(fill='both', expand=1)
    
    win2 = ttk.Toplevel(title='ScrolledFrame')
    win2.lift()
    sf = ScrolledFrame(win2, autohide=False, orient='vertical')
    sf.pack(fill='both', expand=1)
    for i in range(30):
        ttk.Button(sf, text=str(i)+'_'.join(str(i) for i in range(500))).pack()
    
    win3 = ttk.Toplevel(title='ScrolledCanvas', width=1500, height=1000)
    win3.lift()
    sc = ScrolledWidget(win3, widget=ttk.Canvas)
    sc.pack(fill='both', expand=1)
    sc.create_polygon((10, 100), (600, 300), (900, 600), (300, 600), (300, 600),
                      outline='red', stipple='gray25', smooth=1)
    x1, y1, x2, y2 = sc.bbox('all')
    sc.configure(bg='gray', width=x2, height=y2)
    
    for win in [win1, win2, win3]:
        win.place_window_center()
        win.protocol('WM_DELETE_WINDOW', quit_if_all_closed(win))
    root.mainloop()

