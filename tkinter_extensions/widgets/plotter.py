"""
Created on Mon May 22 22:35:24 2023
@author: tungchentsai

DEPRECATED!

Replace the tk widgets in NavigationToolbar2Tk with ttk widgets.

This only works in matplotlib version greater than or equal to 3.8.4 and lower
than 3.9.0. This module will be removed since the upstream matplotlib widgets
(`NavitationToolbar2Tk` and `ToolTip`) vary across different matplotlib versions
and we are not planning to support every version of the widgets with different
code.
"""

import warnings

warnings.warn(
    f'Module `{__name__}` will be removed since the upstream matplotlib '
    'widgets (`NavitationToolbar2Tk` and `ToolTip`) vary across different '
    'matplotlib versions and we are not planning to support every version of '
    'the widgets with different code.',
    DeprecationWarning,
    stacklevel=2
)

import tkinter as tk
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from contextlib import contextmanager

from PIL import Image
import ttkbootstrap as tb
from matplotlib import cbook
import matplotlib.pyplot as plt
from matplotlib.backend_bases import _Mode, NavigationToolbar2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk, ToolTip

from tkinter_extensions import variables as vrb
from tkinter_extensions._constants import DRAWSTARTED, DRAWENDED
from tkinter_extensions.utils import create_image_pair, quit_if_all_closed, defer
from tkinter_extensions.widgets._others import UndockedFrame
from tkinter_extensions.widgets._matplotlib_config import RC
# =============================================================================
# MARK: Functions
# =============================================================================
@contextmanager
def rc_context(rc: dict | None = None, fname=None):
    backend_prev = plt.rcParams["backend"]
    try:
        with plt.rc_context(rc, fname):
            yield plt.rcParams.copy()
    finally:
        plt.rcParams["backend"] = backend_prev


def use_rc_style(rc: dict | None = None, fname=None):
    def _decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with rc_context(rc, fname):
                return func(*args, **kwargs)
        #> end of _wrapper()
        return _wrapper
    #> end of _decorator()
    
    return _decorator


def refresh_figcolors(fig, rc: dict | None = None):
    rc = rc or plt.rcParams
    
    if suptitle := fig._suptitle:
        suptitle.set_color(rc["text.color"])
    fig.set(facecolor=rc["figure.facecolor"], edgecolor=rc["figure.edgecolor"])
    
    for ax in fig.get_axes():
        ax.title.set_color(rc["text.color"])
        ax.set_facecolor(rc["axes.facecolor"])
        
        for spine in ax.spines.values():
            spine.set_color(rc["axes.edgecolor"])
        
        ax.xaxis.get_label().set_color(rc["axes.labelcolor"])
        ax.yaxis.get_label().set_color(rc["axes.labelcolor"])
        
        ax.grid(color=rc["grid.color"], which='both')
        
        if legend := ax.get_legend():
            legend.get_frame().set(
                facecolor=rc["legend.facecolor"],
                edgecolor=rc["legend.edgecolor"]
            )
            for text in legend.get_texts():
                text.set_color(rc["legend.labelcolor"])
        
        ax.tick_params(axis='x', colors=rc["xtick.color"])
        ax.tick_params(axis='y', colors=rc["ytick.color"])


def show_figs(*figs, reverse: bool = False):  # convenience function
    indices = list(range(1, len(figs)+1))
    if reverse:
        indices, figs = reversed(indices), reversed(figs)
    
    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(1)
    
    for i, fig in zip(indices, figs):
        window = tk.Toplevel(root)
        window.title(f'Figure {i}')
        window.protocol('WM_DELETE_WINDOW', quit_if_all_closed(window))
        window.lift()
        
        frame = tk.Frame(window)
        frame.pack(fill='both', expand=True)
        
        canvas = PlotterFigureCanvasTkAgg(fig, master=frame)
        canvas.draw_idle()
        canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        
        toolbar = NavigationToolbar2Ttk(canvas, frame, pack_toolbar=False)
        toolbar.pack(side='top', anchor='w', fill='x')
    
    root.mainloop()
    
    return root


def autoscale(ax: plt.Axes,
              x: bool = True,
              y: bool = True,
              force: bool = False,
              visible_only: bool = True):
    assert x or y, (x, y)
    assert isinstance(ax, plt.Axes), ax
    
    ax.relim(visible_only=visible_only)
    
    if force:  # temporarily enable the autoscale
        if x:
            x_enabled = ax.get_autoscalex_on()
            ax.set_autoscalex_on(True)
        if y:
            y_enabled = ax.get_autoscaley_on()
            ax.set_autoscaley_on(True)
    try:
        ax.autoscale_view(scalex=x, scaley=y)
    finally:
        if force:  # restore the autoscale settings
            if x:
                ax.set_autoscalex_on(x_enabled)
            if y:
                ax.set_autoscaley_on(y_enabled)


# =============================================================================
# MARK: Classes
# =============================================================================
class PlotterFigureCanvasTkAgg(FigureCanvasTkAgg):
    def __init__(self, *args, **kwargs):
        self.resize = defer(400)(self.resize)
        super().__init__(*args, **kwargs)
    
    def draw(self):
        self.get_tk_widget().event_generate(DRAWSTARTED)
        super().draw()
        self.get_tk_widget().event_generate(DRAWENDED)


class ToolTipTtk(ToolTip):
    @staticmethod
    def createToolTip(widget, text):
        toolTip = ToolTipTtk(widget)
        def enter(event):
            toolTip.showtip(text)
        def leave(event):
            toolTip.hidetip()
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
    
    def showtip(self, text):
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox('insert')
        x = x + self.widget.winfo_rootx() + self.widget.winfo_width()
        y = y + self.widget.winfo_rooty()
        self.tipwindow = tw = tb.Toplevel(self.widget)
        try:
            # For Mac OS
            tw.tk.call('::tk::unsupported::MacWindowStyle',
                       'style', tw._w,
                       'help', 'noActivates')
        except tk.TclError:
            pass
        label = tb.Label(tw, text=self.text, justify='left')
        label.pack(padx=1, pady=1)
        foreground = tb.Style().lookup(label["style"], 'foreground')
        tw.configure(background=foreground)  # border color
        
        # Override redirect at the last part and iconify-deiconify to prevent
        # flashing on macOS
        tw.iconify()
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        tw.deiconify()


class NavigationToolbar2Ttk(NavigationToolbar2Tk):
    def __init__(self, canvas, window=None, *, pack_toolbar: bool = True):
        if window is None:
            window = canvas.get_tk_widget().master
        tk.Frame.__init__(
            self,
            master=window,
            borderwidth=2,
            width=int(canvas.figure.bbox.width),
            height=50,
        )
        
        self._label_font = tk.font.Font(root=window, size=10)
        self._buttons = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text == 'Subplots':  # Skip `self.configure_subplots`
                continue
            
            if text is None:
                # Add a spacer; return value is unused.
                self._Spacer()
            else:
                self._buttons[text] = button = self._Button(
                    text,
                    str(cbook._get_data_path(f'images/{image_file}.png')),
                    toggle=callback in ['zoom', 'pan'],
                    command=getattr(self, callback),
                )
                if tooltip_text is not None:
                    ToolTipTtk.createToolTip(button, tooltip_text)
        
        style = tb.Style()
        style.configure('NavigationToolbar.TLabel', font=self._label_font)
        
        label = tb.Label(master=self,
                          text='\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}',
                          style='NavigationToolbar.TLabel')
        label.pack(side='right')
        
        self.message = vrb.StringVar(master=self)
        self._message_label = tb.Label(master=self,
                                        textvariable=self.message,
                                        style='NavigationToolbar.TLabel')
        self._message_label.pack(side='right')
        
        NavigationToolbar2.__init__(self, canvas)
        if pack_toolbar:
            self.pack(side='bottom', fill='x')
        
        # Explicitly preset the black outline to prevent invicible rubberband 
        # on macOS
        canvas._tkcanvas._create_rectangle \
            = create_rectangle \
            = canvas._tkcanvas.create_rectangle
        canvas._tkcanvas.create_rectangle = (
            lambda *args, outline='black', **kwargs:
                create_rectangle(*args, outline=outline, **kwargs)
        )
    
    def _Button(self, text, image_file, toggle, command):
        if not toggle:
            b = tb.Button(
                master=self,
                text=text,
                command=command,
                takefocus=False,
                bootstyle='outline'
            )
        else:
            var = vrb.IntVar(master=self)
            b = tb.Checkbutton(
                master=self,
                text=text,
                command=command,
                variable=var,
                takefocus=False,
                bootstyle='outline-toolbutton'
            )
            b.var = var
        b._image_file = image_file
        b.pack(side='left', padx=[0, 3])
        
        style = tb.Style()
        style_name = f'{text}.NavigationToolbar.{b["style"]}'
        style.configure(style_name, font=self._label_font)
        b.configure(style=style_name)
        
        if image_file is not None:
            self._set_image_for_button(b)
            b.bind(
                '<<ThemeChanged>>', lambda e: self._set_image_for_button(b))
        
        return b
    
    def _set_image_for_button(self, button):
        if button._image_file is None:
            return
        
        path_regular = cbook._get_data_path('images', button._image_file)
        path_large = path_regular.with_name(
            path_regular.name.replace('.png', '_large.png'))
        size = button.winfo_pixels('18p')
        path = path_large if (size > 24 and path_large.exists()) else path_regular
        
        with Image.open(path) as im:
            im = im.resize((size, size))
            (button._ntimage, button._ntimage_alt) = (img_default, img_pressed) =\
                create_image_pair(im, widget=button, photoimage=True, master=self)
        
        # Set dynamic image icon for the button
        image_mapping = [img_default]
        style = tb.Style()
        parent_style = button["style"].rsplit('NavigationToolbar.', 1)[-1]
        for fg_settings in style.map(parent_style, 'foreground'):
            statespec = fg_settings[:-1]
            image_mapping.append(statespec)
            image_mapping.append(img_pressed)
        button.configure(image=image_mapping)
    
    def _update_buttons_checked(self):
        for text, mode in [('Zoom', _Mode.ZOOM), ('Pan', _Mode.PAN)]:
            if text in self._buttons:
                if self.mode == mode:
                    self._buttons[text].var.set(1)
                else:
                    self._buttons[text].var.set(0)
    
    def pan(self, *args):  # on tk Button pressed
        """
        Toggle the pan/zoom tool.

        Pan with left button, zoom with right.
        """
        is_navigating_before = self.mode != _Mode.NONE
        super().pan(*args)
        is_navigating_after = self.mode != _Mode.NONE
        
        if not is_navigating_before and is_navigating_after:  # start navigating
            self._autoscales = self._get_autoscales_on()  # save the settings
            self._set_autoscales_on(False)
        elif is_navigating_before and not is_navigating_after:  # stop navigating
            self._set_autoscales_on(self._autoscales)  # restore the settings
    
    def zoom(self, *args):  # on tk Button pressed
        is_navigating_before = self.mode != _Mode.NONE
        super().zoom(*args)
        is_navigating_after = self.mode != _Mode.NONE
        
        if not is_navigating_before and is_navigating_after:  # start navigating
            self._autoscales = self._get_autoscales_on()  # save the settings
            self._set_autoscales_on(False)
        elif is_navigating_before and not is_navigating_after:  # stop navigating
            self._set_autoscales_on(self._autoscales)  # restore the settings
    
    def home(self, *args):
        """We overwrite the original `home` method. Instead of restoring the 
        original view (the first view in the stack), we autoscale the figure
        """
        super().home(*args)
        
        if self.mode != _Mode.NONE:  # is navigating
            # Temporarily restore the original autoscaling settings to perform 
            # the autoscaling
            autoscales = self._get_autoscales_on()
            try:
                self._set_autoscales_on(self._autoscales)
                # Autoscale the figure
                for ax in self.canvas.figure.get_axes():
                    autoscale(ax)
            finally:
                self._set_autoscales_on(autoscales)
        else:  # not navigating
            # Autoscale the figure
            for ax in self.canvas.figure.get_axes():
                autoscale(ax)
        self.canvas.draw_idle()
    
    def _update_view(self):
        """
        Update the viewlim and position from the view and position stack for
        each Axes.
        """
        autoscales = self._get_autoscales_on()  # save the settings
        try:
            super()._update_view()
        finally:
            self._set_autoscales_on(autoscales)  # restore the settings
    
    def _get_autoscales_on(self) -> dict:
        """Save the states of autoscales for each axes when pan or zoom starts
        """
        axes = [ ax for ax in self.canvas.figure.get_axes()
                 if ax.can_pan() or ax.can_zoom() ]
        return {
            ax: { axis_name: getattr(ax, f"get_autoscale{axis_name}_on")()
                  for axis_name in ax._axis_names }
            for ax in axes
        }
    
    def _set_autoscales_on(self, autoscales: dict | bool):
        """Restore the states of autoscales for each axes. This prevents 
        the autoscaling from always being off after setting axis' limits 
        while navigating
        """
        axes = [ ax for ax in self.canvas.figure.get_axes()
                 if ax.can_pan() or ax.can_zoom() ]
        for ax in axes:
            for axis_name in ax._axis_names:
                new_value = autoscales[ax][axis_name] if isinstance(
                    autoscales, dict) else autoscales
                getattr(ax, f"set_autoscale{axis_name}_on")(new_value)
    
    def save_figure(self, *args, **kw):
        super().save_figure(*args, **kw)  # this may cause empty canvas
        self.canvas.draw_idle()  # redraw the canvas
        top = self.winfo_toplevel()
        top.focus_set()
        top.lift()


class Plotter(UndockedFrame):
    @property
    def delete_on_destroy(self) -> list:
        return self._delete_on_destroy
    
    @property
    def toolbar(self):
        return self._toolbar
    
    @property
    def figurecanvas(self):
        return self._figurecanvas
    
    @property
    def canvas(self):
        return self._figurecanvas.get_tk_widget()
    
    @property
    def figure(self):
        return self._figurecanvas.figure
    
    @property
    def axes(self) -> list:
        return self._figurecanvas.figure.axes
    
    def __init__(self, master, figure, **kw):
        kw.setdefault("window_title", 'Figure')
        super().__init__(master, place_button=False, **kw)
        self._delete_on_destroy = list()
        self._rc = self._fetch_rc()
        self._refresh_on_map: bool = False
        
        # Use grid layout manager to ensure the toolbar and the canvas are
        # always shown even if the container frame is in very small size
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self._figurecanvas = PlotterFigureCanvasTkAgg(figure, master=self)
        self._delete_on_destroy.append(self._figurecanvas)
        canvas = self._figurecanvas.get_tk_widget()
        canvas.grid(row=0, column=0, sticky='nesw')
        canvas.bind(DRAWSTARTED, self._on_draw_started, add=True)
        canvas.bind(DRAWENDED, self._on_draw_ended, add=True)
        
        self._toolbar = NavigationToolbar2Ttk(
            self._figurecanvas, self, pack_toolbar=False)
        self._toolbar.grid(row=1, column=0, sticky='we')
        self._toolbar.update()
        self._delete_on_destroy.append(self._toolbar)
        
        self.place_undock_button(anchor='ne', relx=1., rely=0., x=0, y=-6)
        self.bind('<Map>', self._on_map, add=True)
        self.bind('<<ThemeChanged>>', self._on_theme_changed, add=True)
        self.bind('<Destroy>', self._on_destroy, add=True)
    
    def __del__(self):
        """Delete bindings to free memory
        """
        self._on_destroy()
    
    def _on_destroy(self, event=None):
        """Delete bindings to free memory
        """
        if not self._delete_on_destroy:
            return
        
        for name, attr in list(vars(self).items()):
            if attr in self._delete_on_destroy:
                delattr(self, name)
        self._delete_on_destroy.clear()
    
    def _on_map(self, event=None):
        if self._refresh_on_map:
            self.refresh()
    
    def _on_draw_started(self, event=None):
        pass
    
    def _on_draw_ended(self, event=None):
        pass
    
    def _on_theme_changed(self, event=None):
        old_rc = self._rc
        new_rc = self._rc = self._fetch_rc()
        
        if new_rc == old_rc:
            return False
        
        refresh_figcolors(self.figure)
        self.refresh()
        return True
    
    def _fetch_rc(self, copy: bool = True) -> dict:
        rc = plt.rcParams
        if copy:
            return deepcopy(rc)
        return rc
    
    def draw_idle(self):
        self.figurecanvas.draw_idle()
    
    def autoscale(self, *args, **kwargs):
        autoscale(*args, **kwargs)
    
    def refresh(self):
        if not self.winfo_ismapped():  # defer refreshing until next map
            self._refresh_on_map = True
            return
        
        self._refresh_on_map = False
        
        for ax in self.axes:
            self.autoscale(ax)
        self.draw_idle()

