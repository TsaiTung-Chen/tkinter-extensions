#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import tkinter as tk
from typing import Union, Optional
from copy import deepcopy
from functools import wraps
from contextlib import contextmanager

from PIL import Image
import ttkbootstrap as ttk
import matplotlib.pyplot as plt
from matplotlib.backend_bases import _Mode
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk, ToolTip

from ..utils import create_image_pair, unbind, quit_if_all_closed, defer
from .undocked import UndockedFrame
# =============================================================================
# ---- Functions
# =============================================================================
@contextmanager
def rc_context(rc:Optional[dict]=None, fname=None):
    backend_prev = plt.rcParams["backend"]
    try:
        with plt.rc_context(rc, fname):
            yield plt.rcParams.copy()
    finally:
        plt.rcParams["backend"] = backend_prev


def use_rc_style(rc:Optional[dict]=None, fname=None):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            with rc_context(rc, fname):
                return func(*args, **kwargs)
        return _wrapper
    
    return _decorator


def refresh_figcolors(fig, rc:Optional[dict]=None):
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
        ax.tick_params(axis='x', colors=rc["xtick.color"])
        ax.tick_params(axis='y', colors=rc["ytick.color"])


def show_figs(*figs, reverse=False):  # convenience function
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
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw_idle()
        canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
        
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.pack(side='top', anchor='w', fill='x')
    
    root.mainloop()
    
    return root


def autoscale(ax:plt.Axes, x=True, y=True, force=False, visible_only=True):
    assert x or y, (x, y)
    assert isinstance(ax, plt.Axes), ax
    
    ax.relim(visible_only=visible_only)
    
    if force:
        if x:
            orig_x = ax.get_autoscalex_on()
            ax.set_autoscalex_on(True)
        if y:
            orig_y = ax.get_autoscaley_on()
            ax.set_autoscaley_on(True)
    try:
        ax.autoscale_view(scalex=x, scaley=y)
    finally:
        if force:
            if x:
                ax.set_autoscalex_on(orig_x)
            if y:
                ax.set_autoscaley_on(orig_y)


# =============================================================================
# ---- Classes
# =============================================================================
class PlotterFigureCanvasTkAgg(FigureCanvasTkAgg):
    def __init__(self, *args, **kwargs):
        self.resize = defer(400)(self.resize)
        super().__init__(*args, **kwargs)
    
    def draw(self):
        self.get_tk_widget().event_generate('<<DrawStarted>>')
        super().draw()
        self.get_tk_widget().event_generate('<<DrawEnded>>')


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
        self.tipwindow = tw = ttk.Toplevel(self.widget)
        try:
            # For Mac OS
            tw.tk.call("::tk::unsupported::MacWindowStyle",
                       "style", tw._w,
                       "help", "noActivates")
        except tk.TclError:
            pass
        label = ttk.Label(tw, text=self.text, justify='left')
        label.pack(padx=1, pady=1)
        foreground = ttk.Style().lookup(label["style"], 'foreground')
        tw.configure(background=foreground)  # border color
        
        # Override redirect at the last part and iconify-deiconify to prevent
        # flashing on macOS
        tw.iconify()
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        tw.deiconify()


class NavigationToolbarTtk(NavigationToolbar2Tk):
    def __init__(self, canvas, window=None, *, pack_toolbar=True):
        from matplotlib import cbook
        from matplotlib.backend_bases import NavigationToolbar2
        
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
        
        style = ttk.Style()
        style.configure('NavigationToolbar.TLabel', font=self._label_font)
        
        label = ttk.Label(master=self,
                          text='\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}',
                          style='NavigationToolbar.TLabel')
        label.pack(side='right')
        
        self.message = tk.StringVar(master=self)
        self._message_label = ttk.Label(master=self,
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
            b = ttk.Button(
                master=self,
                text=text,
                command=command,
                bootstyle='outline'
            )
        else:
            var = tk.IntVar(master=self)
            b = ttk.Checkbutton(
                master=self,
                text=text,
                command=command,
                variable=var,
                bootstyle='outline-toolbutton'
            )
            b.var = var
        b._image_file = image_file
        b.pack(side='left')
        
        style = ttk.Style()
        style_name = f'{text}.NavigationToolbar.{b["style"]}'
        style.configure(style_name, font=self._label_font)
        b.configure(style=style_name)
        
        if image_file is not None:
            self._set_image_for_button(b)
        
        return b
    
    def _set_image_for_button(self, button):
        from matplotlib import cbook
        
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
        style = ttk.Style()
        parent_style = button["style"].rsplit('NavigationToolbar.', 1)[-1]
        for fg_settings in style.map(parent_style, 'foreground'):
            statespec = fg_settings[:-1]
            image_mapping.append(statespec)
            image_mapping.append(img_pressed)
        button.configure(image=image_mapping)
        
        button.bind(
            '<<ThemeChanged>>', lambda e: self._set_image_for_button(button))
    
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
            self._autoscales_map = self._get_autoscales_on()  # save the settings
            self._set_autoscales_on(False)
        elif is_navigating_before and not is_navigating_after:  # stop navigating
            self._set_autoscales_on(self._autoscales_map)  # restore the settings
    
    def zoom(self, *args):  # on tk Button pressed
        is_navigating_before = self.mode != _Mode.NONE
        super().zoom(*args)
        is_navigating_after = self.mode != _Mode.NONE
        
        if not is_navigating_before and is_navigating_after:  # start navigating
            self._autoscales_map = self._get_autoscales_on()  # save the settings
            self._set_autoscales_on(False)
        elif is_navigating_before and not is_navigating_after:  # stop navigating
            self._set_autoscales_on(self._autoscales_map)  # restore the settings
    
    def home(self, *args):
        """We overwrite the original `home` method. Instead of restoring the 
        original view (the first view in the stack), we autoscale the figure
        """
        super().home(*args)
        
        if self.mode != _Mode.NONE:  # is navigating
            # Temporarily restore the original autoscaling settings to perform 
            # the autoscaling
            navigating_autoscales_map = self._get_autoscales_on()
            try:
                self._set_autoscales_on(self._autoscales_map)
                # Autoscale the figure
                for ax in self.canvas.figure.get_axes():
                    autoscale(ax)
            finally:
                self._set_autoscales_on(navigating_autoscales_map)
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
        autoscales_map = self._get_autoscales_on()  # save the settings
        try:
            super()._update_view()
        finally:
            self._set_autoscales_on(autoscales_map)  # restore the settings
    
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
    
    def _set_autoscales_on(self, autoscales_map:Union[dict, bool]):
        """Restore the states of autoscales for each axes. This prevents 
        the autoscaling from always being turned off after setting axis' limits 
        while navigating
        """
        axes = [ ax for ax in self.canvas.figure.get_axes()
                 if ax.can_pan() or ax.can_zoom() ]
        for ax in axes:
            for axis_name in ax._axis_names:
                new_value = autoscales_map[ax][axis_name] if isinstance(
                    autoscales_map, dict) else autoscales_map
                getattr(ax, f"set_autoscale{axis_name}_on")(new_value)
    
    def save_figure(self, *args, **kw):
        super().save_figure(*args, **kw)  # this may cause empty canvas
        self.canvas.draw_idle()  # redraw the canvas
        top = self.winfo_toplevel()
        top.focus_set()
        top.lift()


class BasePlotter(UndockedFrame):
    @property
    def figure(self):
        return self._figcanvas.figure
    
    @property
    def figcanvas(self):
        return self._figcanvas
    
    @property
    def canvas(self):
        return self._figcanvas.get_tk_widget()
    
    @property
    def toolbar(self):
        return self._toolbar
    
    @property
    def delete_on_destroy(self) -> list:
        return self._delete_on_destroy
    
    def __init__(self, master, figure, dnd_trigger=False, **kw):
        kw["place_button"] = False
        kw.setdefault("window_title", 'Plotting Pad')
        super().__init__(master, **kw)
        self._delete_on_destroy = list()
        self._rc = self._fetch_rc()
        
        if dnd_trigger:
            dnd_trigger = ttk.Button(self,
                                     text=' '.join(':'*3000),
                                     takefocus=False,
                                     bootstyle='link-primary')
            dnd_trigger.pack(side='top', fill='x', expand=1)
            dnd_trigger._dnd_trigger = True
        
        self._figcanvas = PlotterFigureCanvasTkAgg(figure, master=self)
        self._delete_on_destroy.append(self._figcanvas)
        canvas = self._figcanvas.get_tk_widget()
        canvas.pack(side='top', fill='both', expand=1)
        
        self._toolbar = NavigationToolbarTtk(
            self._figcanvas, self, pack_toolbar=False)
        self._toolbar.pack(side='bottom', anchor='w', fill='x')
        self._toolbar.update()
        self._delete_on_destroy.append(self._toolbar)
        
        self.place_undock_button(anchor='ne', relx=1., rely=0., x=0, y=-6)
        self._init_id = canvas.bind('<Map>', self._on_first_mapped, add='+')
        canvas.bind('<<DrawStarted>>', self._on_draw_started, add='+')
        canvas.bind('<<DrawEnded>>', self._on_draw_ended, add='+')
        canvas.bind('<<ThemeChanged>>', self._on_theme_changed, add='+')
        self.bind('<Destroy>', self._on_destroy, add='+')
    
    def __del__(self):
        """Delete bindings to free memory
        """
        self._on_destroy()
    
    def _on_destroy(self, event=None):
        """Delete bindings to free memory
        """
        if (event is None or event.widget == self) and self.delete_on_destroy:
            for name, attr in list(vars(self).items()):
                if attr in self.delete_on_destroy:
                    delattr(self, name)
            self.delete_on_destroy.clear()
    
    def _on_first_mapped(self, event=None):
        unbind(self.canvas, '<Map>', self._init_id)
        self._on_theme_changed()
        self.draw_idle()

    def _on_draw_started(self, event=None):
        pass
    
    def _on_draw_ended(self, event=None):
        pass
    
    def _on_theme_changed(self, event=None):
        old_rc = self._rc
        new_rc = self._rc = self._fetch_rc()
        if new_rc != old_rc:
            refresh_figcolors(self.figure)
            self.refresh()
            return True
        return False
    
    def _fetch_rc(self, copy:bool=True) -> dict:
        rc = plt.rcParams
        if copy:
            return deepcopy(rc)
        return rc
    
    def draw_idle(self):
        self.figcanvas.draw_idle()
    
    def autoscale(self, *args, **kwargs):
        autoscale(*args, **kwargs)


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    import numpy as np
    
    root = ttk.Window(
        title='Embedding in Ttk', themename='cyborg', size=[500, 500])
    
    t = np.arange(0, 3, .01)
    x = 2 * np.sin(2 * np.pi * 1 * t)
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    ax = fig.subplots()
    line, = ax.plot(t, x, label='f = 1 Hz')
    ax.set_xlabel("time [s]")
    ax.set_ylabel("f(t)")
    ax.legend(loc='upper right')
    
    plotter = BasePlotter(root, fig)
    plotter.pack(side='top', fill='both', expand=1)
    
    def _update_frequency(new_val):
        f = float(new_val)
        
        # update data
        y = 2 * np.sin(2 * np.pi * f * t)
        line.set_ydata(y)
        line.set_label(f'f = {f: .2f} Hz')
        ax.legend(loc='upper right')
        plotter.figcanvas.draw_idle()  # update canvas
    
    slider = ttk.Scale(root,
                       from_=1,
                       to=5,
                       orient='horizontal',
                       command=_update_frequency)
    slider.pack(side='bottom', pady=10)
    
    root.mainloop()

