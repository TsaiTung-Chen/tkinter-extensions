#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:48:41 2024

@author: tungchentsai
"""

from __future__ import annotations
from typing import Callable
import tkinter as tk

import ttkbootstrap as ttk

from tkinter_extensions.utils import unbind
# =============================================================================
# ---- Drag and Drop
# =============================================================================
class DnDItem(tk.Frame):
    def __init__(
            self,
            *args,
            dnd_bordercolor: str = 'yellow',
            dnd_borderwidth: int = 1,
            start_callback: Callable | None = None,
            end_callback: Callable | None = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self.master, DnDContainer):
            raise ValueError(
                "A `DnDItem` must have a parent of type `DnDContainer` "
                f"but got {repr(self.master)} of type `{type(self.master)}`."
            )
        
        # Make a background frame above `self` but below other children
        self._background_fm = tk.Frame(self)
        self._background_fm.place(relwidth=1.0, relheight=1.0)
        self._background_fm.bind('<Map>', self._on_background_map)
        
        # Add padding to `self` later to make it work like a border
        self._dnd_bordercolor = dnd_bordercolor
        self._dnd_borderwidth = dnd_borderwidth
        self._dnd_container: DnDContainer = self.master
        
        self._start_callback = start_callback
        self._end_callback = end_callback
    
    def set_start_callback(self, callback: Callable | None):
        """
        The callback function will be executed once DnD starts.
        This callback function will then receive two arguments. The first
        argument, namely event, has a type of `tk.Event`. The second arguemnt,
        namely source, has a type of `DnDItem`.
        """
        assert callable(callback) or callback is None, callback
        self._start_callback = callback
    
    def set_end_callback(self, callback: Callable | None):
        """
        The callback function will be executed once DnD ends.
        This callback function will then receive two arguments. The first
        argument, namely event, has a type of `tk.Event`. The second arguemnt,
        namely target, has a type of `DnDItem` or `None`.
        """
        assert callable(callback) or callback is None, callback
        self._end_callback = callback
    
    def _on_background_map(self, event: tk.Event):
        if getattr(self, '__root', None):
            self._background_fm.configure(background=self["background"])
    
    def _on_motion(self, event: tk.Event):
        self.dnd_motion(event)
        
        # Update target
        x = event.x_root - self._container_x
        y = event.y_root - self._container_y
        oids = self._dnd_container.dnd_oids
        new_target = None
        for oid in self._dnd_container.find_overlapping(x, y, x, y):
            try:
                hover = oids[oid]
            except KeyError:
                continue
            try:
                dnd_accept = hover.dnd_accept
            except AttributeError:
                continue
            
            if (new_target := dnd_accept(event, self)) is not None:
                break
        
        # Call `dnd_leave` if we are leaving the old target, or call `dnd_enter`
        # if we are entering a new target
        old_target = self._target
        if old_target is not new_target:
            if old_target is not None:  # leaving the old target
                self._target = None
                old_target.dnd_leave(event, self, new_target)
            if new_target is not None:  # entering a new target
                new_target.dnd_enter(event, self, old_target)
                self._target = new_target
    
    def _on_release(self, event: tk.Event | None):
        self._finish(event, commit=True)
    
    def _finish(self, event: tk.Event | None, commit: bool):
        target = self._target
        widget = self._initial_widget
        try:
            unbind(widget, self._motion_pattern, self._motion_id)
            unbind(widget, self._release_pattern, self._release_id)
            
            # Remove border
            self.configure(
                background=self._initial_background,
                padx=self._initial_padx,
                pady=self._initial_pady
            )
            
            self._target = self._initial_widget = self.__root = None
            
            if target is not None:
                if commit:
                    target.dnd_commit(event, self)
                else:
                    target.dnd_leave(event, self)
        finally:
            self.dnd_end(event, target)
    
    def cancel(self, event: tk.Event | None = None):
        self._finish(event, commit=False)
    
    def dnd_start(self, event: tk.Event):
        assert isinstance(button := event.num, int), button
        
        if self._start_callback:
            self._start_callback(event, self)
        
        x, y = self.winfo_rootx(), self.winfo_rooty()
        w, h = self.winfo_width(), self.winfo_height()
        
        self.__root: tk.Tk = self._root()
        self._target: DnDItem | None = None
        self._offset_x: int = x - event.x_root
        self._offset_y: int = y - event.y_root
        self._motion_pattern = f'<B{button}-Motion>'
        self._release_pattern = f'<ButtonRelease-{button}>'
        self._initial_widget = widget = event.widget
        self._initial_background: str = self["background"]
        self._initial_padx: int = self["padx"]
        self._initial_pady: int = self["pady"]
        self._container_x: int = self.master.winfo_rootx()
        self._container_y: int = self.master.winfo_rooty()
        
        focus_widget: tk.BaseWidget | None = self.focus_get()
        tk.Wm.wm_manage(self, self)  # make self frame become a toplevel
        tk.Wm.wm_overrideredirect(self, True)
        tk.Wm.wm_attributes(self, '-topmost', True)
        tk.Wm.wm_geometry(self, f'{w}x{h}+{x}+{y}')
        if focus_widget:
            self.after_idle(focus_widget.focus_set)
        
        self.configure(
            background=self._dnd_bordercolor,
            padx=self._dnd_borderwidth,
            pady=self._dnd_borderwidth
        )
        
        self._motion_id = widget.bind(
            self._motion_pattern, self._on_motion, add=True)
        self._release_id = widget.bind(
            self._release_pattern, self._on_release, add=True)
    
    def dnd_motion(self, event: tk.Event):
        # Move the source widget
        x, y = (event.x_root + self._offset_x, event.y_root + self._offset_y)
        tk.Wm.wm_geometry(self, f'+{x}+{y}')
    
    def dnd_accept(
            self,
            event: tk.Event,
            source: DnDItem
    ) -> DnDItem | None:
        assert isinstance(source, DnDItem), source
        
        if self != source and self.master == source.master:  # are siblings
            return self
    
    def dnd_enter(
            self,
            event: tk.Event,
            source: DnDItem,
            old_target: DnDItem
    ):
        assert isinstance(source, DnDItem), source
    
    def dnd_leave(
            self,
            event: tk.Event,
            source: DnDItem,
            new_target: DnDItem
    ):
        assert isinstance(source, DnDItem), source
    
    def dnd_commit(
            self,
            event: tk.Event,
            source: DnDItem
    ):
        assert isinstance(source, DnDItem), source
    
    def dnd_end(self, event: tk.Event | None, target: DnDItem | None):
        assert isinstance(target, (DnDItem, type(None))), target
        
        container = self._dnd_container
        
        # Restore `self` to become a regular widget
        tk.Wm.wm_forget(self, self)
        
        # Remove and then put the widget onto the canvas again
        container.delete(self._oid)
        self._oid = container.create_window(0, 0, window=self, tags='dnd-item')
        container.dnd_oids.clear()
        container.dnd_oids.update({w._oid: w for w in container.dnd_widgets})
        container._put(self)
        container._resize(self)
        
        if self._end_callback:
            self._end_callback(event, target)


class OrderlyDnDItem(DnDItem):
    def dnd_enter(
            self,
            event: tk.Event,
            source: DnDItem,
            old_target: DnDItem
    ):
        super().dnd_enter(event, source, old_target)
        
        container = self._dnd_container
        widgets = container.dnd_widgets
        new_source_idx = widgets.index(self)
        new_target_idx = widgets.index(source)
        
        # Exchange the indices of target (`self`) and `source` widgets
        widgets[new_target_idx] = self
        widgets[new_source_idx] = source
        
        # Update the oid dictionary
        oids = container.dnd_oids
        oids.clear()
        oids.update({ w._oid: w for w in widgets })
        
        # Exchange the geometries
        geometries = container._dnd_geometries
        old_source_geo = geometries[new_target_idx]
        old_target_geo = geometries[new_source_idx]
        new_source_geo = geometries[new_source_idx].copy()
        new_target_geo = geometries[new_target_idx].copy()
        new_source_geo.update({
            "width": old_source_geo["width"],
            "height": old_source_geo["height"],
            "relwidth": old_source_geo["relwidth"],
            "relheight": old_source_geo["relheight"],
        })
        new_target_geo.update({
            "width": old_target_geo["width"],
            "height": old_target_geo["height"],
            "relwidth": old_target_geo["relwidth"],
            "relheight": old_target_geo["relheight"],
        })
        geometries[new_source_idx] = new_source_geo
        geometries[new_target_idx] = new_target_geo
        
        # Update the UI
        container._put(self, new_target_geo)
        container._resize(self, new_target_geo)


class DnDContainer(tk.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind('<Configure>', self._on_configure)
        self._canvas_w: int = 0
        self._canvas_h: int = 0
        self._dnd_widgets: list[DnDItem] = []
        self._dnd_oids: dict[DnDItem] = {}
    
    @property
    def dnd_widgets(self) -> list[DnDItem]:
        return self._dnd_widgets
    
    @property
    def dnd_oids(self) -> dict[int, DnDItem]:
        return self._dnd_oids
    
    def _on_configure(self, event: tk.Event | None = None):
        if event is None:
            self.update_idletasks()
            self._canvas_w: int = self.winfo_width()
            self._cnavas_h: int = self.winfo_height()
        else:
            self._canvas_w: int = event.width
            self._canvas_h: int = event.height
        
        if self._dnd_widgets:
            for widget, geo in zip(self._dnd_widgets, self._dnd_geometries):
                self._put(widget, geo)
                self._resize(widget, geo)
    
    def dnd_put(
            self,
            items: list[list[DnDItem]] | list[DnDItem],
            orient: str = 'vertical',
            sticky: str = '',
            expand: tuple[bool, bool] | list[bool] | bool = False,
            padding: tuple[int, int] | list[int] | int = 0,
            ipadding: tuple[int, int] | list[int] | int = 0
    ):
        """
        Use this function to put `items` onto the canvas container.
        
        `items` must be a list of `DnDItem`s or a list of lists of 
        `DnDItem`s. If `items` is a list of lists of `DnDItem`s (a 2-D array of 
        `DnDItem`s), the result layout will be in the same order as in the 
        `items`. Otherwise, one can specify the orientation by `orient`, e.g., 
        'vertical' (default) or 'horizontal'.
        
        Other arguments work like the arguments for the grid layout manager.
        """
        assert isinstance(items, list), items
        assert isinstance(expand, (tuple, list, bool, int)), expand
        assert isinstance(padding, (tuple, list, int)), padding
        assert isinstance(ipadding, (tuple, list, int)), ipadding
        assert orient in ('horizontal', 'vertical'), orient
        
        if not isinstance(items[0], list):
            if orient == 'horizontal':
                items = [items]
            else:  # vertical
                items = [ [w] for w in items ]
        R, C = len(items), len(items[0])
        
        pairs = list()
        for arg in [expand, padding, ipadding]:
            if isinstance(arg, (tuple, list)):
                assert len(arg) == 2, arg
                arg = tuple(arg)
            else:
                arg = (arg, arg)
            pairs.append(arg)
        expand, (padx, pady), (ipadx, ipady) = pairs
        assert isinstance(sticky, str), (type(sticky), sticky)
        assert set(sticky).issubset('nesw'), sticky
        assert all( isinstance(p, int) for p in (padx, pady) ), (padx, pady)
        assert all( p >= 0 for p in (padx, pady) ), (padx, pady)
        assert all( isinstance(p, int) for p in (ipadx, ipady) ), (ipadx, ipady)
        assert all( p >= 0 for p in (ipadx, ipady) ), (ipadx, ipady)
        
        # Calculate canvas size and place items onto the canvas
        widgets_flat = [ w for row in items for w in row ]
        assert all([ isinstance(w, DnDItem) for w in widgets_flat]), items
        oids = [
            self.create_window(0, 0, anchor='nw', window=widget, tags='dnd-item')
            for widget in widgets_flat
        ]
        
        self.update_idletasks()
        widths, heights = list(), list()
        for oid, widget in zip(oids, widgets_flat):
            widget._oid: int = oid
            widths.append(widget.winfo_reqwidth())
            heights.append(widget.winfo_reqheight())
        cell_w, cell_h = max(widths), max(heights)
        canvas_w: int = C*cell_w + 2*padx + (C - 1)*ipadx
        canvas_h: int = R*cell_h + 2*pady + (R - 1)*ipady
        self._dnd_geometry_params = {
            "shape": (R, C),
            "expand": expand,
            "padding": (padx, pady),
            "ipadding": (ipadx, ipady),
            "sticky": sticky,
            "natural_canvas_size": (canvas_w, canvas_h)
        }
        
        # Calculate geometry and update items' position and size
        geometries = []
        for i, widget in enumerate(widgets_flat):
            geometries.append(
                self._calculate_geometry(i, widths[i], heights[i])
            )
            
            # Setup widget's dnd functions
            self.rebind_dnd_start(widget)
        
        self._dnd_geometries: list[dict] = geometries
        self._dnd_widgets: list[DnDItem] = widgets_flat
        self._dnd_oids: dict[int, DnDItem] = dict(zip(oids, widgets_flat))
        self.configure(width=canvas_w, height=canvas_h)
        self._on_configure()
    
    def dnd_forget(self, destroy: bool = True):
        if not self._dnd_widgets:
            return
        
        self.delete('dnd-item')
        
        if destroy:
            for widget in self._dnd_widgets:
                widget.destroy()
        
        self._dnd_geometries.clear()
        self._dnd_widgets.clear()
        self._dnd_oids.clear()
        self._dnd_geometry_params.clear()
    
    def _put(self, widget: DnDItem, geo: dict | None = None):
        geo = geo or self._dnd_geometries[self._dnd_widgets.index(widget)]
        x = max(int(geo["relx"] * self._canvas_w + geo["x"]), 0)
        y = max(int(geo["rely"] * self._canvas_h + geo["y"]), 0)
        
        self.itemconfigure(widget._oid, anchor=geo["anchor"])
        self.coords(widget._oid, x, y)
    
    def _resize(self, widget: DnDItem, geo: dict | None = None):
        geo = geo or self._dnd_geometries[self._dnd_widgets.index(widget)]
        w = max(int(geo["relwidth"] * self._canvas_w + geo["width"]), 0)
        h = max(int(geo["relheight"] * self._canvas_h + geo["height"]), 0)
        
        self.itemconfigure(widget._oid, width=w, height=h)
    
    def _calculate_geometry(self, idx, natural_width, natural_height) -> dict:
        """
        This function mimics the grid layout manager by calculating the 
        position and size of each widget in the cell.
        """
        def _calc(
                i, I, size, natural_L, stick, exp, pad, ipad
        ) -> dict:
            # Cell size (size_c) = L/I + (-2*pad - (I-1)*ipad)/I
            r = 1. / I  # relative part
            f = (-2*pad - (I - 1)*ipad)/I  # fixed part
            nat_cell_size = natural_L*r + f
            
            if exp:  # expand => use variable position and size
                if stick == (True, True):  # fill
                    return {
                        "relpos": (2*i + 1)*r / 2,
                        "pos": ((2*i + 1)*f + 2*i*ipad) / 2 + pad,
                        "relsize": r,
                        "size": f
                    }
                elif stick == (True, False):
                    return {
                        "relpos": i*r,
                        "pos": i*f + i*ipad + pad,
                        "relsize": 0.,
                        "size": size
                    }
                elif stick == (False, True):
                    return {
                        "relpos": (i + 1)*r,
                        "pos": (i + 1)*f + i*ipad + pad,
                        "relsize": 0.,
                        "size": size
                    }
                else:  # stick == (False, False)
                    return {
                        "relpos": (2*i + 1)*r / 2,
                        "pos": ((2*i + 1)*f + 2*i*ipad) / 2 + pad,
                        "relsize": 0.,
                        "size": size
                    }
            else:  # don't expand => use fixed position and size
                if stick == (True, True):  # fill
                    return {
                        "relpos": 0.,
                        "pos": ((2*i + 1)*nat_cell_size + 2*i*ipad) / 2 + pad,
                        "relsize": 0.,
                        "size": natural_L*r + f
                    }
                elif stick == (True, False):
                    return {
                        "relpos": 0.,
                        "pos": i*nat_cell_size + i*ipad + pad,
                        "relsize": 0.,
                        "size": size
                    }
                elif stick == (False, True):
                    return {
                        "relpos": 0.,
                        "pos": (i + 1)*nat_cell_size + i*ipad + pad,
                        "relsize": 0.,
                        "size": size
                    }
                else:  # stick == (False, False)
                    return {
                        "relpos": 0.,
                        "pos": ((2*i + 1)*nat_cell_size + 2*i*ipad) / 2 + pad,
                        "relsize": 0.,
                        "size": size
                    }
        #
        
        params = self._dnd_geometry_params
        R, C = params["shape"]
        expx, expy = params["expand"]
        padx, pady = params["padding"]
        ipadx, ipady = params["ipadding"]
        sticky = params["sticky"]
        natural_w, natural_h = params["natural_canvas_size"]
        
        r, c = divmod(idx, C)
        sticky_x = ('w' in sticky, 'e' in sticky)
        sticky_y = ('n' in sticky, 's' in sticky)
        anchor_x = '' if sticky_x[0] == sticky_x[1] else \
                   'w' if sticky_x[0] else 'e'
        anchor_y = '' if sticky_y[0] == sticky_y[1] else \
                   'n' if sticky_y[0] else 's'
        
        geometry_x = _calc(
            c, C, natural_width, natural_w, sticky_x, expx, padx, ipadx
        )
        geometry_y = _calc(
            r, R, natural_height, natural_h, sticky_y, expy, pady, ipady
        )
        
        geometry = {
            "anchor": (anchor_x + anchor_y) or 'center',
            "relx": geometry_x["relpos"],
            "x": geometry_x["pos"],
            "relwidth": geometry_x["relsize"],
            "width": geometry_x["size"],
            "rely": geometry_y["relpos"],
            "y": geometry_y["pos"],
            "relheight": geometry_y["relsize"],
            "height": geometry_y["size"]
        }
        
        return geometry
    
    def rebind_dnd_start(self, moved: DnDItem):
        """
        Overwrite this method to customize the trigger widget.
        """
        self._rebind_dnd_start('<ButtonPress-1>', trigger=moved, moved=moved)
    
    def _rebind_dnd_start(
            self,
            sequence: str,
            *,
            trigger: tk.BaseWidget,
            moved: DnDItem
    ):
        """
        Bind the `moved.dnd_start` method to `trigger` widget and its
        descendants with `sequence` events.
        """
        assert isinstance(moved, DnDItem), (type(moved), moved)
        
        trigger.configure(cursor='hand2')
        if hasattr(trigger, '_dnd_start_id'):
            unbind(trigger, sequence, trigger._dnd_start_id)
        trigger._dnd_start_id = trigger.bind(sequence, moved.dnd_start, add=True)
        
        for child in trigger.winfo_children():
            self._rebind_dnd_start(sequence, trigger=child, moved=moved)
    
    def set_start_callback(self, callback: Callable | None):
        """
        The callback function will be executed once DnD starts.
        This callback function will then receive two arguments. The first
        argument, namely event, has a type of `tk.Event`. The second arguemnt,
        namely `source`, has a type of `DnDItem`.
        """
        for widget in self._dnd_widgets:
            widget.set_start_callback(callback)
    
    def set_end_callback(self, callback: Callable | None):
        """
        The callback function will be executed once DnD ends.
        This callback function will then receive two arguments. The first
        argument, namely event, has a type of `tk.Event`. The second arguemnt,
        namely `target`, has a type of `DnDItem` or `None`.
        """
        for widget in self._dnd_widgets:
            widget.set_end_callback(callback)


class TriggerDnDContainer(DnDContainer):
    def rebind_dnd_start(self, moved: DnDItem):
        bound = False
        for child in moved.winfo_children():
            if hasattr(child, 'dnd_trigger') and child.dnd_trigger:
                self._rebind_dnd_start(
                    '<ButtonPress-1>', trigger=child, moved=moved
                )
                bound = True
        
        if not bound:
            raise ValueError(
                "DnD trigger not found. Please assign a value of `True` to an "
                "child widget's attribute with name 'dnd_trigger'."
            )


# =============================================================================
# ---- Test
# =============================================================================
if __name__ == '__main__':
    import random
    
    root = ttk.Window(title='Drag and Drop', themename='cyborg')
    
    container = DnDContainer(root)
    container.pack(fill='both', expand=1)
    items = []
    for r in range(6):
        items.append([])
        for c in range(3):
            dash = '----' * random.randint(1, 5)
            item = OrderlyDnDItem(container)
            ttk.Button(
                item,
                text=f'|<{dash} ({r}, {c}) {dash}>|',
                takefocus=True,
                bootstyle='outline'
            ).pack(fill='both', expand=True)
            items[-1].append(item)
    container.dnd_put(
        items,
        sticky='nse',
        expand=True,
        padding=10,
        ipadding=6
    )
    root.place_window_center()
    
    window = ttk.Toplevel(title='Button Trigger Drag and Drop')
    window.lift()
    window.after(300, window.focus_set)
    container = TriggerDnDContainer(window)
    container.pack(fill='both', expand=True)
    items = list()
    for r in range(4):
        items.append([])
        for c in range(3):
            dash = '----' * random.randint(1, 5)
            item = OrderlyDnDItem(container)
            trigger = ttk.Button(
                item,
                text='::',
                takefocus=True,
                cursor='hand2',
                bootstyle='success-link'
            )
            trigger.pack(side='left')
            trigger.dnd_trigger = True
            ttk.Label(
                item,
                text=f'|<{dash} ({r}, {c}) {dash}>|',
                bootstyle='success'
            ).pack(side='left')
            items[-1].append(item)
    container.dnd_put(
        items,
        sticky='nsw',
        expand=(False, True),
        padding=10,
        ipadding=6
    )
    window.place_window_center()
    
    root.mainloop()

