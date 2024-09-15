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
from tkinter_extensions import variables as vrb
# =============================================================================
# ---- Drag and Drop
# =============================================================================
class DnDItem(ttk.Frame):
    def __init__(
            self,
            master,
            *args,
            dnd_bordercolor: str = 'yellow',
            dnd_borderwidth: int = 1,
            checkbutton_bootstyle: str = 'default',
            bootstyle='default',
            **kwargs
    ):
        if not isinstance(master, DnDContainer):
            raise ValueError(
                "`master` must be of type `DnDContainer` but got "
                "{repr(master)} of type `{type(master)}`."
            )
        
        self._container = tk.Frame(master, padx=0, pady=0)
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(1, weight=1)
        
        self._background_fm = ttk.Frame(self._container, bootstyle=bootstyle)
        self._background_fm.place(relwidth=1., relheight=1.)
        
        # Make a checkbutton and show it only when the selection mode is active
        self._selected = vrb.BooleanVar(self._container, value=False)
        self._select_bt = ttk.Checkbutton(
            self._container,
            variable=self._selected,
            bootstyle=checkbutton_bootstyle
        )
        self._select_bt.grid(row=0, column=0, padx=(15, 3))
        self._select_bt.grid_remove()
        self._select_mode: bool = False
        
        # Content frame
        super().__init__(self._container, *args, **kwargs)
        self.grid(row=0, column=1, sticky='nswe')
        
        # Add padding to `self` later to make it work like a border
        self._dnd_bordercolor = dnd_bordercolor
        self._dnd_borderwidth: int = int(dnd_borderwidth)
        self._dnd_container: DnDContainer = master
        self._dnd_active: bool = False
    
    @property
    def container(self) -> tk.Frame:
        return self._container
    
    @property
    def dnd_active(self) -> bool:
        return self._dnd_active
    
    @property
    def select_mode(self) -> bool:
        return self._select_mode
    
    @property
    def selected(self) -> vrb.BooleanVar:
        return self._selected
    
    def set_select_mode(self, enable: bool):
        assert isinstance(enable, bool), enable
        
        if self._select_mode == enable:
            return
        
        if enable:  # show checkbutton
            self._select_bt.grid()
        else:  # hide checkbutton
            self._select_bt.grid_remove()
        
        self._selected.set(False)
        self._select_mode = enable
    
    def _on_motion(self, event: tk.Event):
        self.dnd_motion(event)
        
        # Update target
        x = event.x_root - self._dnd_container_x
        y = event.y_root - self._dnd_container_y
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
        try:
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
        if self._select_mode:  # disable DnD if selection mode is on
            return
        
        assert isinstance(button := event.num, int), button
        
        container = self._container
        dnd_container = self._dnd_container
        
        if dnd_container._dnd_start_callback:
            dnd_container._dnd_start_callback(event, self)
        
        x, y = container.winfo_rootx(), container.winfo_rooty()
        w, h = container.winfo_width(), container.winfo_height()
        
        self._dnd_active = True
        self._target: DnDItem | None = None
        self._offset_x: int = x - event.x_root
        self._offset_y: int = y - event.y_root
        self._motion_pattern = f'<B{button}-Motion>'
        self._release_pattern = f'<ButtonRelease-{button}>'
        self._initial_widget = widget = event.widget
        self._initial_background = container["background"]
        self._dnd_container_x: int = self._dnd_container.winfo_rootx()
        self._dnd_container_y: int = self._dnd_container.winfo_rooty()
        
        focus_widget: tk.BaseWidget | None = self.focus_get()
        tk.Wm.wm_manage(container, container)  # make container become a toplevel
        tk.Wm.wm_overrideredirect(container, True)
        tk.Wm.wm_attributes(container, '-topmost', True)
        tk.Wm.wm_geometry(container, f'{w}x{h}+{x}+{y}')
        if focus_widget:
            self.after_idle(focus_widget.focus_set)
        
        # Add border
        container.configure(
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
        tk.Wm.wm_geometry(self._container, f'+{x}+{y}')
    
    def dnd_accept(
            self,
            event: tk.Event,
            source: DnDItem
    ) -> DnDItem | None:
        assert isinstance(source, DnDItem), source
        
        if self != source and self._container.master == source._container.master:
             # are siblings
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
        
        container = self._container
        dnd_container = self._dnd_container
        
        # Restore `self` to become a regular widget
        tk.Wm.wm_forget(container, container)
        container.configure(  # remove border
            background=self._initial_background,
            padx=0,
            pady=0
        )
        
        # Remove and then put the widget onto the canvas again
        dnd_container.delete(self._oid)
        self._oid = dnd_container.create_window(
            0, 0, window=container, tags=dnd_container.dnd_tag
        )
        dnd_container.dnd_oids.clear()
        dnd_container.dnd_oids.update(
            {item._oid: item for item in dnd_container.dnd_items}
        )
        dnd_container._put(self)
        dnd_container._resize(self)
        
        unbind(self._initial_widget, self._motion_pattern, self._motion_id)
        unbind(self._initial_widget, self._release_pattern, self._release_id)
        self._target = self._initial_widget = None
        self._dnd_active = False
        
        if dnd_container._dnd_end_callback:
            dnd_container._dnd_end_callback(event, target)


class OrderlyDnDItem(DnDItem):
    def dnd_enter(
            self,
            event: tk.Event,
            source: DnDItem,
            old_target: DnDItem
    ):
        super().dnd_enter(event, source, old_target)
        
        dnd_container = self._dnd_container
        items = dnd_container.dnd_items
        new_source_idx = items.index(self)
        new_target_idx = items.index(source)
        
        # Exchange the indices of target (`self`) and `source` widgets
        items[new_target_idx] = self
        items[new_source_idx] = source
        
        # Update the oid dictionary
        oids = dnd_container.dnd_oids
        oids.clear()
        oids.update({ item._oid: item for item in items })
        
        # Exchange the geometries
        geometries = dnd_container._dnd_geometries
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
        dnd_container._put(self, new_target_geo)
        dnd_container._resize(self, new_target_geo)


class DnDContainer(tk.Canvas):
    _dnd_tag = 'dnd-item'
    
    def __init__(
            self,
            *args,
            dnd_start_callback: Callable | None = None,
            dnd_end_callback: Callable | None = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bind('<Configure>', self._on_configure)
        self._canvas_w: int = 0
        self._canvas_h: int = 0
        self._dnd_items: list[DnDItem] = []
        self._dnd_oids: dict[DnDItem] = {}
        self._select_mode: bool = False
        self.set_dnd_start_callback(dnd_start_callback)
        self.set_dnd_end_callback(dnd_end_callback)
    
    @property
    def dnd_tag(self) -> str:
        return self._dnd_tag
    
    @property
    def dnd_items(self) -> list[DnDItem]:
        return self._dnd_items
    
    @property
    def dnd_oids(self) -> dict[int, DnDItem]:
        return self._dnd_oids
    
    @property
    def select_mode(self) -> bool:
        return self._select_mode
    
    @property
    def selected_items(self) -> list[DnDItem]:
        return [ item for item in self._dnd_items if item.selected.get() ]
    
    @property
    def deselected_items(self) -> list[DnDItem]:
        return [ item for item in self._dnd_items if not item.selected.get() ]
    
    def _on_configure(self, event: tk.Event | None = None):
        if event is None:
            self.update_idletasks()
            self._canvas_w: int = self.winfo_width()
            self._cnavas_h: int = self.winfo_height()
        else:
            self._canvas_w: int = event.width
            self._canvas_h: int = event.height
        
        if self._dnd_items:
            for item, geo in zip(self._dnd_items, self._dnd_geometries):
                self._put(item, geo)
                self._resize(item, geo)
    
    def select_all(self):
        for item in self._dnd_items:
            item.selected.set(True)
    
    def deselect_all(self):
        for item in self._dnd_items:
            item.selected.set(False)
    
    def toggle_select_mode(self) -> bool:
        self.set_select_mode(not self._select_mode)
        return self._select_mode
    
    def set_select_mode(self, enable: bool):
        assert isinstance(enable, bool), enable
        if not self._dnd_items:
            raise ValueError(
                "Selection mode cannot be set before `dnd_put` be called."
            )
        
        if self._select_mode == enable:
            return
        
        items = self._dnd_items
        for item in items:
            item.set_select_mode(enable)
        
        # Remove and then put the items onto the canvas again to update the
        # natural sizes
        params = self._dnd_geometry_params
        R, C = params["shape"]
        put_kw = {
            "items": [ items[r*C:(r+1)*C] for r in range(R) ],  # 1-D => 2-D
            "sticky": params["sticky"],
            "expand": params["expand"],
            "padding": params["padding"],
            "ipadding": params["ipadding"]
        }
        self.dnd_forget(destroy=False)
        self.dnd_put(**put_kw)
        
        self._select_mode = enable
    
    def set_dnd_start_callback(self, callback: Callable | None):
        """
        The callback function will be executed once DnD starts.
        This callback function will then receive two arguments. The first
        argument, namely `event`, has a type of `tk.Event`. The second arguemnt,
        namely `source`, has a type of `DnDItem`.
        """
        assert callable(callback) or callback is None, callback
        self._dnd_start_callback = callback
    
    def set_dnd_end_callback(self, callback: Callable | None):
        """
        The callback function will be executed once DnD ends.
        This callback function will then receive two arguments. The first
        argument, namely `event`, has a type of `tk.Event`. The second arguemnt,
        namely `target`, has a type of `DnDItem` or `None`.
        """
        assert callable(callback) or callback is None, callback
        self._dnd_end_callback = callback
    
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
        
        if self.find_withtag(self.dnd_tag):
            raise RuntimeError(
                "Call `dnd_forget` or `delete` method to clean up last dnd "
                "items before calling `dnd_put` again."
            )
        
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
        items_flat = [ w for row in items for w in row ]
        assert all([ isinstance(w, DnDItem) for w in items_flat]), items
        oids = [
            self.create_window(
                0, 0, anchor='nw', window=item.container, tags=self.dnd_tag
            )
            for item in items_flat
        ]
        
        self.update_idletasks()
        widths, heights = list(), list()
        for oid, item in zip(oids, items_flat):
            item._oid: int = oid
            widths.append(item.container.winfo_reqwidth())
            heights.append(item.container.winfo_reqheight())
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
        for i, item in enumerate(items_flat):
            geometries.append(
                self._calculate_geometry(i, widths[i], heights[i])
            )
            
            # Setup widget's dnd functions
            self.rebind_dnd_start(item)
        
        self._dnd_geometries: list[dict] = geometries
        self._dnd_items: list[DnDItem] = items_flat
        self._dnd_oids: dict[int, DnDItem] = dict(zip(oids, items_flat))
        self.configure(width=canvas_w, height=canvas_h)
        self._on_configure()
    
    def dnd_forget(self, destroy: bool = True):
        if not self._dnd_items:
            return
        
        self.delete(self.dnd_tag)
        
        if destroy:
            for item in self._dnd_items:
                item.destroy()
        
        self._dnd_geometries.clear()
        self._dnd_items.clear()
        self._dnd_oids.clear()
        self._dnd_geometry_params.clear()
    
    def _put(self, item: DnDItem, geo: dict | None = None):
        assert isinstance(item, DnDItem), type(item)
        geo = geo or self._dnd_geometries[self._dnd_items.index(item)]
        x = max(int(geo["relx"] * self._canvas_w + geo["x"]), 0)
        y = max(int(geo["rely"] * self._canvas_h + geo["y"]), 0)
        
        self.itemconfigure(item._oid, anchor=geo["anchor"])
        self.coords(item._oid, x, y)
    
    def _resize(self, item: DnDItem, geo: dict | None = None):
        assert isinstance(item, DnDItem), type(item)
        geo = geo or self._dnd_geometries[self._dnd_items.index(item)]
        w = max(int(geo["relwidth"] * self._canvas_w + geo["width"]), 0)
        h = max(int(geo["relheight"] * self._canvas_h + geo["height"]), 0)
        
        self.itemconfigure(item._oid, width=w, height=h)
    
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
    container.pack(side='bottom', fill='both', expand=1)
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
    var1 = tk.BooleanVar(root, value=False)
    select_bt = ttk.Checkbutton(
        root,
        text='Select Mode',
        variable=var1,
        command=container.toggle_select_mode
    )
    select_bt.pack(side='top')
    root.place_window_center()
    root.update_idletasks()
    
    top = ttk.Toplevel(title='Button Trigger Drag and Drop')
    top.lift()
    top.after(300, top.focus_set)
    container = TriggerDnDContainer(top)
    container.pack(side='bottom', fill='both', expand=True)
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
    var2 = tk.BooleanVar(top, value=False)
    select_bt = ttk.Checkbutton(
        top,
        text='Select Mode',
        variable=var2,
        command=container.toggle_select_mode
    )
    select_bt.pack(side='top')
    top.place_window_center()
    
    root.mainloop()

