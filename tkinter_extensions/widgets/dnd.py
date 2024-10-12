#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:48:41 2024

@author: tungchentsai
"""

from __future__ import annotations
import math
from typing import Any, Callable
import tkinter as tk

import ttkbootstrap as ttk

from tkinter_extensions.utils import unbind
from tkinter_extensions import variables as vrb
# =============================================================================
# ---- Drag and Drop
# =============================================================================
class DnDItem(ttk.Frame):
    """
    `DnDItem` is a frame widget which can be put into `DnDContainer` (a canvas
    widget) using the method `DnDContainer.dnd_put`.
    """
    def __init__(
            self,
            master=None,
            bootstyle='default',
            dnd_bordercolor: str = 'warning',
            dnd_borderwidth: int = 1,
            selectbutton: bool = False,
            selectbutton_kw: dict = {
                "bootstyle": 'primary',
            },
            dragbutton: bool = False,
            dragbutton_kw: dict = {
                "text": '::',
                "takefocus": False,
                "bootstyle": 'link-primary'
            },
            button_sticky: str = '',
            **kwargs
    ):
        if not isinstance(master, DnDContainer):
            raise ValueError(
                "`master` must be of type `DnDContainer` but got "
                "{repr(master)} of type `{type(master)}`."
            )
        
        # Create a wrapper frame
        self._wrapper = tk.Frame(master, padx=0, pady=0)
        self._wrapper.grid_rowconfigure(0, weight=1)
        self._wrapper.grid_columnconfigure(2, weight=1)
        
        self._background_fm = ttk.Frame(self._wrapper, bootstyle=bootstyle)
        self._background_fm.place(relwidth=1., relheight=1.)
        
        # Make a selectbutton and show it only when the rearrangement is active
        self._rearrange_active: bool = False
        self._selected = vrb.BooleanVar(self._wrapper, value=False)
        if selectbutton:
            self._select_bt = ttk.Checkbutton(
                self._wrapper,
                variable=self._selected,
                **selectbutton_kw
            )
            self._select_bt.grid(
                row=0, column=0, sticky=button_sticky, padx=(3, 0)
            )
            self._select_bt.grid_remove()
        else:
            self._select_bt = None
        
        if dragbutton:
            self._drag_bt = ttk.Button(self._wrapper, **dragbutton_kw)
            self._drag_bt.grid(row=0, column=1, sticky=button_sticky)
            self._drag_bt.grid_remove()
            self._drag_bt.dnd_trigger: bool = True
        else:
            self._drag_bt = None
        
        # Content frame
        super().__init__(self._wrapper, bootstyle=bootstyle, **kwargs)
        self.grid(row=0, column=2, sticky='nswe')
        
        # Pad `self` later to make it work like a border
        self._dnd_bordercolor = dnd_bordercolor
        self._dnd_borderwidth: int = int(dnd_borderwidth)
        self._dnd_container: DnDContainer = master
        self._dnd_active: bool = False
    
    @property
    def wrapper(self) -> tk.Frame:
        return self._wrapper
    
    @property
    def dnd_active(self) -> bool:
        """
        Returns whether the DnD is currently working.
        """
        return self._dnd_active
    
    @property
    def selected(self) -> vrb.BooleanVar:
        """
        Returns whether this item is selected.
        """
        return self._selected
    
    @property
    def rearrange_active(self) -> bool:
        """
        Returns whether the rearrangement mode is on.
        
        You can switch the rearrangement mode on or off by the method
        `set_rearrangement`.
        """
        return self._rearrange_active
    
    def set_rearrangement(self, enable: bool):
        """
        Switches the rearrangement mode on or off.
        
        If `enable` is `True`, turn on the rearrangement mode and show the
        available buttons. Otherwise, turn off the rearrangement mode and hide
        the available buttons.
        """
        assert isinstance(enable, bool), enable
        
        if not self._select_bt and not self._drag_bt:
            raise ValueError(
                "Can't switch the rearrangement mode on or off when both "
                "`selectbutton` and `dragbutton` are `False`."
            )
        
        if self._rearrange_active == enable:
            return
        
        if enable:  # show selectbutton
            if self._select_bt:
                self._select_bt.grid()
            if self._drag_bt:
                self._drag_bt.grid()
        else:  # hide selectbutton
            if self._select_bt:
                self._select_bt.grid_remove()
            if self._drag_bt:
                self._drag_bt.grid_remove()
        
        self._selected.set(False)
        self._rearrange_active = enable
    
    def _on_motion(self, event: tk.Event):
        """
        This method will be called when the mouse cursor moves after the DnD
        starts.
        """
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
        self._target = new_target
        if old_target is not new_target:
            if old_target is not None:  # leaving the old target
                old_target.dnd_leave(event, self, new_target)
            if new_target is not None:  # entering a new target
                new_target.dnd_enter(event, self, old_target)
    
    def _on_release(self, event: tk.Event | None):
        """
        This method will be called once the button is released.
        """
        self._finish(event, commit=True)
    
    def _finish(self, event: tk.Event | None, commit: bool):
        target = self._target
        try:
            if commit and target is not None:
                target.dnd_commit(event, self)
        finally:
            self.dnd_end(event)
    
    def cancel(self, event: tk.Event | None = None):
        """
        Stop the drag and drop. `dnd_commit` will not be called.
        """
        self._finish(event, commit=False)
    
    def dnd_start(self, event: tk.Event):
        """
        Starts drag and drop mechanism.
        
        This method is a entry point for DnD and should be bind to some widget.
        """
        assert isinstance(button := event.num, int), button
        
        widget = event.widget
        wrapper = self._wrapper
        dnd_container = self._dnd_container
        
        if dnd_container._dnd_start_callback:
            dnd_container._dnd_start_callback(event, self)
        
        x, y = wrapper.winfo_rootx(), wrapper.winfo_rooty()
        w, h = wrapper.winfo_width(), wrapper.winfo_height()
        
        self._dnd_active = True
        self._target: DnDItem | None = None
        self._offset_x: int = x - event.x_root
        self._offset_y: int = y - event.y_root
        self._motion_pattern = f'<B{button}-Motion>'
        self._release_pattern = f'<ButtonRelease-{button}>'
        self._initial_widget: tk.BaseWidget | None = widget
        self._initial_items: list[DnDItem] | None = dnd_container.dnd_items.copy()
        self._initial_background = wrapper["background"]
        self._dnd_container_x: int = self._dnd_container.winfo_rootx()
        self._dnd_container_y: int = self._dnd_container.winfo_rooty()
        
        focus_widget: tk.BaseWidget | None = self.focus_get()
        tk.Wm.wm_manage(wrapper, wrapper)  # make wrapper become a toplevel
        tk.Wm.wm_overrideredirect(wrapper, True)
        tk.Wm.wm_attributes(wrapper, '-topmost', True)
        tk.Wm.wm_geometry(wrapper, f'{w}x{h}+{x}+{y}')
        if focus_widget:
            self.after_idle(focus_widget.focus_set)
        
        # Add border
        style = ttk.Style.get_instance()
        if (bordercolor := style.colors.get(self._dnd_bordercolor)) is None:
            bordercolor = self._dnd_bordercolor
        wrapper.configure(
            background=bordercolor,
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
        tk.Wm.wm_geometry(self._wrapper, f'+{x}+{y}')
    
    def dnd_accept(
            self,
            event: tk.Event,
            source: DnDItem
    ) -> DnDItem | None:
        """
        When the mouse cursor is hovering on `self`, this method will be called
        and returns a new target widget if the condition is matched.
        """
        assert isinstance(source, DnDItem), source
        
        if self != source and self._wrapper.master == source._wrapper.master:
             # are siblings
            return self
    
    def dnd_enter(
            self,
            event: tk.Event,
            source: DnDItem,
            old_target: DnDItem
    ):
        """
        When the mouse cursor moves into this item, this method will be called.
        """
        assert isinstance(source, DnDItem), source
    
    def dnd_leave(
            self,
            event: tk.Event,
            source: DnDItem,
            new_target: DnDItem
    ):
        """
        When the mouse cursor leave this item, this method will be called.
        """
        assert isinstance(source, DnDItem), source
    
    def dnd_commit(
            self,
            event: tk.Event,
            source: DnDItem
    ):
        """
        When the DnD is going to end, this method of the new target will be
        called.
        """
        assert isinstance(source, DnDItem), source
    
    def dnd_end(self, event: tk.Event | None):
        """
        This method is the last DnD method, which will be called when DnD ends.
        """
        initial_items = self._initial_items
        wrapper = self._wrapper
        dnd_container = self._dnd_container
        
        # Restore `self` to become a regular widget
        tk.Wm.wm_forget(wrapper, wrapper)
        wrapper.configure(  # remove border
            background=self._initial_background,
            padx=0,
            pady=0
        )
        
        # Remove and then put the widget onto the canvas again
        dnd_container.delete(self._oid)
        self._oid = dnd_container.create_window(
            0, 0, window=wrapper, tags=dnd_container.dnd_tag
        )
        dnd_container.dnd_oids.clear()
        dnd_container.dnd_oids.update(
            {item._oid: item for item in dnd_container.dnd_items}
        )
        dnd_container._put(self)
        dnd_container._resize(self)
        
        # Restore settings from `dnd_start`
        unbind(self._initial_widget, self._motion_pattern, self._motion_id)
        unbind(self._initial_widget, self._release_pattern, self._release_id)
        self._target = self._initial_widget = self._initial_items = None
        self._dnd_active = False
        
        if dnd_container._dnd_end_callback:
            dnd_container._dnd_end_callback(event, initial_items)


class OrderlyDnDItem(DnDItem):
    def dnd_enter(
            self,
            event: tk.Event,
            source: DnDItem,
            old_target: DnDItem
    ):
        """
        Exchange the position of the `source` item and the new target the mouse
        cursor just entered.
        """
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
    """
    `DnDContainer` is a canvas widget that handles DnD mechanism. The DnD
    items with type `DnDItem` should be put onto this canvas with the method
    `DnDContainer.dnd_put`.
    """
    
    _dnd_tag = 'dnd-item'
    
    def __init__(
            self,
            master=None,
            dnd_start_callback: Callable[[tk.Event, DnDItem], Any] | None = None,
            dnd_end_callback: Callable[[tk.Event, list[DnDItem]], Any] | None \
                = None,
            **kwargs
    ):
        super().__init__(master=master, **kwargs)
        self.bind('<Configure>', self._on_configure)
        self._canvas_w: int = 0
        self._canvas_h: int = 0
        self._rearrange_active: bool = False
        self._dnd_items: list[DnDItem] = []
        self._dnd_oids: dict[DnDItem] = {}
        self.set_dnd_start_callback(dnd_start_callback)
        self.set_dnd_end_callback(dnd_end_callback)
    
    @property
    def dnd_tag(self) -> str:
        return self._dnd_tag
    
    @property
    def dnd_items(self) -> list[DnDItem]:
        """
        Returns the items put onto this canvas with `self.dnd_put`.
        
        Please avoid modifying the returned list.
        """
        return self._dnd_items
    
    @property
    def dnd_oids(self) -> dict[int, DnDItem]:
        """
        Returns the DnD items' canvas oids.
        
        Please avoid modifying the returned list.
        """
        return self._dnd_oids
    
    @property
    def rearrange_active(self) -> bool:
        """
        Returns whether the rearrangement mode is on.
        """
        return self._rearrange_active
    
    @property
    def selected_items(self) -> list[DnDItem]:
        """
        Returns the currently selected DnD items.
        """
        return [ item for item in self._dnd_items if item.selected.get() ]
    
    @property
    def deselected_items(self) -> list[DnDItem]:
        """
        Returns the DnD items which are not currently selected.
        """
        return [ item for item in self._dnd_items if not item.selected.get() ]
    
    def _consistent_rearrangement(self) -> bool:
        if not self._dnd_items:
            return True
        
        items = self._dnd_items.copy()
        active = items.pop().rearrange_active
        return all( item.rearrange_active == active for item in items )
    
    def _on_configure(self, event: tk.Event | None = None):
        """
        Refreshes the DnD items' positions and sizes.
        """
        if event is None:
            self.update_idletasks()
            self._canvas_w: int = self.winfo_width()
            self._cnavas_h: int = self.winfo_height()
        else:
            self._canvas_w: int = event.width
            self._canvas_h: int = event.height
        
        if self._dnd_items:
            for item, geometry in zip(self._dnd_items, self._dnd_geometries):
                self._put(item, geometry)
                self._resize(item, geometry)
    
    def dnd_put(
            self,
            items: list[list[DnDItem]] | list[DnDItem],
            orient: str = 'vertical',
            sticky: str = '',
            expand: tuple[bool, bool] | list[bool] | bool = False,
            ipadding: tuple[int, int] | list[int] | int = 0,
            padding: tuple[int, int] | tuple[int, int, int, int] | list[int] |
                int = 0
    ):
        """
        Use this method to put `items` onto this canvas container.
        
        `items` must be a list of `DnDItem`s or a list of lists of 
        `DnDItem`s. If `items` is a list of lists of `DnDItem`s (a 2-D array of 
        `DnDItem`s), the result layout will be in the same order as in the 
        `items`. Otherwise, one can specify the orientation by `orient`, e.g., 
        'vertical' (default) or 'horizontal'.
        
        `padding` must be a `int`, a tuple of 2 `int`s, a tuple of 4 `int`s, or
        a list of `int`. If `padding` is a tuple or list of 2 `int`s, the first
        `int` is in x direction and the second `int` is in y direction. If
        `padding` is a tuple or list of 4 `int`s, they are left, top, right,
        and bottom padding, in order.
        
        Other arguments work like the arguments for the grid layout manager.
        """
        assert isinstance(items, list), items
        assert isinstance(expand, (tuple, list, bool, int)), expand
        assert isinstance(padding, (tuple, list, int)), padding
        assert isinstance(ipadding, (tuple, list, int)), ipadding
        assert orient in ('horizontal', 'vertical'), orient
        
        if self.find_withtag(self.dnd_tag):
            raise RuntimeError(
                "Call `dnd_forget` or `delete` method to clean up last DnD "
                "items before calling `dnd_put` again."
            )
        
        if isinstance(padding, (tuple, list)):
            assert len(padding) in (2, 4), padding
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
            else:
                padding = tuple(padding)
        else:  # int
            padding = (padding, padding, padding, padding)
        
        pairs = list()
        for name, arg in [('expand', expand), ('ipadding', ipadding)]:
            if isinstance(arg, (tuple, list)):
                assert len(arg) == 2, (name, arg)
                arg = tuple(arg)
            else:
                arg = (arg, arg)
            pairs.append(arg)
        expand, (ipadx, ipady) = pairs
        assert isinstance(sticky, str), (type(sticky), sticky)
        assert set(sticky).issubset('nesw'), sticky
        assert all( isinstance(p, int) for p in padding ), padding
        assert all( p >= 0 for p in padding ), padding
        assert all( isinstance(p, int) for p in (ipadx, ipady) ), (ipadx, ipady)
        assert all( p >= 0 for p in (ipadx, ipady) ), (ipadx, ipady)
        
        # Ensure `items` is a 2D structure
        if not isinstance(items[0], list):
            if orient == 'horizontal':
                items = [items]
            else:  # vertical
                items = [ [w] for w in items ]
        R, C = len(items), max( len(item) for item in items )
        
        # Make items' rearrangements state consistent
        items_flat = [ w for row in items for w in row ]
        assert all([ isinstance(w, DnDItem) for w in items_flat]), items
        rearrange_active = self._rearrange_active
        for item in items_flat:
            if item.rearrange_active != rearrange_active:
                item.set_rearrangement(rearrange_active)
        
        # Calculate canvas size and place items onto the canvas
        oids = [
            self.create_window(
                0, 0, anchor='nw', window=item.wrapper, tags=self.dnd_tag
            )
            for item in items_flat
        ]
        self.update_idletasks()
        widths, heights = list(), list()
        for oid, item in zip(oids, items_flat):
            item._oid: int = oid
            widths.append(item.wrapper.winfo_reqwidth())  # natural width
            heights.append(item.wrapper.winfo_reqheight())  # natural height
        cell_w, cell_h = max(widths), max(heights)
        canvas_w: int = C*cell_w + (padding[0] + padding[2]) + (C - 1)*ipadx
        canvas_h: int = R*cell_h + (padding[1] + padding[3]) + (R - 1)*ipady
        self._dnd_geometry_params = {
            "shape": (R, C),
            "expand": expand,
            "ipadding": (ipadx, ipady),
            "padding": padding,
            "sticky": sticky,
            "natural_canvas_size": (canvas_w, canvas_h)
        }
        
        # Calculate geometry and update items' position and size.
        # The resultant geometries are based on these items' natural sizes
        geometries = []
        for i, item in enumerate(items_flat):
            geometries.append(
                self._calculate_geometry(i, widths[i], heights[i])
            )
            
            # Setup widget's DnD functions
            self.rebind_dnd_start(item)
        
        self._dnd_geometries: list[dict] = geometries
        self._dnd_items: list[DnDItem] = items_flat
        self._dnd_oids: dict[int, DnDItem] = dict(zip(oids, items_flat))
        self.configure(width=canvas_w, height=canvas_h)
        self._on_configure()  # ensure the item layout is refreshed
    
    def dnd_forget(self, destroy: bool = True):
        """
        Removes the DnD items which were put onto the canvas. Destroy the items
        if `destroy` is `True`.
        """
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
    
    def _put(self, item: DnDItem, geometry: dict | None = None):
        """
        Updates the position of `item` according to `geometry`. If `geometry` is
        `None`, get the geometry from `self._dnd_geometries`.
        """
        assert isinstance(item, DnDItem), type(item)
        geometry = geometry or self._dnd_geometries[self._dnd_items.index(item)]
        x = max(int(geometry["relx"] * self._canvas_w + geometry["x"]), 0)
        y = max(int(geometry["rely"] * self._canvas_h + geometry["y"]), 0)
        
        self.itemconfigure(item._oid, anchor=geometry["anchor"])
        self.coords(item._oid, x, y)
    
    def _resize(self, item: DnDItem, geometry: dict | None = None):
        """
        Updates the size of `item` according to `geometry`. If `geometry` is
        `None`, get the geometry from `self._dnd_geometries`.
        """
        assert isinstance(item, DnDItem), type(item)
        geometry = geometry or self._dnd_geometries[self._dnd_items.index(item)]
        w = max(
            int(geometry["relwidth"] * self._canvas_w + geometry["width"]),
            0
        )
        h = max(
            int(geometry["relheight"] * self._canvas_h + geometry["height"]),
            0
        )
        
        self.itemconfigure(item._oid, width=w, height=h)
    
    def _calculate_geometry(self, idx, natural_width, natural_height) -> dict:
        """
        This method mimics the grid layout manager by calculating the 
        position and size of each widget in the cell.
        """
        def _calc(i, I, size, natural_L, stick, exp, ipad, p1, p2) -> dict:
            # Cell size (size_c) = L/I + (-(p1 + p2) - (I-1)*ipad)/I
            r = 1. / I  # relative part
            f = (-(p1 + p2) - (I - 1)*ipad)/I  # fixed part
            nat_cell_size = natural_L*r + f
            
            if exp:  # expand => use variable position and size
                if stick == (True, True):  # fill
                    return {
                        "relpos": (2*i + 1)*r / 2,
                        "pos": ((2*i + 1)*f + 2*i*ipad) / 2 + p1,
                        "relsize": r,
                        "size": f
                    }
                elif stick == (True, False):
                    return {
                        "relpos": i*r,
                        "pos": i*f + i*ipad + p1,
                        "relsize": 0.,
                        "size": size
                    }
                elif stick == (False, True):
                    return {
                        "relpos": (i + 1)*r,
                        "pos": (i + 1)*f + i*ipad + p1,
                        "relsize": 0.,
                        "size": size
                    }
                else:  # stick == (False, False)
                    return {
                        "relpos": (2*i + 1)*r / 2,
                        "pos": ((2*i + 1)*f + 2*i*ipad) / 2 + p1,
                        "relsize": 0.,
                        "size": size
                    }
            else:  # don't expand => use fixed position and size
                if stick == (True, True):  # fill
                    return {
                        "relpos": 0.,
                        "pos": ((2*i + 1)*nat_cell_size + 2*i*ipad) / 2 + p1,
                        "relsize": 0.,
                        "size": natural_L*r + f
                    }
                elif stick == (True, False):
                    return {
                        "relpos": 0.,
                        "pos": i*nat_cell_size + i*ipad + p1,
                        "relsize": 0.,
                        "size": size
                    }
                elif stick == (False, True):
                    return {
                        "relpos": 0.,
                        "pos": (i + 1)*nat_cell_size + i*ipad + p1,
                        "relsize": 0.,
                        "size": size
                    }
                else:  # stick == (False, False)
                    return {
                        "relpos": 0.,
                        "pos": ((2*i + 1)*nat_cell_size + 2*i*ipad) / 2 + p1,
                        "relsize": 0.,
                        "size": size
                    }
        #
        
        params = self._dnd_geometry_params
        R, C = params["shape"]
        expx, expy = params["expand"]
        ipadx, ipady = params["ipadding"]
        padding = params["padding"]
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
            c, C, natural_width, natural_w,
            sticky_x, expx, ipadx, padding[0], padding[2]
        )
        geometry_y = _calc(
            r, R, natural_height, natural_h,
            sticky_y, expy, ipady, padding[1], padding[3]
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
        Overwrites this method to customize the trigger widget.
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
        Rebinds the `moved.dnd_start` method to `trigger` widget and its
        descendants with `sequence` events.
        """
        assert isinstance(moved, DnDItem), (type(moved), moved)
        
        trigger.configure(cursor='hand2')
        if hasattr(trigger, '_dnd_start_id'):
            unbind(trigger, sequence, trigger._dnd_start_id)
        trigger._dnd_start_id = trigger.bind(sequence, moved.dnd_start, add=True)
        
        for child in trigger.winfo_children():
            self._rebind_dnd_start(sequence, trigger=child, moved=moved)
    
    def select_all(self):
        """
        Selects all DnD items.
        """
        for item in self._dnd_items:
            item.selected.set(True)
    
    def deselect_all(self):
        """
        Deselects all DnD items.
        """
        for item in self._dnd_items:
            item.selected.set(False)
    
    def toggle_rearrangement(self) -> bool:
        """
        Toggle on or off the rearrangement mode.
        """
        enable = not self._rearrange_active
        self.set_rearrangement(enable)
        return enable
    
    def set_rearrangement(self, enable: bool):
        """
        Set the rearrangement mode.
        
        If the rearrangement mode is on, the available rearrangement buttons
        for each item will appear on the left side of each DnD item. Otherwise,
        the rearrangement buttons for each DnD item will disappear.
        """
        assert isinstance(enable, bool), enable
        
        if self._rearrange_active == enable and self._consistent_rearrangement():
            return
        
        self._rearrange_active = enable
        
        if not self._dnd_items:
            return
        
        self._set_rearrangement(enable, self._dnd_geometry_params)
    
    def _set_rearrangement(self, enable: bool, geometry_params: dict):
        assert self._dnd_items, self._dnd_items
        
        items = self._dnd_items
        for item in items:
            item.set_rearrangement(enable)
        
        # Remove and then put the items onto the canvas again to update the
        # natural sizes
        params = geometry_params
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
    
    def set_dnd_start_callback(
            self,
            callback: Callable[[tk.Event, DnDItem], Any] | None
    ):
        """
        The callback function will be executed once DnD starts.
        
        This callback function will receive two arguments. The first
        argument, namely `event`, has a type of `tk.Event`. The second arguemnt,
        namely `source`, has a type of `DnDItem`.
        """
        assert callable(callback) or callback is None, callback
        self._dnd_start_callback = callback
    
    def set_dnd_end_callback(
            self,
            callback: Callable[[tk.Event, list[DnDItem]], Any] | None
    ):
        """
        The callback function will be executed once DnD ends.
        
        This callback function will receive two arguments. The first
        argument, namely `event`, has a type of `tk.Event`. The second arguemnt,
        namely `initial_items`, has a type of `list[DnDItem]`.
        """
        assert callable(callback) or callback is None, callback
        self._dnd_end_callback = callback


class RearrangedDnDContainer(DnDContainer):
    """
    DnDContainer which presents a rearrangement button allowing user to
    rearrange the items.
    """
    def __init__(
            self,
            master=None,
            button_loc: str = 'top-right',
            button_kw: dict = {
                "text": '⋯',
                "takefocus": False,
                "bootstyle": 'primary'
            },
            **kwargs
    ):
        super().__init__(master=master, **kwargs)
        
        if button_loc not in ('top-left', 'top-right',
                              'bottom-left', 'bottom-right'):
            raise ValueError(
                "`button_loc` must be one of 'top-left', 'top-right', "
                f"'bottom-left', and 'bottom-right' but got {repr(button_loc)}."
            )
        
        # Plase info
        if 'top' in button_loc:
            anchor = 'n'
            rely = 0.
        else:
            anchor = 's'
            rely = 1.
        if 'left' in button_loc:
            anchor += 'w'
            relx = 0.
        else:
            anchor += 'e'
            relx = 1.
        
        # Create the rearrangement button
        self._rearrange_bt = ttk.Button(
            self,
            command=self._on_button_clicked,
            **button_kw
        )
        self._rearrange_bt.place(relx=relx, rely=rely, anchor=anchor)
        self._rearrange_bt.configure(style=self._create_bt_style(new=True))
        
        self.update_idletasks()
        self._button_h: int = self._rearrange_bt.winfo_reqheight()
        self._button_loc: str = button_loc
        self._original_padding: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.configure(height=self._button_h)
        self.bind('<<ThemeChanged>>', self._create_bt_style)
        self.set_rearrange_commands({
            "label": 'Remove Selected', "command": self.remove_selected
        })
        self.set_other_commands()
        self.set_rearrange_callback()
    
    def _create_bt_style(self, event=None, new: bool = False) -> str:
        if new:
            bt_style = 'Rearrange.DnDContainer.' + self._rearrange_bt["style"]
        else:
            bt_style = self._rearrange_bt["style"]
        style = ttk.Style.get_instance()
        style.configure(bt_style, padding=(6, 3))  # modify padding
        
        return bt_style
    
    def _on_button_clicked(self):
        """
        This method will be called once the rearrangement button is clicked.
        This will bring up a right-click menu containing some options related
        to rearrangement.
        """
        rearrange_active = self._rearrange_active
        
        # Get button position (show the menu from the bottome left corner)
        bt = self._rearrange_bt
        x = bt.winfo_rootx()
        y = bt.winfo_rooty() + bt.winfo_height()
        
        # Create a right-click menu
        toggle = 'Done' if rearrange_active else 'Start'
        menu = tk.Menu(self)
        menu.add_command(
            label=f'{toggle} Rearrangement',
            command=self.toggle_rearrangement
        )
        
        if rearrange_active:
            ## Commands in rearrangement mode
            if self._rearrange_commands:
                menu.add_command(label='Select All', command=self.select_all)
                menu.add_command(label='Deselect All', command=self.deselect_all)
            for kw in self._rearrange_commands:
                if cmd := kw.get('command', None):
                    cmd = lambda c=cmd: (c(), self.set_rearrangement(False))
                    kw["command"] = cmd
                menu.add_command(**kw)
        else:
            ## Commands not in rearrangement mode
            for kw in self._other_commands:
                menu.add_command(**kw)
        
        if menu.index('end') > 0:  # has mode commands
            menu.insert_separator(1)  # insert a separator below the toggle
        
        menu.post(x, y)
        self.after_idle(menu.destroy)  # destroy the menu when it disappears
        
        if self._rearrange_callback:
            self._rearrange_callback(self)
    
    def remove_selected(self) -> list[DnDItem]:
        """
        Remove currently selected items.
        
        This method returns a list of the selected items.
        """
        if not (selected := self.selected_items.copy()):
            return selected
        
        items = self.deselected_items
        params = self._dnd_geometry_params
        C = params["shape"][1]
        R = math.ceil(len(items) / C)
        put_kw = {
            "items": [ items[r*C:(r+1)*C] for r in range(R) ],  # 1-D => 2-D
            "sticky": params["sticky"],
            "expand": params["expand"],
            "padding": params["padding"],
            "ipadding": params["ipadding"]
        }
        self.dnd_forget(destroy=False)
        if items:
            self.dnd_put(offset=False, **put_kw)
        
        for item in selected:
            item.destroy()
        
        return selected
    
    def dnd_put(self, *args, padding=0, offset: bool = True, **kwargs):
        """
        Modify the super-class method `dnd_put` to make space for the
        rearrrangement button before putting the items onto the canvas if
        `offset` is `True`. If `offset` is False, directly put the items onto
        the canvas as if the value of `padding` has been modified for the
        rearrangement button, and then calculate and save the original
        (unmodified) padding.
        """
        # Assume `padding` has been modified
        if not offset:
            super().dnd_put(*args, padding=padding, **kwargs)
            
            ## Offset upwards
            px1, py1, px2, py2 = padding
            if 'top' in self._button_loc:
                py1 -= self._button_h
            else:
                py2 -= self._button_h
            
            self._original_padding = (px1, py1, px2, py2)
            return
        
        # Modify the y padding to make space for the rearrangement button
        assert isinstance(padding, (tuple, list, int)), padding
        
        if isinstance(padding, (tuple, list)):
            assert len(padding) in (2, 4), padding
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
            else:
                padding = tuple(padding)
        else:  # int
            padding = (padding, padding, padding, padding)
        assert all( isinstance(p, int) for p in padding ), padding
        assert all( p >= 0 for p in padding ), padding
        
        self._original_padding = padding
        
        ## Offset downwards
        px1, py1, px2, py2 = padding
        if 'top' in self._button_loc:
            py1 += self._button_h
        else:
            py2 += self._button_h
        
        super().dnd_put(*args, padding=(px1, py1, px2, py2), **kwargs)
    
    def dnd_forget(self, *args, **kwargs):
        super().dnd_forget(*args, **kwargs)
        self.configure(height=self._button_h)
    
    def rebind_dnd_start(self, moved: DnDItem):
        """
        Rebind `dnd_start` to `moved._drag_bt`.
        
        Parameters
        ----------
        moved : DnDItem
            This item must have a `_drag_bt` attribute. Set the argument
            `dragbutton` of `moved` to `True` to create the drag button with
            name `_drag_bt`.
        
        Raises
        ------
        ValueError
            If `moved` does not have a drag button, a `ValueError` will be
            raised.
        
        Returns
        -------
        None.
        
        """
        if moved._drag_bt:
            self._rebind_dnd_start(
                '<ButtonPress-1>', trigger=moved._drag_bt, moved=moved
            )
    
    def set_rearrangement(self, enable: bool):
        """
        Set the rearrangement mode.
        
        If the rearrangement mode is on, the available rearrangement buttons
        for each item will appear on the left side of each DnD item. Otherwise,
        the rearrangement buttons for each DnD item will disappear.
        """
        assert isinstance(enable, bool), enable
        
        if self._rearrange_active == enable and self._consistent_rearrangement():
            return
        
        self._rearrange_active = enable
        
        if not self._dnd_items:
            return
        
        geometry_params = self._dnd_geometry_params.copy()
        geometry_params["padding"] = self._original_padding
        self._set_rearrangement(enable, geometry_params)
    
    def set_rearrange_commands(self, *kw_dictionaries: dict[str, Any]):
        """
        Sets the keyword arguments for each menu command which shows up once the
        rearrangement button is clicked and `self.rearrangement_active` is
        `True`.
        """
        self._rearrange_commands: list[dict[str, Any]] = list(kw_dictionaries)
    
    def set_other_commands(self, *kw_dictionaries: dict[str, Any]):
        """
        Sets the keyword arguments for each menu command which shows up once the
        rearrangement button is clicked and `self.rearrangement_active` is
        `False`.
        """
        self._other_commands: list[dict[str, Any]] = list(kw_dictionaries)
    
    def set_rearrange_callback(self, callback: Callable | None = None):
        """
        Sets the callback function which will be called every time the
        rearrangement button disappears.
        """
        self._rearrange_callback: Callable | None = callback


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
            item = OrderlyDnDItem(container, selectbutton=True)
            ttk.Button(
                item,
                text=f'|<{dash} ({r}, {c}) {dash}>|',
                takefocus=False,
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
    var = tk.BooleanVar(root, value=False)
    rearrange_bt = ttk.Checkbutton(
        root,
        text='Rearrange',
        variable=var,
        command=container.toggle_rearrangement
    )
    rearrange_bt.pack(side='top')
    root.place_window_center()
    root.update_idletasks()
    
    top = ttk.Toplevel(title='Button Trigger Drag and Drop')
    top.lift()
    top.after(300, top.focus_set)
    container = RearrangedDnDContainer(top)
    container.pack(side='bottom', fill='both', expand=True)
    items = list()
    for r in range(4):
        items.append([])
        for c in range(3):
            dash = '----' * random.randint(1, 5)
            item = OrderlyDnDItem(container, selectbutton=True, dragbutton=True)
            ttk.Label(
                item,
                text=f'|<{dash} ({r}, {c}) {dash}>|',
                bootstyle='success'
            ).pack(fill='both', expand=True)
            items[-1].append(item)
    container.dnd_put(
        items,
        sticky='nsw',
        expand=(False, True),
        padding=(3, 6),
        ipadding=12
    )
    top.place_window_center()
    
    root.mainloop()

