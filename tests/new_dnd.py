#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:48:41 2024

@author: tungchentsai
"""

import tkinter as tk
import ttkbootstrap as ttk

from tkinter_extensions.utils import unbind
# =============================================================================
# ---- Drag and Drop
# =============================================================================
class DnDWidget(tk.Frame):
    def __init__(
            self,
            *args,
            dnd_bordercolor: str = 'yellow',
            dnd_borderwidth: int = 1,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self.master, BaseDnDContainer):
            raise ValueError(
                "A `DnDWidget` must have a parent of type `BaseDnDContainer` "
                f"but got {repr(self.master)} of type `{type(self.master)}`."
            )
        self._dnd_bordercolor: str = dnd_bordercolor
        self._dnd_borderwidth: int = dnd_borderwidth
        self._dnd_container: BaseDnDContainer = self.master
    
    def _on_motion(self, event: tk.Event):
        self.dnd_motion(event)
        
        # Update target
        x, y = event.x, event.y
        dnd_oids = self._dnd_container.dnd_oids
        new_target = None
        for oid in self._dnd_container.find_overlapping(x, y, x, y):
            try:
                hover = dnd_oids[oid]
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
            widget.configure(cursor=self._initial_cursor)
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
        
        x, y = self.winfo_rootx(), self.winfo_rooty()
        w, h = self.winfo_width(), self.winfo_height()
        
        self.__root = self._root()
        self._target = None
        self._offset_x = x - event.x_root
        self._offset_y = y - event.y_root
        self._motion_pattern = f'<B{button}-Motion>'
        self._release_pattern = f'<ButtonRelease-{button}>'
        self._initial_widget = widget = event.widget
        self._initial_cursor = widget['cursor'] or ''
        self._initial_background = self["background"]
        self._initial_padx = self["padx"]
        self._initial_pady = self["pady"]
        
        focus_widget = self.focus_get()
        tk.Wm.wm_manage(self, self)  # make self frame become a toplevel
        tk.Wm.wm_overrideredirect(self, True)
        tk.Wm.wm_attributes(self, '-topmost', True)
        tk.Wm.wm_geometry(self, f'{w}x{h}+{x}+{y}')
        self.after_idle(focus_widget.focus_set)
        
        self.configure(
            background=self._dnd_bordercolor,
            padx=self._dnd_borderwidth,
            pady=self._dnd_borderwidth
        )
        widget.configure(cursor='hand2')
        
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
            source: tk.BaseWidget
    ) -> tk.BaseWidget | None:
        if self != source and self.master == source.master:  # is a sibling
            return self
    
    def dnd_enter(
            self,
            event: tk.Event,
            source: tk.BaseWidget,
            old_target: tk.BaseWidget
    ):
        pass
    
    def dnd_leave(
            self,
            event: tk.Event,
            source: tk.BaseWidget,
            new_target: tk.BaseWidget
    ):
        pass
    
    def dnd_commit(
            self,
            event: tk.Event,
            source: tk.BaseWidget
    ):
        pass
    
    def dnd_end(self, event: tk.Event, target: tk.BaseWidget):
        container = self._dnd_container
        
        # Restore `self` to become a regular widget
        tk.Wm.wm_forget(self, self)
        
        # Recreate the widget onto the canvas
        container.delete(self._oid)
        container.dnd_oids.pop(self._oid)
        self._oid = container.create_window(0, 0, window=self)
        container.dnd_oids[self._oid] = self  # update oid
        container._put(self)
        container._resize(self)


class BaseDnDContainer(tk.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind('<Configure>', self._on_configure)
        self._dnd_widgets: list[DnDWidget] = []
        self._dnd_place_info: list[dict] = []
        self._dnd_oids: dict[DnDWidget] = {}
    
    @property
    def dnd_widgets(self) -> list[DnDWidget]:
        return self._dnd_widgets
    
    @property
    def dnd_oids(self) -> dict[DnDWidget]:
        return self._dnd_oids
    
    def _on_configure(self, event: tk.Event | None = None):
        if event is None:
            self.update_idletasks()
            self._canvas_w = self.winfo_width()
            self._cnavas_h = self.winfo_height()
        else:
            self._canvas_w, self._canvas_h = event.width, event.height
        
        for widget, info in zip(self._dnd_widgets, self._dnd_place_info):
            self._put(widget, info)
            self._resize(widget, info)
    
    def bind_dnd_start(self, moved: DnDWidget):
        """
        Overwrite this method to customize the trigger widget
        """
        raise NotImplementedError()
    
    def _bind_dnd_start(
            self,
            sequence: str,
            *,
            trigger: tk.BaseWidget,
            moved: DnDWidget
    ):
        """
        Bind the `moved.dnd_start` method to `trigger` widget and its
        descendants with `sequence` events.
        """
        assert isinstance(moved, DnDWidget), (type(moved), moved)
        
        trigger.configure(cursor='hand2')
        trigger._dnd_start_id = trigger.bind(sequence, moved.dnd_start)
        for child in trigger.winfo_children():
            self._bind_dnd_start(sequence, trigger=child, moved=moved)
    
    def dnd_put(
            self,
            widgets: list[list[DnDWidget]] | list[DnDWidget],
            orient: str = 'vertical',
            sticky: str | None = None,
            expand: tuple[bool, bool] | list[bool] | bool = False,
            padding: tuple[int, int] | list[int] | int = 0,
            ipadding: tuple[int, int] | list[int] | int = 0
    ):
        """
        Use this function to put `widgets` into the container (canvas).
        
        `widgets` must be a list of widgets or a list of lists of 
        widgets. If `widgets` is a list of lists of widgets (a 2-D array of 
        widgets), the result layout will be in the same order as in the 
        `widgets`. Otherwise, one can specify the orientation by `orient`, e.g., 
        'vertical' (default) or 'horizontal'.
        
        Other arguments work like the arguments in the grid layout manager.
        """
        assert isinstance(widgets, list), widgets
        assert isinstance(expand, (tuple, list, bool, int)), expand
        assert isinstance(padding, (tuple, list, int)), padding
        assert isinstance(ipadding, (tuple, list, int)), ipadding
        assert orient in ('horizontal', 'vertical'), orient
        
        if not isinstance(widgets[0], list):
            if orient == 'horizontal':
                widgets = [widgets]
            else:  # vertical
                widgets = [ [w] for w in widgets ]
        nrows, ncols = len(widgets), len(widgets[0])
        
        pairs = list()
        for arg in [expand, padding, ipadding]:
            if isinstance(arg, (tuple, list)):
                assert len(arg) == 2, arg
                arg = tuple(arg)
            else:
                arg = (arg, arg)
            pairs.append(arg)
        expand, (padx, pady), (ipadx, ipady) = pairs
        
        # Calculate canvas size and place widgets onto the canvas
        widgets_flat = [ w for row in widgets for w in row ]
        assert all([ isinstance(w, DnDWidget) for w in widgets_flat]), widgets
        widget_ids = [ self.create_window(0, 0, anchor='nw', window=widget)
                       for widget in widgets_flat ]
        
        self.update_idletasks()
        widths, heights = list(), list()
        for oid, widget in zip(widget_ids, widgets_flat):
            widget._oid = oid
            widths.append(widget.winfo_reqwidth())
            heights.append(widget.winfo_reqheight())
        cell_w, cell_h = max(widths), max(heights)
        canvas_w = ncols*cell_w + 2*padx + 2*(ncols - 1)*ipadx
        canvas_h = nrows*cell_h + 2*pady + 2*(nrows - 1)*ipady
        params = {
            "expand": expand,
            "padding": (padx, pady),
            "ipadding": (ipadx, ipady),
            "grid_size": (nrows, ncols),
            "sticky": sticky,
            "cell_w": cell_w, "cell_h": cell_h,
        }
        self.configure(width=canvas_w, height=canvas_h)
        self.update_idletasks()
        
        # Calculate place_info and update widgets' position and size
        place_info = []
        i = 0
        for r, row in enumerate(widgets):
            for c, widget in enumerate(row):
                place_info.append(
                    self._calculate_place_info(
                        r, c, widths[i], heights[i], **params
                    )
                )
                i += 1
        
        # Setup widget's dnd functions
        for widget in widgets_flat:
            self.bind_dnd_start(widget)
        
        self._dnd_place_info: list[dict] = place_info
        self._dnd_widgets: list[DnDWidget] = widgets_flat
        self._dnd_oids = dict(zip(widget_ids, widgets_flat))
        self._on_configure()
    
    def _put(self, widget: DnDWidget, info: dict | None = None):
        info = info or self._dnd_place_info[self._dnd_widgets.index(widget)]
        x = max(int(info["relx"] * self._canvas_w + info["x"]), 0)
        y = max(int(info["rely"] * self._canvas_h + info["y"]), 0)
        
        self.itemconfigure(widget._oid, anchor=info["anchor"])
        self.coords(widget._oid, x, y)
    
    def _resize(self, widget: DnDWidget, info: dict | None = None):
        info = info or self._dnd_place_info[self._dnd_widgets.index(widget)]
        w = max(int(info["relwidth"] * self._canvas_w + info["width"]), 0)
        h = max(int(info["relheight"] * self._canvas_h + info["height"]), 0)
        
        self.itemconfigure(widget._oid, width=w, height=h)
    
    def _calculate_place_info(
            self,
            r, c,
            width, height,
            cell_w, cell_h,
            grid_size,
            sticky, expand,
            padding, ipadding
    ) -> dict:
        """
        This function simulates the grid layout manager by calculating the 
        position and size of each widget in the cell.
        """
        def _calc(
                xy, dim, dimension, L, dimension_cell, n, N,
                _sticky, expand, pad, ipad
        ) -> str:
            xy_cell = n*L + (N - 2*n)*pad + 2*n*ipad
            dimension_cell *= N
            if expand:
                if _sticky == (True, True):  # fill
                    place_info.update({
                        f"rel{xy}": n/N + 1./(2.*N),
                        xy: (xy_cell - n*L)//N + (dimension_cell - L)//(2*N),
                        f"rel{dim}": 1. / N,
                        dim: (dimension_cell - L) // N
                    })
                    return ''  # center
                
                place_info.update({dim: dimension})
                if _sticky == (True, False):
                    place_info.update({
                        f"rel{xy}": n / N,
                        xy: (xy_cell - n*L) // N
                    })
                    return 'w' if xy == 'x' else 'n'  # lower bound
                elif _sticky == (False, True):
                    place_info.update({
                        f"rel{xy}": n/N + 1./N,
                        xy: (xy_cell - n*L)//N + (dimension_cell - L)//N
                    })
                    return 'e' if xy == 'x' else 's'  # higher bound
                # (False, False) => center
                place_info.update({
                    f"rel{xy}": n/N + 1./(2.*N),
                    xy: (xy_cell - n*L)//N + (dimension_cell - L)//(2*N)
                })
                return ''  # center
            
            # Don't expand
            if _sticky == (True, True):
                place_info.update({
                    xy: xy_cell//N + dimension_cell//(2*N),
                    dim: dimension_cell // N
                })
                return ''  # center
            
            place_info.update({dim: dimension})
            if _sticky == (True, False):
                place_info[xy] = xy_cell // N
                return 'w' if xy == 'x' else 'n'  # lower bound
            if _sticky == (False, True):
                place_info[xy] = xy_cell//N + dimension_cell//N
                return 'e' if xy == 'x' else 's'  # higher bound
            # (False, False) => center
            place_info[xy] = xy_cell//N + dimension_cell//(2*N)
            return ''  # center
        #
        
        place_info = dict.fromkeys(
            ["relx", "x",
             "rely", "y",
             "relwidth", "width",
             "relheight", "height"],
            0.
        )
        nrows, ncols = grid_size
        sticky = sticky or ''
        sticky_x = ('w' in sticky, 'e' in sticky)
        sticky_y = ('n' in sticky, 's' in sticky)
        
        anchor_x = _calc('x', 'width', width, self._canvas_w, cell_w,
            c, ncols, sticky_x, expand[0], padding[0], ipadding[0])
        anchor_y = _calc('y', 'height', height, self._canvas_h, cell_h,
            r, nrows, sticky_y, expand[1], padding[1], ipadding[1])
        
        place_info["anchor"] = (anchor_x + anchor_y) or 'center'
        
        return place_info


class OrderlyContainer(BaseDnDContainer):
    def bind_dnd_start(self, moved: DnDWidget):
        self._bind_dnd_start('<ButtonPress-1>', trigger=moved, moved=moved)


# =============================================================================
# ---- Test
# =============================================================================
root = ttk.Window('New DnD Test', themename='cyborg')

dnd = OrderlyContainer(root)
dnd.pack(fill='both', expand=True)

wrapper1 = DnDWidget(dnd)
label1 = ttk.Label(wrapper1, text='-'.join(['Label1']*10))
label1.pack()

wrapper2 = DnDWidget(dnd)
label2 = ttk.Label(wrapper2, text='-'.join(['Label2']*10))
label2.pack()

dnd.dnd_put([wrapper1, wrapper2], padding=6, ipadding=3, expand=True)

root.mainloop()

