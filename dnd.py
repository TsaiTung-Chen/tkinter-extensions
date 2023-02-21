# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:26:58 2023

@author: Jeff_Tsai
"""

import tkinter as tk
import tkinter.dnd
from typing import Union, Optional, List, Tuple

import ttkbootstrap as ttk
# =============================================================================
# ---- Functions
# =============================================================================
class OrderedContainer(ttk.Canvas):
    """This container allows a drag-and-drop feature which is very useful when 
    one would like to reorder some widgets in the container. This can be 
    be achieved intuitively by drag-and-drop with the mouse cursor
    """
    @property
    def dnd_widgets(self):
        return self._dnd_widgets
    
    def dnd_put(self,
                widgets:list,
                orient:Optional[str]=None,
                sticky:Optional[str]=None,
                expand:Union[Tuple[bool], List[bool], bool]=False,
                padding:Union[Tuple[int], List[int], int]=0,
                ipadding:Union[Tuple[int], List[int], int]=0):
        """Use this function to put widgets `widgets` onto the container 
        (canvas).
        
        `widgets` must be a list of widgets or a list of lists of 
        widgets. If `widgets` is a list of lists of widgets (a 2-D array of 
        widgets), the result layout will be in the same order as in the 
        `widgets`. Otherwise, one can specify the orientation by `orient`, e.g., 
        'vertical' (default) or 'horizontal'.
        
        Other arguments work like the arguments in the grid layout manager.
        """
        assert isinstance(widgets, list), widgets
        assert isinstance(expand, (tuple, list, bool)), expand
        assert isinstance(padding, (tuple, list, int)), padding
        assert isinstance(ipadding, (tuple, list, int)), ipadding
        assert orient in ['horizontal', 'vertical', None], orient
        
        if not isinstance(widgets[0], list):
            if orient == 'horizontal':
                widgets = [widgets]
            else:  # vertical
                widgets = [ [w] for w in widgets ]
        nrows, ncols = len(widgets), len(widgets[0])
        
        two_dimensional = list()
        for arg in [expand, padding, ipadding]:
            if isinstance(arg, (tuple, list)):
                arg = tuple(arg)
                if len(arg) == 1:
                    arg *= 2
                else:
                    assert len(arg) == 2, arg
            else:
                arg = (arg,) * 2
            two_dimensional.append(arg)
        expand, (padx, pady), (ipadx, ipady) = two_dimensional
        params = {"expand": expand,
                  "padding": (padx, pady),
                  "ipadding": (ipadx, ipady),
                  "grid_size": (nrows, ncols),
                  "sticky": sticky}
        
        # Calculate canvas size and place widgets onto the canvas
        widgets_flat = [ w for row in widgets for w in row ]
        widths, heights = list(), list()
        for widget in widgets_flat:
            widget._id = self.create_window(0, 0, anchor='nw', window=widget)
            widths.append(widget.winfo_reqwidth())
            heights.append(widget.winfo_reqheight())
        width_cell, height_cell = max(widths), max(heights)
        width_canvas = ncols*width_cell + 2*padx + 2*(ncols - 1)*ipadx
        height_canvas = nrows*height_cell + 2*pady + 2*(nrows - 1)*ipady
        self.configure(width=width_canvas, height=height_canvas)
        params.update({
            "width_cell": width_cell, "height_cell": height_cell,
            "width_canvas": width_canvas, "height_canvas": height_canvas
        })
        
        # Calculate place_info and update widgets' position and size
        widget_ids = list()
        place_info_list = list()
        i = 0
        for r, row in enumerate(widgets):
            for c, widget in enumerate(row):
                info = self._calculate_place_info(
                    r, c, widths[i], heights[i], **params)
                self._put(widget, info, width_canvas, height_canvas)
                self._resize(widget, info, width_canvas, height_canvas)
                widget_ids.append(widget._id)
                place_info_list.append(info)
                i += 1
        
        # Setup widget's dnd functions
        for widget in widgets_flat:
            widget.bind('<ButtonPress-1>', self._get_dnd_widget_start(widget))
            widget.dnd_widget_accept = self._get_dnd_widget_accept(widget)
            widget.dnd_widget_motion = self._get_dnd_widget_motion(widget)
            widget.dnd_widget_enter = self._get_dnd_widget_enter(widget)
            widget.dnd_widget_leave = self._get_dnd_widget_leave(widget)
            widget.dnd_end = self._get_dnd_widget_end(widget)
        
        self.bind('<Configure>', self._update_widgets)
        self._dnd_place_info = place_info_list
        self._dnd_widgets = widgets_flat
        self._dnd_ids = dict(zip(widget_ids, widgets_flat))
        self._dnd_grid_size = params["grid_size"]
        self._dnd_handler = None
        self._dnd = dict()
    
    def _calculate_place_info(self,
                              r,
                              c,
                              width,
                              height,
                              width_canvas,
                              height_canvas,
                              width_cell,
                              height_cell,
                              grid_size,
                              sticky,
                              expand,
                              padding,
                              ipadding):
        """This function simulates the grid layout manager by calculating the 
        position and size of each widget in the cell
        """
        def _calc(xy, dim, dimension, L, dimension_cell, n, N,
                  _sticky, expand, pad, ipad):
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
                    return 's' if xy == 'x' else 'e'  # higher bound
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
        
        anchor_x = _calc('x', 'width', width, width_canvas, width_cell,
            c, ncols, sticky_x, expand[0], padding[0], ipadding[0])
        anchor_y = _calc('y', 'height', height, height_canvas, height_cell,
            r, nrows, sticky_y, expand[1], padding[1], ipadding[1])
        
        place_info["anchor"] = (anchor_x + anchor_y) or 'center'
        
        return place_info
    
    def _update_widgets(self, event=None):
        self.update_idletasks()
        width_canvas, height_canvas = self.winfo_width(), self.winfo_height()
        for widget, info in zip(self._dnd_widgets, self._dnd_place_info):
            self._put(widget, info, width_canvas, height_canvas)
            self._resize(widget, info, width_canvas, height_canvas)
    
    def _put(self, widget, info=None, width_canvas=None, height_canvas=None):
        info = info or self._dnd_place_info[self._dnd_widgets.index(widget)]
        width_canvas = width_canvas or self._dnd["width_canvas"]
        height_canvas = height_canvas or self._dnd["height_canvas"]
        
        x = max(int(info["relx"] * width_canvas + info["x"]), 0)
        y = max(int(info["rely"] * height_canvas + info["y"]), 0)
        
        self.itemconfigure(widget._id, anchor=info["anchor"])
        self.coords(widget._id, x, y)
    
    def _resize(self, widget, info, width_canvas=None, height_canvas=None):
        info = info or self._dnd_place_info[self._dnd_widgets.index(widget)]
        width_canvas = width_canvas or self._dnd["width_canvas"]
        height_canvas = height_canvas or self._dnd["height_canvas"]
        
        w = max(int(info["relwidth"] * width_canvas + info["width"]), 0)
        h = max(int(info["relheight"] * height_canvas + info["height"]), 0)
        
        self.itemconfigure(widget._id, width=w, height=h)
    
    def dnd_accept(self, source, event):
        return self
    
    def dnd_enter(self, source, event):
        pass
    
    def dnd_motion(self, source, event):
        def _find_target(widget):
            dummy = lambda *args: None
            target = getattr(widget, 'dnd_widget_accept', dummy)(source, event)
            if target is not None:
                return target
            
            for w in widget.winfo_children():
                target = _find_target(w)
                if target is not None:
                    return target
        #
        # Widget motion
        self._get_dnd_widget_motion()(source, event)
        
        # Find the target under mouse cursor
        x = event.x_root - self._dnd["canvas_x"]
        y = event.y_root - self._dnd["canvas_y"]
        new_target = None
        for widget_id in self.find_overlapping(x, y, x, y):
            new_target = _find_target(self._dnd_ids[widget_id])
            if new_target is not None:
                break
        
        old_target = self._dnd["target"]
        if new_target is old_target:  # in the same target
            return
        
        # Left `old_target` and entered `new_target`
        self._dnd["target"] = None
        if old_target:
            old_target.dnd_widget_leave(source, event)
        if new_target:
            new_target.dnd_widget_enter(source, event)
        self._dnd["target"] = new_target
    
    def dnd_leave(self, source, event):
        pass
    
    def dnd_commit(self, source, event):
        self.dnd_leave(source, event)
        self._put(source)
    
    def _get_dnd_widget_start(self, source):
        def _init_dnd(event):
            self._dnd_handler = tk.dnd.dnd_start(source, event)
            if not self._dnd_handler:
                return
            
            self._dnd = {
                "event": None,
                "source": source,
                "target": None,
                "mouse_x": event.x_root,
                "mouse_y": event.y_root,
                "canvas_x": self.winfo_rootx(),
                "canvas_y": self.winfo_rooty(),
                "width_canvas": self.winfo_width(),
                "height_canvas": self.winfo_height()
            }
            source.lift()
            source.focus_set()
        #
        return _init_dnd
    
    def _get_dnd_widget_accept(self, target):
        def _get_target(source, event):
            if source is not target:
                return target
        #
        return _get_target
    
    def _get_dnd_widget_enter(self, target):
        def _exchange_order_indices(source, event):
            # Exchange the order indices of `source` and `target`
            neworder_src = oldorder_tar = self.dnd_widgets.index(target)
            neworder_tar = oldorder_src = self.dnd_widgets.index(source)
            
            for widget, neworder in [(source, neworder_src),
                                     (target, neworder_tar)]:
                self.dnd_widgets.pop(neworder)
                self.dnd_widgets.insert(neworder, widget)
            
            # Exchange the dimensions of `source` and `target`
            newinfo_src = self._dnd_place_info[oldorder_tar]
            newinfo_tar = self._dnd_place_info[oldorder_src]
            oldinfo_src, oldinfo_tar = newinfo_tar.copy(), newinfo_src.copy()
            
            for oldinfo, newinfo in [(oldinfo_src, newinfo_src),
                                     (oldinfo_tar, newinfo_tar)]:
                newinfo.update({ key: oldinfo[key] for key in
                    ["width", "height", "relwidth", "relheight"] })
            
            # Update position of `target`
            self._put(target)
        #
        return _exchange_order_indices
    
    def _get_dnd_widget_motion(self, widget=None):
        def _move_source(source, event):
            old_event, self._dnd["event"] = self._dnd.pop("event"), event
            if old_event is event:  # `source` has been moved before
                return
            
            # Move the widget according to dragging distance
            self.move(source._id,
                      event.x_root - self._dnd.pop("mouse_x"),
                      event.y_root - self._dnd.pop("mouse_y"))
            self._dnd.update(mouse_x=event.x_root, mouse_y=event.y_root)
        #
        return _move_source
    
    def _get_dnd_widget_leave(self, target):
        def _none(source, event): pass
        return _none
    
    def _get_dnd_widget_end(self, widget):
        def _unset_focus(target, event):
            self.winfo_toplevel().focus_set()
        #
        return _unset_focus


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    import random
    
    root = ttk.Window(title='Harmonic Analyzer', themename='cyborg')
    
    container = OrderedContainer(root)
    container.pack(fill='both', expand=1)
    buttons = list()
    for r in range(6):
        buttons.append(list())
        for c in range(3):
            dash = '----' * random.randint(1, 5)
            button = ttk.Button(container,
                                text=f'|<{dash} ({r}, {c}) {dash}>|',
                                takefocus=True,
                                bootstyle='outline')
            buttons[-1].append(button)
    
    container.dnd_put(buttons,
                      sticky='nse',
                      expand=(False, True),
                      padding=10,
                      ipadding=6)
    
    root.place_window_center()
    root.mainloop()

