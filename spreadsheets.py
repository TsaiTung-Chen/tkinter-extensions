#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import sys
import copy
import tkinter as tk
import tkinter.font
from contextlib import contextmanager
from typing import Union, Optional, List, Tuple, Dict, Callable, Literal

import numpy as np
import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.dialogs import dialogs, colorchooser

from .dnd import ButtonTriggerOrderlyContainer
from .scrolled_widgets import AutoHiddenScrollbar, ScrolledFrame

COMMAND = 'Mod1' if sys.platform == 'darwin' else 'Control'
OPTION = 'Mod2' if sys.platform == 'darwin' else 'Alt'
CONTROL = 'Control'
SHIFT = 'Shift'
LOCK = 'Lock'
MODIFIERS = {COMMAND, OPTION, CONTROL, SHIFT}

MODIFIER_MASKS = {
    "Shift": int('0b1', base=2),
    "Lock": int('0b10', base=2),
    "Control": int('0b100', base=2),
    "Mod1": int('0b1000', 2),  # command (Mac)
    "Mod2": int('0b10000', base=2),   # option (Mac)
    "Mod3": int('0b100000', base=2),
    "Mod4": int('0b1000000', base=2),
    "Mod5": int('0b10000000', base=2),
    "Button1": int('0b100000000', base=2),
    "Button2": int('0b1000000000', base=2),
    "Button3": int('0b10000000000', base=2),
    "Button4": int('0b100000000000', base=2),
    "Button5": int('0b1000000000000', base=2)
}
# =============================================================================
# ---- Functions
# =============================================================================
def get_modifiers(state:int):
    modifiers = list()
    for mod, mask in MODIFIER_MASKS.items():
        if state & mask:
            modifiers.append(mod)
    
    return modifiers


def infinite_loop(item):
    while True:
        yield item


# =============================================================================
# ---- Classes
# =============================================================================
class _DialogPositioning:
    def show(self, position:Union[tuple, list, Callable, None]=None):
        # Edit: accept a positioning function argument
        
        self._result = None
        self.build()
        
        if position is None:
            self._locate()
        else:
            try:
                #EDIT
                if callable(position):
                    position(self._toplevel)
                else:
                    x, y = position
                    self._toplevel.geometry(f'+{x}+{y}')
            except:
                self._locate()
        
        self._toplevel.deiconify()
        if self._alert:
            self._toplevel.bell()
        
        if self._initial_focus:
            self._initial_focus.focus_force()
        
        self._toplevel.grab_set()
        self._toplevel.wait_window()


class FontDialog(_DialogPositioning, dialogs.FontDialog):
    from ttkbootstrap.localization import MessageCatalog as _MessageCatalog
    
    def __init__(self,
                 title="Font Selector",
                 parent=None,
                 default:Optional[tk.font.Font]=None):
        # Edit: set the default font as `default`
        
        #EDIT
        assert isinstance(default, (tk.font.Font, type(None))), default
        
        title = self._MessageCatalog.translate(title)
        super().__init__(parent=parent, title=title)
        
        #EDIT
        if default is None:
            default = tk.font.nametofont('TkDefaultFont')
        
        self._style = ttk.Style()
        self._default:tk.font.Font = default  #EDIT
        self._actual = self._default.actual()
        self._size = ttk.Variable(value=self._actual["size"])
        self._family = ttk.Variable(value=self._actual["family"])
        self._slant = ttk.Variable(value=self._actual["slant"])
        self._weight = ttk.Variable(value=self._actual["weight"])
        self._overstrike = ttk.Variable(value=self._actual["overstrike"])
        self._underline = ttk.Variable(value=self._actual["underline"])
        self._preview_font = tk.font.Font()
        self._slant.trace_add("write", self._update_font_preview)
        self._weight.trace_add("write", self._update_font_preview)
        self._overstrike.trace_add("write", self._update_font_preview)
        self._underline.trace_add("write", self._update_font_preview)
        
        _headingfont = tk.font.nametofont("TkHeadingFont")
        _headingfont.configure(weight="bold")
        
        self._update_font_preview()
        self._families = set([self._family.get()])
        for f in tk.font.families():
            if all([f, not f.startswith("@"), "emoji" not in f.lower()]):
                self._families.add(f)
    
    def create_body(self, master):
        # Edit: use natural window size
        
        #EDIT: width = utility.scale_size(master, 600)
        #EDIT: height = utility.scale_size(master, 500)
        #EDIT: self._toplevel.geometry(f"{width}x{height}")
        
        family_size_frame = ttk.Frame(master, padding=10)
        family_size_frame.pack(fill='x', anchor='n')
        self._initial_focus = self._font_families_selector(family_size_frame)
        self._font_size_selector(family_size_frame)
        self._font_options_selectors(master, padding=10)
        self._font_preview(master, padding=10)
    
    def _font_options_selectors(self, master, padding: int):
        # Edit: don't change the values of the tk variables
        
        container = ttk.Frame(master, padding=padding)
        container.pack(fill='x', padx=2, pady=2, anchor='n')

        weight_lframe = ttk.Labelframe(
            container, text=self._MessageCatalog.translate("Weight"), padding=5
        )
        weight_lframe.pack(side='left', fill='x', expand=1)
        opt_normal = ttk.Radiobutton(
            master=weight_lframe,
            text=self._MessageCatalog.translate("normal"),
            value="normal",
            variable=self._weight,
        )
        #EDIT: opt_normal.invoke()
        opt_normal.pack(side='left', padx=5, pady=5)
        opt_bold = ttk.Radiobutton(
            master=weight_lframe,
            text=self._MessageCatalog.translate("bold"),
            value="bold",
            variable=self._weight,
        )
        opt_bold.pack(side='left', padx=5, pady=5)

        slant_lframe = ttk.Labelframe(
            container, text=self._MessageCatalog.translate("Slant"), padding=5
        )
        slant_lframe.pack(side='left', fill='x', padx=10, expand=1)
        opt_roman = ttk.Radiobutton(
            master=slant_lframe,
            text=self._MessageCatalog.translate("roman"),
            value="roman",
            variable=self._slant,
        )
        #EDIT: opt_roman.invoke()
        opt_roman.pack(side='left', padx=5, pady=5)
        opt_italic = ttk.Radiobutton(
            master=slant_lframe,
            text=self._MessageCatalog.translate("italic"),
            value="italic",
            variable=self._slant,
        )
        opt_italic.pack(side='left', padx=5, pady=5)

        effects_lframe = ttk.Labelframe(
            container, text=self._MessageCatalog.translate("Effects"), padding=5
        )
        effects_lframe.pack(side='left', padx=(2, 0), fill='x', expand=1)
        opt_underline = ttk.Checkbutton(
            master=effects_lframe,
            text=self._MessageCatalog.translate("underline"),
            variable=self._underline,
        )
        opt_underline.pack(side='left', padx=5, pady=5)
        opt_overstrike = ttk.Checkbutton(
            master=effects_lframe,
            text=self._MessageCatalog.translate("overstrike"),
            variable=self._overstrike,
        )
        opt_overstrike.pack(side='left', padx=5, pady=5)
    
    def _font_preview(self, master, padding:int):
        # Edit: don't turn off `pack_propagate` and set a small width
        
        container = ttk.Frame(master, padding=padding)
        container.pack(fill='both', expand=1, anchor='n')

        header = ttk.Label(
            container,
            text=self._MessageCatalog.translate('Preview'),
            font='TkHeadingFont',
        )
        header.pack(fill='x', pady=2, anchor='n')

        content = self._MessageCatalog.translate(
            'The quick brown fox jumps over the lazy dog.'
        )
        self._preview_text = ttk.Text(
            master=container,
            height=3,
            width=1,   #EDIT: prevent the width from becoming too large
            font=self._preview_font,
            highlightbackground=self._style.colors.primary
        )
        self._preview_text.insert('end', content)
        self._preview_text.pack(fill='both', expand=1)
        #EDIT: container.pack_propagate(False)
    
    def _update_font_preview(self, *_):
        # Edit: configure the weight of text and update `self._result` when 
        # submitted
        
        self._preview_font.config(
            family=self._family.get(),
            size=self._size.get(),
            slant=self._slant.get(),
            weight=self._weight.get(),   #EDIT
            overstrike=self._overstrike.get(),
            underline=self._underline.get()
        )
        try:
            self._preview_text.configure(font=self._preview_font)
        except:
            pass
        #EDIT: self._result = self._preview_font
    
    def _on_submit(self):
        # Edit: update `self._result` when submitted
        
        self._result = self._preview_font  #EDIT
        return super()._on_submit()


class ColorChooserDialog(_DialogPositioning, colorchooser.ColorChooserDialog):
    pass


class History:
    @property
    def step(self) -> int:
        return self._step
    
    @property
    def backable(self) -> bool:
        return self.step > 0
    
    @property
    def forwardable(self) -> bool:
        return self.step < len(self._stack["forward"])
    
    def __init__(self):
        self._step = 0
        self._sequence:Union[Dict[str, List[Callable]], None] = None
        self._stack = {"forward": list(), "backward": list()}
    
    def add(self, forward:Callable, backward:Callable):
        assert callable(forward) and callable(backward), (forward, backward)
        
        if self._sequence is None:
            self._stack.update(
                forward=self._stack["forward"][:self.step] + [forward],
                backward=self._stack["backward"][:self.step] + [backward]
            )
            self._step += 1
        else:
            self._sequence["forward"].append(forward)
            self._sequence["backward"].append(backward)
    
    @contextmanager
    def add_sequence(self):
        yield self.start_sequence()
        self.stop_sequence()
    
    def start_sequence(self) -> dict:
        assert self._sequence is None, self._sequence
        self._sequence = {"forward": list(), "backward": list()}
        return self._sequence
    
    def stop_sequence(self):
        assert isinstance(self._sequence, dict), self._sequence
        sequences = self._sequence
        self._sequence = None
        self.add(
            forward=lambda: [ func() for func in sequences["forward"] ],
            backward=lambda: [ func() for func in sequences["backward"][::-1] ]
        )
    
    def pop(self) -> Dict[str, List[Callable]]:
        assert self.step > 0, self.step
        
        self._step -= 1
        trailing = {"forward": self._stack["forward"][self.step:],
                    "backward": self._stack["backward"][self.step:]}
        self._stack.update(forward=self._stack["forward"][:self.step],
                            backward=self._stack["backward"][:self.step])
        return trailing
    
    def reset(self):
        self.__init__()
    
    def back(self):
        assert self.step > 0, self.step
        self._step -= 1
        self._stack["backward"][self.step]()
        return self.step
    
    def forward(self):
        forward_stack = self._stack["forward"]
        assert self.step < len(forward_stack), (self.step, self._stack)
        forward_stack[self.step]()
        self._step += 1
        return self.step


class Sheet(ttk.Frame):
    _valid_header_states = ('normal', 'hover', 'selected')
    _valid_cell_states = ('normal', 'readonly')
    
    @property
    def RightClick(self) -> str:
        if self._windowingsystem == 'aqua':  # macOS
            return '<ButtonPress-2>'
        return '<ButtonPress-3>'
    
    @property
    def MouseScroll(self) -> List[str]:
        if self._windowingsystem == 'x11':  # Linux
            return ['<ButtonPress-4>', '<ButtonPress-5>']
        return ['<MouseWheel>']

    @property
    def canvas(self) -> tk.Canvas:  # cell canvas
        return self._canvas
    
    @property
    def cornercanvas(self) -> tk.Canvas:
        return self._cornercanvas
    
    @property
    def rowcanvas(self) -> tk.Canvas:
        return self._rowcanvas
    
    @property
    def colcanvas(self) -> tk.Canvas:
        return self._colcanvas
    
    @property
    def hbar(self) -> AutoHiddenScrollbar:
        return self._hbar
    
    @property
    def vbar(self) -> AutoHiddenScrollbar:
        return self._vbar
    
    @property
    def values(self) -> pd.DataFrame:
        return self._values
    
    def __init__(self,
                 master,
                 shape:Union[Tuple[int], List[int]]=(10, 10),
                 cell_width:int=80,
                 cell_height:int=25,
                 min_width:int=10,
                 min_height:int=10,
                 get_style:Optional[Callable]=None,
                 autohide_scrollbar:bool=True,
                 mousewheel_sensitivity=1.,
                 bootstyle_scrollbar='round',
                 _reset:bool=False,
                 **kw):
        self._init_configs = {
            k: v for k, v in locals().items()
            if k not in ("self", "kw", "_reset", "__class__")
        }
        self._init_configs.update(kw)
        
        if not _reset:
            super().__init__(master)
        assert len(shape) == 2 and all( s > 0 for s in shape ), shape
        assert cell_width >= 1 and cell_height >= 1, (cell_width, cell_height)
        assert get_style is None or callable(get_style), get_style
        
        # Stacking order: CornerCanvas > ColCanvas > RowCanvas > CoverFrame
        # > SelectionFrame > Entry > CellCanvas
        top_left = {"row": 0, "column": 0}
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self._canvas = canvas = tk.Canvas(self, **kw)  # cell canvas
        self._canvas.grid(**top_left, sticky='nesw', rowspan=2, columnspan=2)
        self._rowcanvas = rowcanvas = tk.Canvas(self, **kw)
        self._rowcanvas.grid(**top_left, rowspan=2, sticky='nesw')
        self._colcanvas = colcanvas = tk.Canvas(self, **kw)
        self._colcanvas.grid(**top_left, columnspan=2, sticky='nesw')
        self._cornercanvas = tk.Canvas(self, **kw)
        self._cornercanvas.grid(**top_left, sticky='nesw')
        
        self._hbar = AutoHiddenScrollbar(
            master=self,
            autohide=autohide_scrollbar,
            command=self.xview,
            bootstyle=bootstyle_scrollbar,
            orient='horizontal',
        )
        self._hbar.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(1, 0))
        self._vbar = AutoHiddenScrollbar(
            master=self,
            autohide=autohide_scrollbar,
            command=self.yview,
            bootstyle=bootstyle_scrollbar,
            orient='vertical',
        )
        self._vbar.grid(row=0, column=2, rowspan=2, sticky='ns', padx=(1, 0))
        self._cover = ttk.Frame(self)  # covers entry and selection frames
        self._cover.grid(row=2, column=2, sticky='nesw')  # right bottom corner
        self._cover.lift(canvas)
        self._mousewheel_sensitivity = mousewheel_sensitivity
        
        # Create an invisible background which makes this sheet become the focus 
        # if being clicked
        canvas.create_rectangle(1, 1, 1, 1, width=0, tag='invisible-bg')
        canvas.tag_bind('invisible-bg', '<Button-1>', self._be_focus)
        rowcanvas.create_rectangle(1, 1, 1, 1, width=0, tag='invisible-bg')
        rowcanvas.tag_bind('invisible-bg', '<Button-1>', self._be_focus)
        colcanvas.create_rectangle(1, 1, 1, 1, width=0, tag='invisible-bg')
        colcanvas.tag_bind('invisible-bg', '<Button-1>', self._be_focus)
        
        self._get_style = get_style
        self._default_styles = self._update_default_styles()
        self._default_cell_shape = shape
        self._default_cell_sizes = (cell_height, cell_width) = (
            int(cell_height), int(cell_width)
        )
        self._min_sizes = [int(min_height), int(min_width)]
        
        self._values = pd.DataFrame(np.full(shape, '', dtype=object))
        self._cell_sizes = [
            np.full(shape[0] + 1, cell_height),
            np.full(shape[1] + 1, cell_width)
        ]
        self._cell_styles = np.array(
            [ [ dict() for c in range(shape[1]) ] for r in range(shape[0]) ],
            dtype=object
        )
        
        self.update_idletasks()
        self._canvas_size = (canvas.winfo_width(), canvas.winfo_height())
        self._content_size = self._update_content_size()
        self._pview = [(0, 0), (0, 0)]  # x view and y view in pixel
        self._visible_rcs, self._gy2s_gx2s = self._update_visible_rcs_gp2s()
        self._focus = None
        self._hover = None
        self._resize_start = None
        self._history = History()
        
        self._rightclick_menu = tk.Menu(self, tearoff=0)
        self._focus_var = tk.StringVar(self)
        self._entry = entry = tk.Entry(self, textvariable=self._focus_var)
        entry.place(x=0, y=0)
        entry.lower()
        entry.bind('<KeyPress>', self._on_key_press_entry)
        
        self._selframes = selframes = [ tk.Frame(self) for i in range(4) ]
        for frame in selframes:
            frame.place(x=0, y=0)
            frame.lift(canvas)
            frame.bind('<ButtonPress-1>', self._on_leftclick_press_selframe)
            frame.bind('<B1-Motion>', self._on_leftclick_motion_selframe)
            frame.bind(
                '<Double-ButtonPress-1>',
                self._on_leftclick_double_press_selframe
            )
        
        self._selection_rcs:Tuple[int] = (-1, -1, -1, -1)
        self._selection_rcs:Tuple[int] = self._select_cells(0, 0, 0, 0)
        
        self.bind('<<ThemeChanged>>', self._on_theme_changed)
        self.bind('<KeyPress>', self._on_key_press)
        self.bind('<<SelectAll>>', self._on_select_all)
        self.bind('<<Copy>>', self._on_copy)
        self.bind('<<Paste>>', self._on_paste)
        canvas.bind('<Configure>', self._on_configure_canvas)
        for widget in [canvas, rowcanvas, colcanvas, entry, *selframes]:
            widget.configure(takefocus=0)
            for scrollseq in self.MouseScroll:
                widget.bind(scrollseq, self._on_mousewheel_scroll)
        
        self.xview_scroll(0, 'units')
        self.yview_scroll(0, 'units')
        self.focus_set()
    
    def _be_focus(self, *_, **__):
        self._focus_out_cell()
        self.focus_set()
    
    def _on_theme_changed(self, event=None):
        self._update_default_styles()
        self._canvases_delete('temp')
        self.redraw()
    
    def _on_configure_canvas(self, event):
        self._canvas_size = canvas_size = (event.width, event.height)
        self.canvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.rowcanvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.colcanvas.coords('invisible-bg', 0, 0, *canvas_size)
        self.xview_scroll(0, 'units')
        self.yview_scroll(0, 'units')
    
    def _on_key_press_entry(self, event):
        keysym = event.keysym
        modifiers = get_modifiers(event.state)
        
        if (keysym in ('z', 'Z')) and (COMMAND in modifiers):
            self.undo()
            return self._selection_rcs
        
        elif (keysym in ('y', 'Y')) and (COMMAND in modifiers):
            self.redo()
            return self._selection_rcs
        
        elif keysym in ('Return', 'Tab'):
            if keysym == 'Return':
                direction = 'up' if SHIFT in modifiers else 'down'
            else:
                direction = 'left' if SHIFT in modifiers else 'right'
            return self._move_selections(direction)
        
        elif keysym == 'Escape':
            self._focus_out_cell(discard=True)
            return True
    
    def _on_key_press(self, event):
        keysym, char = event.keysym, event.char
        modifiers = get_modifiers(event.state)
        modifiers_wo_lock = [ mod for mod in modifiers if mod != LOCK ]
        
        if self._on_key_press_entry(event):
            return
        
        elif keysym in ('Up', 'Down', 'Left', 'Right'):
            direction = keysym.lower()
            area = 'paragraph' if COMMAND in modifiers else None
            expand = SHIFT in modifiers
            return self._move_selections(direction, area=area, expand=expand)
        
        elif keysym in ('Home', 'End', 'Prior', 'Next'):
            direction = {
                "Home": 'left',
                "End": 'right',
                "Prior": 'up',
                "Next": 'down'
            }[keysym]
            expand = SHIFT in modifiers
            return self._move_selections(direction, area='all', expand=expand)
        
        elif keysym == 'Delete':  # delete all characters in the selected cells
            return self._selection_erase_values(undo=True)
        
        elif (MODIFIERS.isdisjoint(modifiers) and keysym == 'BackSpace') or (
                modifiers_wo_lock in ([], [SHIFT]) and char):  # normal typing
            self._focus_in_cell()
            self._entry.delete(0, 'end')
            self._entry.insert('end', char)
    
    def _on_mousewheel_scroll(self, event):
        """Callback for when the mouse wheel is scrolled.
        Modified from: `ttkbootstrap.scrolled.ScrolledFrame._on_mousewheel`
        """
        if event.num == 4:  # Linux
            delta = 10.
        elif event.num == 5:  # Linux
            delta = -10.
        elif self._windowingsystem == "win32":  # Windows
            delta = event.delta / 120.
        else:  # Mac
            delta = event.delta
        number = -round(delta * self._mousewheel_sensitivity)
        
        if event.state & MODIFIER_MASKS["Shift"]:
            self.xview_scroll(number, 'units')
        else:
            self.yview_scroll(number, 'units')
    
    def _on_select_all(self, event=None):
        self._select_cells()
    
    def _on_copy(self, event=None):
        self._selection_copy_values()
    
    def _on_paste(self, event=None):
        self._selection_paste_values()
    
    def _update_default_styles(self):
        if self._get_style:
            self._default_styles = dict(self._get_style())
            return self._default_styles
        
        style = ttk.Style.get_instance()
        
        header = ttk.Checkbutton(self, bootstyle='primary-outline-toolbutton')
        header_style = header["style"]
        
        cell = ttk.Entry(self, bootstyle='secondary')
        cell_style = cell["style"]
        
        selection = ttk.Frame(self, bootstyle='primary')
        selection_style = selection["style"]
        
        self._default_styles = {
            "header": {
                "background": {
                    "normal": style.lookup(header_style, "background"),
                    "hover": style.lookup(
                        header_style, "background", ('hover', '!disabled')),
                    "selected": style.lookup(
                        header_style, "background", ('selected', '!disabled'))
                },
                "foreground": {
                    "normal": style.lookup(header_style, "foreground"),
                    "hover": style.lookup(
                        header_style, "foreground", ('hover', '!disabled')),
                    "selected": style.lookup(
                        header_style, "foreground", ('selected', '!disabled'))
                },
                "bordercolor": {
                    "normal": style.lookup(header_style, "bordercolor"),
                    "hover": style.lookup(
                        header_style, "bordercolor", ('hover', '!disabled')),
                    "selected": style.lookup(
                        header_style, "bordercolor", ('selected', '!disabled'))
                },
                "font": 'TkDefaultFont',
                "cursor": {
                    "default": 'arrow',
                    "hhandle": 'sb_v_double_arrow',
                    "vhandle": 'sb_h_double_arrow'
                }
            },
            "cell": {
                "background": {
                    "normal": style.lookup(cell_style, "fieldbackground"),
                    "readonly": style.lookup(cell_style, "foreground")
                },
                "foreground": {
                    "normal": style.lookup(cell_style, "foreground"),
                    "readonly": style.lookup(cell_style, "fieldbackground")
                },
                "bordercolor": {
                    "normal": style.lookup(cell_style, "bordercolor"),
                    "focus": style.lookup(selection_style, "background")
                },
                "font": 'TkDefaultFont',
                "alignx": 'w',   # w, e, or center
                "aligny": 'n',   # n, s, or center
                "padding": (3, 2)  # (padx, pady)
            },
            "selection": {
                "color": style.lookup(selection_style, "background"),
                "width": 2
            }
        }
        
        header.destroy()
        cell.destroy()
        selection.destroy()
        
        return self._default_styles
    
    def _update_content_size(self):
        self._content_size = tuple(
            np.sum(self._cell_sizes[axis]) + 1 for axis in range(2)
        )[::-1]
        return self._content_size
    
    def _get_center_position(self) -> Tuple[int]:
        self.update_idletasks()
        width, height = self.winfo_width(), self.winfo_height()
        x_root, y_root = self.winfo_rootx(), self.winfo_rooty()
        return (x_root + width//2, y_root + height//2)
    
    def _center_window(self, toplevel:tk.BaseWidget):
        x_center, y_center = self._get_center_position()
        width, height = toplevel.winfo_reqwidth(), toplevel.winfo_reqheight()
        x, y = (x_center - width//2, y_center - height//2)
        tk.Wm.wm_geometry(toplevel, f'+{x}+{y}')
    
    def __view(self, axis:int, *args):
        """Update the view of the canvas
        """
        assert axis in (0, 1), axis
        
        if not args:
            f1, f2 = self.__to_fraction(axis, *self._pview[axis])
            return max(f1, 0.), min(f2, 1.)
        
        action, args = args[0], args[1:]
        if action == 'moveto':
            return self.__view_moveto(axis, float(args[0]))
        elif action == 'scroll':
            return self.__view_scroll(axis, int(args[0]), args[1])
        raise ValueError("The first argument must be 'moveto' or 'scroll' but "
                         f"got: {repr(args[0])}")
    
    xview = lambda self, *args: self.__view(0, *args)
    yview = lambda self, *args: self.__view(1, *args)
    
    def __view_moveto(self, axis:int, fraction:float):
        """Move the view of the canvas
        """
        # Check the start and stop locations are valid
        start = self.__to_pixel(axis, fraction)
        
        # Update the canvas and scrollbar
        self.__update_content_and_scrollbar(axis, start)
    
    xview_moveto = lambda self, *args, **kw: self.__view_moveto(0, *args, **kw)
    yview_moveto = lambda self, *args, **kw: self.__view_moveto(1, *args, **kw)
    
    def __view_scroll(self, axis:int, number:int, what:str):
        """Scroll the view of the canvas
        """
        magnification = {"units": 10., "pages": 50.}[what]
        # Check the start and stop locations are valid
        start, _ = self._pview[axis]
        start += round(number * magnification)
        
        # Update widgets
        self.__update_content_and_scrollbar(axis, start)
    
    xview_scroll = lambda self, *args, **kw: self.__view_scroll(0, *args, **kw)
    yview_scroll = lambda self, *args, **kw: self.__view_scroll(1, *args, **kw)
    
    def __to_fraction(self, axis:int, *pixels) -> Tuple[float]:
        assert axis in (0, 1), axis
        complete = self._content_size[axis]
        return tuple( pixel / complete for pixel in pixels )
    
    def __to_pixel(self, axis:int, *fractions) -> Union[Tuple[int], int]:
        assert axis in (0, 1), axis
        complete = self._content_size[axis]
        numbers = tuple( round(fraction * complete) for fraction in fractions )
        if len(numbers) == 1:
            return numbers[0]
        return numbers
    
    def __confine_region(self, axis:int, new, is_start:bool=True):
        complete = self._content_size[axis]
        showing = min(self._canvas_size[axis], complete)
        
        if is_start:
            start, stop = (new, new + showing)
        else:
            start, stop = (new - showing, new)
        
        if start < 0:
            start = 0
            stop = showing
        elif stop > complete:
            stop = complete
            start = stop - showing
        
        return start, stop
    
    def __update_content_and_scrollbar(self,
                                       axis:int,
                                       new:int,
                                       is_start:bool=True):
        new_start, new_stop = self.__confine_region(axis, new, is_start)
        prev_start, prev_stop = self._pview[axis]
        prev_r1, prev_c1, prev_r2, prev_c2 = self._visible_rcs
        self._pview[axis] = (new_start, new_stop)
        (new_r1, new_c1, new_r2, new_c2), _ = self._update_visible_rcs_gp2s()
        
        # Move xscrollable or yscrollable items
        delta_canvas = prev_start - new_start  # -delta_view
        if axis == 0:
            key = "col"
            prev_i1, prev_i2, new_i1, new_i2 = (prev_c1, prev_c2, new_c1, new_c2)
            header_canvas = self.colcanvas
            
            self.canvas.move('xscroll', delta_canvas, 0)
            header_canvas.move('all', delta_canvas, 0)
            for widget in [self._entry, *self._selframes]:
                widget.place(x=int(widget.place_info()["x"]) + delta_canvas)
        else:
            key = "row"
            prev_i1, prev_i2, new_i1, new_i2 = (prev_r1, prev_r2, new_r1, new_r2)
            header_canvas = self.rowcanvas
            
            self.canvas.move('yscroll', 0, delta_canvas)
            header_canvas.move('all', 0, delta_canvas)
            for widget in [self._entry, *self._selframes]:
                widget.place(y=int(widget.place_info()["y"]) + delta_canvas)
        
        # Delete out-of-view items
        idc_out = set(range(prev_i1, prev_i2+1)) - set(range(new_i1, new_i2+1))
        tags_out = [ self._make_tag(key, row=i, col=i) for i in idc_out ]
        for tag in tags_out:
            for canvas in (self.canvas, header_canvas):
                canvas.delete(tag)
        
        # Draw new items
        self.redraw(
            update_visible_rcs=False,
            skip_exist=True,
            trace=False
        )
        
        # Update x or y scrollbar
        first, last = self.__to_fraction(axis, new_start, new_stop)
        (self.hbar, self.vbar)[axis].set(first, last)
    
    def _make_tags(self,
                   type_=None,
                   subtype=None,
                   row=None,
                   col=None,
                   others=('temp',),
                   *,
                   withkey:bool=True,
                   to_tuple:bool=False) -> Union[dict, tuple]:
        params = locals()
        params["type"] = type_
        
        tagdict = {
            "type": f'type={type_}',
            "subtype": f'subtype={subtype}',
            "row": f'row={row}',
            "col": f'col={col}',
            "row:col": f'row:col={row}:{col}',
            "type:row": f'type:row={type_}:{row}',
            "type:col":f'type:col={type_}:{col}',
            "type:row:col": f'type:row:col={type_}:{row}:{col}',
            "type:subtype": f'type:subtype={type_}:{subtype}',
            "type:subtype:row": f'type:subtype:row={type_}:{subtype}:{row}',
            "type:subtype:col": f'type:subtype:col={type_}:{subtype}:{col}',
            "type:subtype:row:col":
                f'type:subtype:row:col={type_}:{subtype}:{row}:{col}',
        }
        
        if not withkey:
            tagdict = { k: (None if (v := _v.split('=', 1)[1]) == 'None' else v)
                        for k, _v in tagdict.items() }
            tagdict["row"] = row if (row := tagdict["row"]) is None else int(row)
            tagdict["col"] = col if (col := tagdict["col"]) is None else int(col)
        
        others = tuple(others)
        
        if to_tuple:
            return tuple(tagdict.values()) + others
        
        tagdict["others"] = others
        return tagdict
    
    def _make_tag(self, key:str, *args, **kwargs) -> str:
        return self._make_tags(*args, **kwargs)[key]
    
    def _get_tags(self,
                  oid:Union[int, str],
                  withkey:bool=False,
                  to_tuple:bool=False,
                  canvas:tk.Canvas=None) -> Union[dict, tuple]:
        assert isinstance(oid, (int, str)), oid
        assert isinstance(canvas, (type(None), tk.Canvas)), canvas
        
        canvas = canvas or self.canvas
        
        others = tuple()
        tagdict = {
            "type": None, "subtype": None,
            "row": None, "col": None, "row:col": None,
            "type:row": None, "type:col": None,
            "type:subtype": None,
            "type:subtype:row": None, "type:subtype:col": None,
            "type:subtype:row:col": None
        }
        
        tags = canvas.gettags(oid)
        for tag in tags:
            try:
                key, value = tag.split('=')
            except ValueError:
                pass
            else:
                if key in tagdict:
                    tagdict[key] = tag
                    continue
            others += (tag,)
        
        if not withkey:
            tagdict = { k: (None if (v := _v.split('=', 1)[1]) == 'None' else v)
                        for k, _v in tagdict.items() }
            tagdict["row"] = row if (row := tagdict["row"]) is None else int(row)
            tagdict["col"] = col if (col := tagdict["col"]) is None else int(col)
        
        if to_tuple:
            return tuple(tagdict.values()) + others
        
        tagdict["others"] = others
        return tagdict
    
    def _get_tag(self, key:str, *args, **kwargs) -> str:
        return self._get_tags(*args, **kwargs)[key]
    
    def _get_rc(self,
                oid_or_tagdict:Union[int, dict, str],
                to_tuple:bool=False,
                canvas:tk.Canvas=None) -> Union[dict, tuple]:
        assert isinstance(oid_or_tagdict, (int, dict, str)), oid_or_tagdict
        
        if isinstance(oid_or_tagdict, dict):
            tagdict = oid_or_tagdict
        else:
            tagdict = self._get_tags(oid_or_tagdict, canvas=canvas)
        
        rc = { k: tagdict[k] for k in ["row", "col"] }
        
        if to_tuple:
            return tuple(rc.values())
        return rc
    
    def _get_rcs(self, tag:str) -> dict:
        return dict(zip(
            ["rows", "cols"],
            map(
                set,
                zip(*[ self._get_rc(oid, to_tuple=True)
                       for oid in self.canvas.find_withtag(tag) ])
            )
        ))
    
    def _update_visible_rcs_gp2s(self) -> tuple:
        vx12, vy12 = self._pview
        r12, c12, gy2s_gx2s = [None, None], [None, None], [None, None]
        for axis, [(v1, v2), i12] in enumerate(zip([vy12, vx12], [r12, c12])):
            gy2s_gx2s[axis] = np.cumsum(self._cell_sizes[axis])
            v2s = gy2s_gx2s[axis][1:]
            within = (v1 <= v2s) & (v2s <= v2)
            i12[0] = i1 = 0 if within.all() else within.argmax()
            i12[1] = len(within) - 1 if (tail := within[i1:]).all() \
                else tail.argmin() + i1
        
        self._visible_rcs = (r12[0], c12[0], r12[1], c12[1])  # (r1, c1, r2, c2)
        self._gy2s_gx2s = tuple(gy2s_gx2s)  # (y2s_headers, x2s_header)
        
        return self._visible_rcs, self._gy2s_gx2s
    
    def _canvasx(self, *xs):
        (gx1, gx2), (gy1, gy2) = self._pview
        transformed = tuple(np.asarray(xs) - gx1)
        return transformed
    
    def _canvasy(self, *ys):
        (gx1, gx2), (gy1, gy2) = self._pview
        transformed = tuple(np.asarray(ys) - gy1)
        return transformed
    
    def _fit_size(self, text:str, font, width:int, height:int) -> str:
        width, height = (max(width, 0), max(height, 0))
        canvas = self.canvas
        txt_split = text.split('\n')
        oid = canvas.create_text(*self._canvas_size, text=text, font=font)
        x1, y1, x2, y2 = canvas.bbox(oid)
        canvas.delete(oid)
        txt_longest = sorted(txt_split, key=lambda t: len(t))[-1]
        n_chars = int( len(txt_longest) / (x2 - x1) * width )
        n_lines = int( len(txt_split) / (y2 - y1) * height )
        
        return '\n'.join( t[:n_chars] for t in txt_split[:n_lines] )
    
    def _build_general_rightclick_menu(self) -> tuple:
        menu = self._rightclick_menu
        
        # Manipulate values in cells
        menu.add_separator()
        menu.add_command(
            label='Erase value(s)',
            command=lambda: self._selection_erase_values(undo=True)
        )
        menu.add_command(
            label='Copy value(s)',
            command=self._selection_copy_values
        )
        menu.add_command(
            label='Paste value(s)',
            command=lambda: self._selection_paste_values(undo=True)
        )
        menu.add_separator()
        
        # Change text colors
        menu_textcolor = tk.Menu(menu, tearoff=0)
        menu_textcolor.add_command(
            label='Choose color...',
            command=lambda: self._selection_set_foregroundcolor(undo=True)
        )
        menu_textcolor.add_command(
            label='Reset color(s)',
            command=lambda: self._selection_set_foregroundcolor(
                choose=False, undo=True)
        )
        menu.add_cascade(label='Text color(s)', menu=menu_textcolor)
        
        # Change background colors
        menu_bgcolor = tk.Menu(menu, tearoff=0)
        menu_bgcolor.add_command(
            label='Choose color...',
            command=lambda: self._selection_set_backgroundcolor(undo=True)
        )
        menu_bgcolor.add_command(
            label='Reset color(s)',
            command=lambda: self._selection_set_backgroundcolor(
                choose=False, undo=True)
        )
        menu.add_cascade(label='Background color(s)', menu=menu_bgcolor)
        
        # Change fonts
        menu_font = tk.Menu(menu, tearoff=0)
        menu_font.add_command(
            label='Choose font...',
            command=lambda: self._selection_set_font(undo=True)
        )
        menu_font.add_command(
            label='Reset font(s)',
            command=lambda: self._selection_set_font(choose=False, undo=True)
        )
        menu.add_cascade(label='Font(s)', menu=menu_font)
        
        # Change text alignments
        menu_align = tk.Menu(menu, tearoff=0)
        menu_align.add_command(
            label='Top',
            command=lambda: self._selection_set_property(
                "aligny", 'n', undo=True)
        )
        menu_align.add_command(
            label='Bottom',
            command=lambda: self._selection_set_property(
                "aligny", 's', undo=True)
        )
        menu_align.add_command(
            label='Center',
            command=lambda: self._selection_set_property(
                "aligny", 'center', undo=True)
        )
        menu_align.add_command(
            label='Reset',
            command=lambda: self._selection_set_property(
                "aligny", None, undo=True)
        )
        menu_align.add_separator()
        menu_align.add_command(
            label='Left',
            command=lambda: self._selection_set_property(
                "alignx", 'w', undo=True)
        )
        menu_align.add_command(
            label='Right',
            command=lambda: self._selection_set_property(
                "alignx", 'e', undo=True)
        )
        menu_align.add_command(
            label='Center',
            command=lambda: self._selection_set_property(
                "alignx", 'center', undo=True)
        )
        menu_align.add_command(
            label='Reset',
            command=lambda: self._selection_set_property(
                "alignx", None, undo=True)
        )
        menu.add_cascade(label='Align', menu=menu_align)
        
        menu.add_separator()
        
        submenus = (menu_textcolor, menu_bgcolor, menu_font, menu_align)
        
        return menu, submenus
    
    def _redirect_widget_event(self, event) -> tk.Event:
        widget, canvas = (event.widget, self.canvas)
        event.x += widget.winfo_x() - canvas.winfo_x()
        event.y += widget.winfo_y() - canvas.winfo_y()
        event.widget = canvas
        return event
    
    def _on_leftclick_press_selframe(self, event):
        event = self._redirect_widget_event(event)
        self.__on_leftclick_to_select(event, expand=False)
    
    def _on_leftclick_motion_selframe(self, event):
        event = self._redirect_widget_event(event)
        self.__on_leftclick_to_select(event, expand=True)
    
    def _on_leftclick_double_press_selframe(self, event):
        self._be_focus()
        event = self._redirect_widget_event(event)
        x, y, canvas = (event.x, event.y, event.widget)
        for oid in canvas.find_overlapping(x, y, x, y):
            try:
                r, c = self._get_rc(oid, to_tuple=True, canvas=canvas)
            except:  # the item does not belong to any cell. Try the next item
                continue
            self._focus_in_cell(r, c)
    
    def __on_leftclick_to_select(self, event, expand:bool):
        self._be_focus()
        x, y, canvas = (event.x, event.y, event.widget)
        for oid in canvas.find_overlapping(x, y, x, y):
            try:
                rc2 = self._get_rc(oid, to_tuple=True, canvas=canvas)
            except:  # the item does not belong to any cell. Try the next item
                continue
            r2, c2 = [ None if i < 0 else i for i in rc2 ]
            if expand:
                r1, c1, *_ = self._selection_rcs
            else:
                r1, c1 = rc2
            self._select_cells(r1, c1, r2, c2, trace=False)
    
    _on_leftclick_press = lambda self, event: self.__on_leftclick_to_select(
        event, expand=False)
    _on_leftclick_motion = lambda self, event: self.__on_leftclick_to_select(
        event, expand=True)
    
    def redraw_cornerheader(self, skip_exist=False):
        type_ = 'cornerheader'
        x1, y1, x2, y2 = (0, 0, self._cell_sizes[1][0], self._cell_sizes[0][0])
        width, height = (x2 - x1 + 1, y2 - y1 + 1)
        
        style = self._default_styles["header"]
        background, bordercolor = style["background"], style["bordercolor"]
        canvas = self.cornercanvas
        canvas.configure(width=x2 - x1 + 1, height=y2 - y1 + 1)
        self.rowcanvas.configure(width=width)
        self.colcanvas.configure(height=height)
        
        tag = self._make_tag("type", type_=type_)
        if skip_exist and canvas.find_withtag(tag):
            return
        
        # Delete the existing components
        canvas.delete(tag)
        
        # Draw components for the cornerheader
        kw = {"type_": type_, "row": -1, "col": -1, "to_tuple": True}
        # Background
        tags = self._make_tags(subtype='background', **kw)
        canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=background["normal"],
            outline=bordercolor["normal"],
            tags=tags
        )
        
        # Handles
        tags = self._make_tags(subtype='hhandle', **kw)
        canvas.create_line(x1, y2, x2, y2, width=3, fill='', tags=tags)
        
        tags = self._make_tags(subtype='vhandle', **kw)
        canvas.create_line(x2, y1, x2, y2, width=3, fill='', tags=tags)
        
        # Add bindings
        tag_cornerheader = self._make_tag("type", type_=type_)
        tag_bg = self._make_tag("type", type_=type_, subtype='background')
        canvas.tag_bind(tag_cornerheader, '<Enter>', self._on_enter_header)
        canvas.tag_bind(tag_cornerheader, '<Leave>', self._on_leave_header)
        canvas.tag_bind(
            tag_cornerheader, self.RightClick, self._on_rightclick_press_header)
        canvas.tag_bind(tag_bg, '<ButtonPress-1>', self._on_leftclick_press)
        
        for handle in ['hhandle', 'vhandle']:
            tag_cornerhandle = self._make_tag(
                "type:subtype", type_=type_, subtype=handle)
            canvas.tag_raise(tag_cornerhandle)  # topmost
            
            canvas.tag_bind(
                tag_cornerhandle,
                '<ButtonPress-1>',
                getattr(self, f"_on_leftclick_press_{handle}")
            )
    
    def redraw_headers(self, i1=None, i2=None, *, axis:int, skip_exist=False):
        axis = int(axis)
        assert (i1 is not None) or (i2 is None), (i1, i2)
        assert axis in (0, 1), axis
        
        if (i1 is None) or (i2 is None):
            r1, c1, r2, c2 = self._visible_rcs
        
        if i1 is None:
            i1, i2 = (r1, r2) if axis == 0 else (c1, c2)
        elif i2 is None:
            i2 = r2 if axis == 0 else c2
        else:
            i1, i2 = sorted([i1, i2])
        
        max_i = self.values.shape[axis] - 1
        assert 0 <= i1 <= i2 <= max_i, (i1, i2, max_i)
        
        (vx1, vx2), (vy1, vy2) = self._pview
        heights, widths = self._cell_sizes
        if axis == 0:
            type_, prefix, handle = ('rowheader', 'R', 'hhandle')
            x1, x2 = self._canvasx(vx1, vx1 + widths[0])
            y2s = self._canvasy(*self._gy2s_gx2s[axis][1:])
            y1s = y2s - heights[1:]
            coords_gen = (
                (r,
                 {"type_": type_, "row": r, "col": -1},
                 (x1, y1, x2, y2),
                 (x1, y2, x2, y2)
                )
                for r, (y1, y2) in enumerate(zip(y1s[i1:i2+1], y2s[i1:i2+1]), i1)
            )
            canvas = self.rowcanvas
        else:
            type_, prefix, handle = ('colheader', 'C', 'vhandle')
            y1, y2 = self._canvasy(vy1, vy1 + heights[0])
            x2s = self._canvasx(*self._gy2s_gx2s[axis][1:])
            x1s = x2s - widths[1:]
            coords_gen = (
                (c,
                 {"type_": type_, "row": -1, "col": c},
                 (x1, y1, x2, y2),
                 (x2, y1, x2, y2)
                )
                for c, (x1, x2) in enumerate(zip(x1s[i1:i2+1], x2s[i1:i2+1]), i1)
            )
            canvas = self.colcanvas
        
        style = self._default_styles["header"]
        background, foreground = style["background"], style["foreground"]
        bordercolor, font = style["bordercolor"], style["font"]
        
        # Draw components for each header
        for i, kw, (x1, y1, x2, y2), xys_handle in coords_gen:
            tag = self._make_tag("type:row:col", **kw)
            
            if skip_exist and canvas.find_withtag(tag):
                continue
            
            # Delete the existing components
            canvas.delete(tag)
            
            # Create new components
            kw["to_tuple"] = True
            # Background
            tags = self._make_tags(subtype='background', **kw)
            canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=background["normal"],
                outline=bordercolor["normal"],
                tags=tags
            )
            
            # Text
            tags = self._make_tags(subtype='text', **kw)
            text = self._fit_size(
                f'{prefix}{i}', font, width=x2 - x1, height=y2 - y1)
            canvas.create_text(
                (x1 + x2)/2., (y1 + y2)/2.,
                text=text,
                font=font,
                fill=foreground["normal"],
                tags=tags
            )
            
            # Handle
            tags = self._make_tags(subtype=handle, **kw)
            canvas.create_line(*xys_handle, width=3, fill='', tags=tags)
        
        # Stacking order: CornerHeader > Row/ColHeaderHandle > Row/ColHeader
        tag_header = self._make_tag("type", type_=type_)
        tag_bg = self._make_tag("type:subtype", type_=type_, subtype='background')
        tag_text = self._make_tag("type:subtype", type_=type_, subtype='text')
        tag_handle = self._make_tag("type:subtype", type_=type_, subtype=handle)
        canvas.tag_raise(tag_handle)
        
        # Add bindings
        canvas.tag_bind(tag_header, '<Enter>', self._on_enter_header)
        canvas.tag_bind(tag_header, '<Leave>', self._on_leave_header)
        canvas.tag_bind(
            tag_header, self.RightClick, self._on_rightclick_press_header)
        
        for tag in [tag_bg, tag_text]:
            canvas.tag_bind(tag, '<ButtonPress-1>', self._on_leftclick_press)
            canvas.tag_bind(tag, '<B1-Motion>', self._on_leftclick_motion)
        canvas.tag_bind(
            tag_handle,
            '<ButtonPress-1>',
            getattr(self, f"_on_leftclick_press_{handle}")
        )
    
    def _set_header_state(self,
                          tagdict:dict,
                          state:Literal[_valid_header_states],
                          skip_selected:bool=False):
        assert isinstance(tagdict, dict), tagdict
        assert state in self._valid_header_states, state
        
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), type_
        
        r1, c1, r2, c2 = self._selection_rcs
        (r1, r2), (c1, c2) = sorted([r1, r2]), sorted([c1, c2])
        
        if type_ == 'cornerheader':
            if skip_selected and (
                (r1 == c1 == 0) and
                (r2 >= self.values.shape[0] - 1) and
                (c2 >= self.values.shape[1] - 1)):
                return 'selected'
            
            canvas = self.cornercanvas
            tag = self._make_tag("type", type_=type_)
            tag_background = self._make_tag(
                "type:subtype", type_=type_, subtype='background')
            tag_text = None
        else:
            if type_ == 'rowheader':
                key = "row"
                i, i1, i2 = (tagdict[key], r1, r2)
                canvas = self.rowcanvas
            else:
                key = "col"
                i, i1, i2 = (tagdict[key], c1, c2)
                canvas = self.colcanvas
            
            if skip_selected and (i1 <= i <= i2):
                return 'selected'
            
            kw = {"type_": type_, key: i}
            tag = self._make_tag(f"type:{key}", **kw)
            tag_background = self._make_tag(
                f"type:subtype:{key}", subtype='background', **kw)
            tag_text = self._make_tag(
                f"type:subtype:{key}", subtype='text', **kw)
        
        if not canvas.find_withtag(tag):  # items have not been created yet
            return
        
        style = self._default_styles["header"]
        
        # Set background color and border color
        oid, = canvas.find_withtag(tag_background)
        canvas.itemconfigure(
            oid,
            fill=style["background"][state],
            outline=style["bordercolor"][state]
        )
        
        # Set text color
        if tag_text is not None:
            oid, = canvas.find_withtag(tag_text)
            canvas.itemconfigure(oid, fill=style["foreground"][state])
        
        return state
    
    def __on_enter_leave_header(self, event, enter_or_leave:str):
        assert enter_or_leave in ('enter', 'leave'), enter_or_leave
        
        state = 'hover' if enter_or_leave == 'enter' else 'normal'
        
        canvas = event.widget
        tagdict = self._get_tags('current', canvas=canvas)
        self._set_header_state(tagdict, state=state, skip_selected=True)
        
        # Set mouse cursor style
        cursor_styles = self._default_styles["header"]["cursor"]
        if enter_or_leave == 'enter':
            subtype = tagdict.get("subtype", None)
            cursor = cursor_styles.get(subtype, cursor_styles["default"])
        else:
            cursor = cursor_styles["default"]
        canvas.configure(cursor=cursor)
        
        self._hover = tagdict if enter_or_leave == 'enter' else None
    
    _on_enter_header = lambda self, event: self.__on_enter_leave_header(
        event, 'enter')
    _on_leave_header = lambda self, event: self.__on_enter_leave_header(
        event, 'leave')
    
    def _on_rightclick_press_header(self, event):
        tagdict = self._hover
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), self._hover
        
        # Select the current row/col if it is not selected
        r, c = self._get_rc(tagdict, to_tuple=True, canvas=event.widget)
        r1, c1, r2, c2 = self._selection_rcs
        (r1, r2), (c1, c2) = [sorted([r1, r2]), sorted([c1, c2])]
        max_r, max_c = [ s - 1 for s in self.values.shape ]
        if type_ == 'rowheader':
            axis_name, axis = ('row', 0)
            if not ((r1 <= r <= r2) and (c1 == 0) and (c2 >= max_c)):
                self._select_cells(r, 0, r, max_c, trace=False)
        elif type_ == 'colheader':
            axis_name, axis = ('column', 1)
            if not ((c1 <= c <= c2) and (r1 == 0) and (r2 >= max_r)):
                self._select_cells(0, c, max_r, c, trace=False)
        else:
            axis_name, axis = ('row', 0)
            if not ((r1 == c1 == 0) and (r1 >= max_r) and (c2 >= max_c)):
                self._select_cells(0, 0, max_r, max_c, trace=False)
        
        # Setup the right click menu
        menu = self._rightclick_menu
        
        if type_ in ('rowheader', 'colheader'):
            menu.add_command(
                label=f'Insert a new {axis_name} ahead',
                command=lambda: self._selection_insert_cells(
                    axis, mode='ahead', undo=True)
            )
            menu.add_command(
                label=f'Insert a new {axis_name} behind',
                command=lambda: self._selection_insert_cells(
                    axis, mode='behind', undo=True)
            )
        menu.add_command(
            label=f'Delete the selected {axis_name}(s)',
            command=lambda: self._selection_delete_cells(undo=True)
        )
        menu, submenus = self._build_general_rightclick_menu()
        
        menu.post(event.x_root, event.y_root)
        try:
            menu.delete(0, 'end')
        except:
            pass
        for submenu in submenus:
            submenu.destroy()
    
    def __on_leftclick_press_handle(self, event, axis:int):  # resize starts
        tagdict = self._hover
        type_ = tagdict["type"]
        assert type_ in ('cornerheader', 'rowheader', 'colheader'), self._hover
        assert axis in (0, 1), axis
        
        r, c = self._get_rc(tagdict, to_tuple=True)
        canvas = event.widget
        
        if axis == 0:
            i = -1 if r is None else r
            canvas.bind('<B1-Motion>', self._on_leftclick_motion_hhandle)
            canvas.bind('<ButtonRelease-1>', self._on_leftclick_release_handle)
        else:
            i = -1 if c is None else c
            canvas.bind('<B1-Motion>', self._on_leftclick_motion_vhandle)
            canvas.bind('<ButtonRelease-1>', self._on_leftclick_release_handle)
        
        _i = i + 1
        self._resize_start = {
            "x": event.x,
            "y": event.y,
            "i": i,
            "size": self._cell_sizes[axis][_i],
            "step": self._history.step
        }
    
    _on_leftclick_press_hhandle = lambda self, event: (
        self.__on_leftclick_press_handle(event, axis=0))
    _on_leftclick_press_vhandle = lambda self, event: (
        self.__on_leftclick_press_handle(event, axis=1))
    
    def __on_leftclick_motion_handle(self, event, axis:int):  # resizing
        start = self._resize_start
        if axis == 0:
            size = start["size"] + event.y - start["y"]
        else:
            size = start["size"] + event.x - start["x"]
        self.resize_cells(start["i"], axis, size=size, trace=False, undo=False)
        
        history = self._history
        if history.step > start["step"]:
            history.pop()
        history.add(
            forward=lambda: self.resize_cells(start["i"], axis, size=size),
            backward=lambda: self.resize_cells(
                start["i"], axis, size=start["size"])
        )
    
    _on_leftclick_motion_hhandle = lambda self, event: (
        self.__on_leftclick_motion_handle(event, axis=0))
    _on_leftclick_motion_vhandle = lambda self, event: (
        self.__on_leftclick_motion_handle(event, axis=1))
    
    def _on_leftclick_release_handle(self, event):  # resize ends
        for seq in ['<B1-Motion>', '<ButtonRelease-1>']:
            event.widget.unbind(seq)
        self._resize_start = None
    
    def redraw_cells(self, r1=None, c1=None, r2=None, c2=None, skip_exist=False):
        assert (r1 is not None) or (r2 is None), (r1, r2)
        assert (c1 is not None) or (c2 is None), (c1, c2)
        
        gy2s, gx2s = self._gy2s_gx2s
        
        if r1 is None:
            r1, _, r2, _ = self._visible_rcs
        elif r2 is None:
            _, _, r2, _ = self._visible_rcs
        else:
            r1, r2 = sorted([r1, r2])
        if c1 is None:
            _, c1, _, c2 = self._visible_rcs
        elif c2 is None:
            *_, c2 = self._visible_rcs
        else:
            c1, c2 = sorted([c1, c2])
        
        max_r, max_c = [ s - 1 for s in self.values.shape ]
        assert 0 <= r1 <= r2 <= max_r, (r1, r2, max_r)
        assert 0 <= c1 <= c2 <= max_c, (c1, c2, max_c)
        
        heights, widths = self._cell_sizes
        x2s, y2s = (self._canvasx(*gx2s[1:]), self._canvasy(*gy2s[1:]))
        x1s, y1s = (x2s - widths[1:], y2s - heights[1:])
        type_ = 'cell'
        default_style = self._default_styles["cell"]
        default_style = {
            "background": default_style["background"]["normal"],
            "foreground": default_style["foreground"]["normal"],
            "bordercolor": default_style["bordercolor"]["normal"],
            "font": default_style["font"],
            "alignx": default_style["alignx"],
            "aligny": default_style["aligny"],
            "padding": default_style["padding"]
        }
        
        # Draw components for each header
        canvas = self.canvas
        values = self.values
        cell_styles = self._cell_styles
        for r, (y1, y2) in enumerate(zip(y1s[r1:r2+1], y2s[r1:r2+1]), r1):
            for c, (x1, x2) in enumerate(zip(x1s[c1:c2+1], x2s[c1:c2+1]), c1):
                cell_style = default_style.copy()
                cell_style.update(cell_styles[r, c])
                kw = {"row": r, "col": c, "others": ('xscroll', 'yscroll')}
                tag = self._make_tag("type:row:col", type_=type_, **kw)
                
                if skip_exist and canvas.find_withtag(tag):
                    continue
                
                # Delete the existing components
                canvas.delete(tag)
                
                # Create new components
                kw["to_tuple"] = True
                # Background
                tags = self._make_tags(type_=type_, subtype='background', **kw)
                canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=cell_style["background"],
                    outline=cell_style["bordercolor"],
                    tags=tags
                )
                
                # Text
                tags = self._make_tags(type_=type_, subtype='text', **kw)
                padx, pady = cell_style["padding"]
                anchor, xy = (list(), list())
                for a, d, p1, p2 in zip(
                        [cell_style["alignx"], cell_style["aligny"]],
                        ['we', 'ns'],
                        [x1, y1],
                        [x2, y2]):
                    if a in ('n', 'w'):
                        anchor.append(d[0])
                        xy.append(p1 + (pady if 'n' in a else padx) + 1)
                    elif a in ('s', 'e'):
                        anchor.append(d[1])
                        xy.append(p2 - (pady if 's' in a else padx))
                    else:
                        xy.append((p1 + p2) / 2.)
                justify = 'left' if 'w' in anchor else (
                    'right' if 'e' in anchor else 'center')
                anchor = ''.join(anchor[::-1]) if anchor else 'center'
                text = self._fit_size(
                    values.iat[r, c],
                    cell_style["font"],
                    width=x2 - x1 - padx,
                    height=y2 - y1 - pady
                )
                canvas.create_text(
                    *xy,
                    anchor=anchor,
                    text=text,
                    justify=justify,
                    font=cell_style["font"],
                    fill=cell_style["foreground"],
                    tags=tags
                )
        
        # Add Bindings
        tag_cell = self._make_tag("type", type_=type_)
        canvas.tag_bind(
            tag_cell, self.RightClick, self._on_rightclick_press_cell)
        canvas.tag_bind(tag_cell, '<ButtonPress-1>', self._on_leftclick_press)
        canvas.tag_bind(tag_cell, '<B1-Motion>', self._on_leftclick_motion)
        canvas.tag_bind(
            tag_cell,
            '<Double-ButtonPress-1>',
            self._on_leftclick_double_press_cell
        )
    
    def _refresh_entry(self, r:Optional[int]=None, c:Optional[int]=None):
        assert (r is None) == (c is None), (r, c)
        
        if (r is None) and (c is None):
            r1, c1, r2, c2 = self._selection_rcs
            r, c = min(r1, r2), min(c1, c2)
        
        heights, widths = self._cell_sizes
        x2 = np.cumsum(widths)[c+1]
        y2 = np.cumsum(heights)[r+1]
        x1, y1 = (x2 - widths[c+1], y2 - heights[r+1])
        x1, x2 = self._canvasx(x1, x2)
        y1, y2 = self._canvasy(y1, y2)
        old_text:str = self.values.iat[r, c]
        
        default_style = self._default_styles["cell"]
        cell_style = {
            "background": default_style["background"]["normal"],
            "foreground": default_style["foreground"]["normal"],
            "bordercolor": default_style["bordercolor"]["normal"],
            "font": default_style["font"],
            "alignx": default_style["alignx"]
        }
        cell_style.update(self._cell_styles[r, c])
        
        en = self._entry
        en.configure(
            background=cell_style["background"],
            foreground=cell_style["foreground"],
            highlightcolor=cell_style["bordercolor"],
            font=cell_style["font"],
            justify={"w": 'left', "e": 'right', "center": 'center'}[
                cell_style["alignx"]]
        )
        en.place(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)
        en.lift(self.canvas)
        self._focus = (r, c, old_text)
    
    def _focus_in_cell(self, r:Optional[int]=None, c:Optional[int]=None):
        self._focus_out_cell()
        self._refresh_entry(r, c)
        self._entry.focus_set()
    
    def _focus_out_cell(self, discard:bool=False):
        if self._focus:
            r, c, old_value = self._focus
            rcs = (r, c, r, c)
            if (not discard) and (new_value := self._entry.get()) != old_value:
                # Apply the new value
                self.set_values(*rcs, values=new_value, redraw=False)
                self.redraw_cells(*rcs)
                 # put redraw here to avoid recursive function calls
                self._history.add(
                    forward=lambda: self.set_values(*rcs, values=new_value),
                    backward=lambda: self.set_values(*rcs, values=old_value)
                )
            else:  # Restore the old value
                self._entry.delete(0, 'end')
                self._entry.insert('end', old_value)
            self._entry.lower()
            self.focus_set()
            self._focus = None
    
    def _set_selection(self, r1=None, c1=None, r2=None, c2=None) -> tuple:
        assert (r1 is not None) or (r2 is None), (r1, r2)
        assert (c1 is not None) or (c2 is None), (c1, c2)
        
        max_r, max_c = [ s - 1 for s in self.values.shape ]
        r1 = 0 if r1 is None else np.clip(r1, 0, max_r)
        c1 = 0 if c1 is None else np.clip(c1, 0, max_c)
        r2 = max_r if r2 is None else np.clip(r2, 0, max_r)
        c2 = max_c if c2 is None else np.clip(c2, 0, max_c)
        
        self._selection_rcs:Tuple[int] = (r1, c1, r2, c2)
        
        return self._selection_rcs
    
    def _select_cells(
            self, r1=None, c1=None, r2=None, c2=None, trace:bool=True) -> tuple:
        self._focus_out_cell()
        
        r1, c1, r2, c2 = self._set_selection(r1, c1, r2, c2)
        r_low, r_high = sorted([r1, r2])
        c_low, c_high = sorted([c1, c2])
        
        # Update selection frames' styles
        selection_style = self._default_styles["selection"]
        color, w = selection_style["color"], selection_style["width"]
        for selframe in self._selframes:
            selframe.configure(background=color)
        
        # Relocate the selection frames
        left, top, right, bottom = self._selframes
        gy2s, gx2s = self._gy2s_gx2s
        x1, x2 = self._canvasx(gx2s[c_low] + 1, gx2s[c_high+1])
        y1, y2 = self._canvasy(gy2s[r_low] + 1, gy2s[r_high+1])
        left.place(anchor='ne', x=x1, y=y1-w, width=w, height=y2-y1+2*w)
        top.place(anchor='sw', x=x1, y=y1, width=x2-x1, height=w)
        right.place(anchor='sw', x=x2, y=y2+w, width=w, height=y2-y1+2*w)
        bottom.place(anchor='ne', x=x2, y=y2, width=x2-x1, height=w)
        
        # Relocate the viewing window to trace the last selected cell (r2, c2)
        if trace:
            (gx1_view, gx2_view), (gy1_view, gy2_view) = self._pview
            heights, widths = self._cell_sizes
            gx2, gy2 = (gx2s[c2+1] + 1, gy2s[r2+1] + 1)
            gx1, gy1 = (gx2 - widths[c2+1], gy2 - heights[r2+1])
            if gx1 < (gx1_view + widths[0]):
                self.__update_content_and_scrollbar(
                    axis=0, new=gx1 - widths[0], is_start=True)
            elif gx2 > gx2_view:
                self.__update_content_and_scrollbar(
                    axis=0, new=gx2, is_start=False)
            if gy1 < (gy1_view + heights[0]):
                self.__update_content_and_scrollbar(
                    axis=1, new=gy1 - heights[0], is_start=True)
            elif gy2 > gy2_view:
                self.__update_content_and_scrollbar(
                    axis=1, new=gy2, is_start=False)
        
        # Set each header's state
        max_r, max_c = [ s - 1 for s in self.values.shape ]
        rows_on = np.arange(r_low, r_high+1)
        cols_on = np.arange(c_low, c_high+1)
        for r in range(0, max_r+1):
            tagdict = self._make_tags(type_='rowheader', row=r, withkey=False)
            self._set_header_state(
                tagdict, state='selected' if r in rows_on else 'normal'
            )
        for c in range(0, max_c+1):
            tagdict = self._make_tags(type_='colheader', col=c, withkey=False)
            self._set_header_state(
                tagdict, state='selected' if c in cols_on else 'normal'
            )
        if ((r_low == 0) and (r_high == max_r) and
            (c_low == 0) and (c_high == max_c)):
            corner_state = 'selected'
        else:
            corner_state = 'normal'
        tagdict = self._make_tags(type_='cornerheader', withkey=False)
        self._set_header_state(tagdict, state=corner_state)
        
        # Update entry value
        self._entry.delete(0, 'end')
        self._entry.insert('end', self.values.iat[r_low, c_low])
        
        return self._selection_rcs
    
    _reselect_cells = lambda self, *args, **kw: self._select_cells(
        *self._selection_rcs, *args, **kw)
    
    def _move_selections(
            self, direction:str, area:Optional[str]=None, expand:bool=False):
        assert direction in ('up', 'down', 'left', 'right'), direction
        assert area in ('paragraph', 'all', None), area
        assert isinstance(expand, bool), expand
        
        old_r1, old_c1, old_r2, old_c2 = self._selection_rcs
        max_rc = [ s - 1 for s in self.values.shape ]
        axis = 0 if direction in ('up', 'down') else 1
        new_rc1 = [old_r1, old_c1]
        old_rc2, new_rc2 = [old_r2, old_c2], [old_r2, old_c2]
        
        if area == 'all':
            new_rc2[axis] = 0 if direction in ('up', 'left') else max_rc[axis]
            
            if not expand:  # single-cell selection
                new_rc1 = new_rc2
            
            return self._select_cells(*new_rc1, *new_rc2)
        
        elif area == 'paragraph':
            # Move the last selection to the nearset nonempty cell in the same
            # paragraph or next paragraph
            
            # Slice the cell value array to an 1-D DataFrame
            slices = [old_r2, old_c2]  # [row index, col index]
            if direction in ('up', 'left'):
                lim_rc = [0, 0]
                flip = slice(None, None, -1)
                slices[axis] = slice(None, slices[axis] + 1)
                values = self.values.iloc[tuple(slices)]
                i_correction1, i_correction2 = (-1, values.size - 1)
            else:  # down or right
                lim_rc = max_rc
                flip = slice(None, None, None)
                i_correction1, i_correction2 = (1, slices[axis])
                slices[axis] = slice(slices[axis], None)
                values = self.values.iloc[tuple(slices)]
            
            # Find the nearset nonempty cell that in the same paragraph or next
            # paragraph
            diff = np.diff((values[flip] != '').astype(int))
            vary_at = np.flatnonzero(diff)
            if vary_at.size and (vary_at[0] == 0) and (diff[0] == -1):
                vary_at = vary_at[1:]
            
            if vary_at.size:  # found
                new_rc2[axis] = (vary_at[0] if diff[vary_at[0]] == -1
                    else vary_at[0] + 1) * i_correction1 + i_correction2
            else:  # not found
                new_rc2[axis] = lim_rc[axis]
            
            if not expand:  # single-cell selection
                new_rc1 = new_rc2
            
            return self._select_cells(*new_rc1, *new_rc2)
        
        # Move the last selection by 1 step
        step = -1 if direction in ('up', 'left') else +1
        new_rc2[axis] = np.clip(old_rc2[axis] + step, 0, max_rc[axis])
        
        if not expand:  # single-cell selection
            new_rc1 = new_rc2
        
        return self._select_cells(*new_rc1, *new_rc2)
    
    def _on_leftclick_double_press_cell(self, event=None):
        tagdict = self._get_tags('current')
        self._focus_in_cell(tagdict["row"], tagdict["col"])
    
    def _on_rightclick_press_cell(self, event):
        tagdict = self._get_tags('current')
        r, c = self._get_rc(tagdict, to_tuple=True)
        r1, c1, r2, c2 = self._selection_rcs
        (r_low, r_high), (c_low, c_high) = sorted([r1, r2]), sorted([c1, c2])
        
        if not ((r_low <= r <= r_high) and (c_low <= c <= c_high)):
            self._select_cells(r, c, r, c, trace=False)
        
        menu, submenus = self._build_general_rightclick_menu()
        menu.post(event.x_root, event.y_root)
        try:
            menu.delete(0, 'end')
        except:
            pass
        for submenu in submenus:
            submenu.destroy()
    
    def redraw(self,
               update_visible_rcs:bool=True,
               skip_exist:bool=False,
               trace:bool=True):
        if update_visible_rcs:
            self._update_visible_rcs_gp2s()
        self.redraw_cornerheader(skip_exist=skip_exist)
        self.redraw_headers(axis=0, skip_exist=skip_exist)
        self.redraw_headers(axis=1, skip_exist=skip_exist)
        self.redraw_cells(skip_exist=skip_exist)
        self._reselect_cells(trace=trace)
    
    def refresh(self, scrollbar:Optional[str]='both', trace:bool=True):
        assert scrollbar in (None, 'x', 'y', 'both'), scrollbar
        
        self._canvases_delete('temp')
        self.redraw(trace=trace)
        if scrollbar in ('x', 'both'):
            self.xview_scroll(0, 'units')
        if scrollbar in ('y', 'both'):
            self.yview_scroll(0, 'units')
    
    def _canvases_delete(self, tag:str):
        self.cornercanvas.delete(tag)
        self.rowcanvas.delete(tag)
        self.colcanvas.delete(tag)
        self.canvas.delete(tag)
    
    def undo(self):
        self._focus_out_cell()
        history = self._history
        if history.backable:
            self._history.back()
        else:
            print('Not backable: current step =', history.step)
    
    def redo(self):
        self._focus_out_cell()
        history = self._history
        if history.forwardable:
            self._history.forward()
        else:
            print('Not forwardable: current step =', history.step)
    
    def reset(self, history:bool=True):
        for child in self.winfo_children():
            child.destroy()
        _history = self._history
        self.__init__(_reset=True, **self._init_configs)
        if not history:
            self._history = _history
    
    def resize_cells(self,
                     i:int,
                     axis:int,
                     size:Optional[int]=None,
                     trace:bool=True,
                     undo:bool=False):
        assert i >= -1, i
        assert axis in (0, 1), axis
        
        r1, c1, r2, c2 = self._visible_rcs
        _i = int(i) + 1
        key = ("row", "col")[axis]
        header_canvas = (self.rowcanvas, self.colcanvas)[axis]
        min_size = self._min_sizes[axis]
        old_size = self._cell_sizes[axis][_i]
        new_size = self._default_cell_sizes[axis] if size is None else int(size)
        delta = max(new_size - old_size, min_size - old_size)
        self._cell_sizes[axis][_i] += delta
        i2 = (r2, c2)[axis]
        dx, dy = [(0, delta), (delta, 0)][axis]
        
        # Move the rows below or cols on the right side
        for i_move in range(i+1, i2+1):
            tag_move = self._make_tag(key, row=i_move, col=i_move)
            header_canvas.move(tag_move, dx, dy)
            self.canvas.move(tag_move, dx, dy)
        
        tag_resize = self._make_tag(key, row=i, col=i)
        self._canvases_delete(tag_resize)  # delete the row or col
        self._update_content_size()
        
        # Redraw the deleted row or col
        (self.yview_scroll, self.xview_scroll)[axis](0, 'units')
        self._select_cells(
            **(dict(r1=i, r2=i), dict(c1=i, c2=i))[axis], trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.resize_cells(i, axis=axis, size=size),
                backward=lambda: self.resize_cells(i, axis=axis, size=old_size)
            )
    
    def insert_cells(self,
                     i:int,
                     axis:int,
                     df:Optional[pd.DataFrame]=None,
                     size:Optional[int]=None,
                     styles=None,
                     redraw:bool=True,
                     trace:bool=True,
                     undo:bool=False):
        assert axis in (0, 1), axis
        _df, shape = self.values, self.values.shape
        max_i = shape[axis] - 1
        assert 0 <= i <= max_i + 1, (i, max_i)
        assert isinstance(df, (type(None), pd.DataFrame)), df
        
        # Create a new row or col
        shape_new = list(shape)
        shape_new[axis] = 1
        df_new = pd.DataFrame(np.full(shape_new, '', dtype=object)) \
            if df is None else df
        
        # Insert the new row or col to the DataFrame
        if axis == 0:
            df_leading, df_trailing = _df.iloc[:i, :], _df.iloc[i:, :]
        else:
            df_leading, df_trailing = _df.iloc[:, :i], _df.iloc[:, i:]
        self._values = pd.concat(
            [df_leading, df_new, df_trailing],
            axis=axis,
            ignore_index=True,
            copy=False
        )
        
        # Insert a new size with the default value
        _i = i + 1
        new_size = self._default_cell_sizes[axis] if size is None else size
        self._cell_sizes[axis] = np.insert(self._cell_sizes[axis], _i, new_size)
        self._update_content_size()
        
        # Insert a new row or col of styles
        new_styles = [ dict() for _i in range(shape_new[axis-1]) ] \
            if styles is None else styles
        assert np.ndim(new_styles) == 1, np.shape(new_styles)
        self._cell_styles = np.insert(
            self._cell_styles, i, new_styles, axis=axis)
        
        self._set_selection(**(dict(r1=i, r2=i), dict(c1=i, c2=i))[axis])
        
        # Redraw
        if redraw:
            self.refresh(scrollbar=('y', 'x')[axis], trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.insert_cells(
                    i,
                    axis,
                    df=None if df is None else df.copy(),
                    size=None if size is None else copy.copy(size),
                    styles=None if styles is None else np.array(
                        [ [ d.copy() for d in dicts] for dicts in styles ]),
                    redraw=redraw
                ),
                backward=lambda: self.delete_cells(i, axis=axis, redraw=redraw)
            )
    
    def delete_cells(self,
                     i:int,
                     axis:int,
                     redraw:bool=True,
                     trace:bool=True,
                     undo:bool=False):
        assert axis in (0, 1), axis
        max_i = self.values.shape[axis] - 1
        assert 0 <= i <= max_i, (i, max_i)
        
        # Delete the row or col in the DataFrame
        if axis == 0:
            idc, idc_2d = (i, slice(None)), (slice(i, i+1), slice(None))
        else:
            idc, idc_2d = (slice(None), i), (slice(None), slice(i, i+1))
        df_deleted = self.values.iloc[idc_2d].copy()
        self.values.drop(i, axis=axis, inplace=True)
        if axis == 0:
            self.values.index = range(max_i)
        else:
            self.values.columns = range(max_i)
        
        # Delete the size
        _i = i + 1
        all_sizes = [ _sizes.copy() for _sizes in self._cell_sizes ]
        size_deleted = self._cell_sizes[axis][_i]
        self._cell_sizes[axis] = np.delete(self._cell_sizes[axis], _i)
        self._update_content_size()
        
        # Delete the row or col of styles
        styles_deleted = np.array([ d.copy() for d in self._cell_styles[idc] ])
        self._cell_styles = np.delete(self._cell_styles, i, axis=axis)
        
        self._set_selection(**(dict(r1=i, r2=i), dict(c1=i, c2=i))[axis])
        
        reset = False
        if self.values.empty:  # reset the Sheet if no cells exist
            self.reset(history=False)
            reset = True
        elif redraw:
            self.refresh(scrollbar=('y', 'x')[axis], trace=trace)
        
        if undo:
            if not reset:
                all_sizes = None
            self._history.add(
                forward=lambda: self.delete_cells(i, axis=axis, redraw=redraw),
                backward=lambda: self._undo_delete_cells(
                    i,
                    axis,
                    df=df_deleted,
                    size=size_deleted,
                    styles=styles_deleted,
                    all_sizes=all_sizes,
                    reset=reset,
                    redraw=redraw
                )
            )
    
    def _undo_delete_cells(self,
                           i:int,
                           axis:int,
                           df:pd.DataFrame,
                           size:int,
                           styles,
                           all_sizes:Optional[List[np.ndarray]]=None,
                           reset:bool=False,
                           redraw:bool=True):
        assert isinstance(df, pd.DataFrame), df
        assert df.ndim == 2, df.shape
        
        if reset:
            n_rows, n_cols = df.shape
            assert isinstance(all_sizes, list), all_sizes
            assert isinstance(styles, np.ndarray), styles
            assert df.shape[axis-1] == styles.size, (df.shape, styles.shape)
            assert n_rows == all_sizes[0].size - 1, (n_rows, all_sizes[0].size)
            assert n_cols == all_sizes[1].size - 1, (n_cols, all_sizes[1].size)
            
            self._values = df.copy()
            self._cell_sizes = [ _sizes.copy() for _sizes in all_sizes ]
            self._cell_styles = np.expand_dims(
                np.array([ d.copy() for d in styles ]),
                axis
            )
            self._update_content_size()
            self._set_selection()
            if redraw:
                self.refresh(trace=False)
                self.xview_moveto(0.)
                self.yview_moveto(0.)
        else:
            self.insert_cells(i, axis, df=df, size=size, styles=styles)
    
    def set_values(self,
                   r1=None, c1=None, r2=None, c2=None,
                   values:Union[pd.DataFrame, str]='',
                   redraw:bool=True,
                   trace:bool=True,
                   undo:bool=False):
        assert isinstance(values, (pd.DataFrame, str)), type(values)
        
        df = pd.DataFrame([[values]]) if isinstance(values, str) else values
        
        r1, c1, r2, c2 = rcs = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        
        old_values = self.values.iloc[idc].copy()
        self.values.iloc[idc] = df.copy()
        
        if redraw:
            self.redraw_cells(*rcs)
            self._reselect_cells(trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.set_values(
                    *rcs, values=df.copy(), redraw=redraw),
                backward=lambda: self.set_values(
                    *rcs, values=old_values, redraw=redraw)
            )
    
    def erase_values(self,
                     r1=None, c1=None, r2=None, c2=None,
                     redraw:bool=True,
                     trace:bool=True,
                     undo:bool=False):
        self.set_values(
            r1, c1, r2, c2, values='', redraw=redraw, undo=undo, trace=trace)
    
    def copy_values(self, r1=None, c1=None, r2=None, c2=None):
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        self.values.iloc[idc].to_clipboard(sep='\t', index=False, header=False)
    
    def set_property(self,
                     r1=None, c1=None, r2=None, c2=None,
                     *,
                     property_:str,
                     values=None,
                     redraw:bool=True,
                     trace:bool=True,
                     undo:bool=False):
        r1, c1, r2, c2 = rcs = self._set_selection(r1, c1, r2, c2)
        [r_low, r_high], [c_low, c_high] = sorted([r1, r2]), sorted([c1, c2])
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        styles = self._cell_styles[idc]
        
        if not isinstance(values, np.ndarray):
            values = np.full(styles.shape, values, dtype=object)
        assert values.shape == styles.shape, (values.shape, styles.shape)
        
        old_values = np.array(
            [ [ d.get(property_) for d in dicts ] for dicts in styles ])
        if property_ == 'font':
            for r, row in enumerate(old_values):
                for c, font in enumerate(row):
                    if isinstance(font, tk.font.Font):
                        old_values[r, c] = font.name
            
            for r, row in enumerate(values):
                for c, font in enumerate(row):
                    if isinstance(font, str):
                        values[r, c] = tk.font.nametofont(font)
        
        for style, value in zip(styles.flat, values.flat):
            if value is None:
                style.pop(property_, None)
            else:
                style[property_] = value
        
        if redraw:
            self.redraw_cells(r1, c1, r2, c2)
            self._reselect_cells(trace=trace)
        
        if undo:
            self._history.add(
                forward=lambda: self.set_property(
                    *rcs, property_=property_, values=values, redraw=redraw),
                backward=lambda: self.set_property(
                    *rcs, property_=property_, values=old_values, redraw=redraw)
            )
    
    def set_font(self,
                  r1=None, c1=None, r2=None, c2=None,
                  fonts=None,
                  choose:bool=False,
                  redraw:bool=True,
                  trace:bool=True,
                  undo:bool=False):
        if choose:
            style_topleft = self._cell_styles[min(r1, r2), min(c1, c2)]
            font_topleft = style_topleft.get("font")
            dialog = FontDialog(parent=self, default=font_topleft)
            dialog.show(position=self._center_window)
            if (fonts := dialog.result) is None:
                return
        
        self.set_property(
            r1, c1, r2, c2,
            property_='font',
            values=fonts,
            redraw=redraw,
            undo=undo,
            trace=trace
        )
    
    def _set_color(self,
                   r1=None, c1=None, r2=None, c2=None,
                   field:str='foreground',
                   colors=None,
                   choose:bool=False,
                   redraw:bool=True,
                   trace:bool=True,
                   undo:bool=False):
        assert field in ('foreground', 'background'), field
        
        if choose:
            style_topleft = self._cell_styles[min(r1, r2), min(c1, c2)]
            color_topleft = style_topleft.get(field)
            dialog = ColorChooserDialog(parent=self, initialcolor=color_topleft)
            dialog.show(position=self._center_window)
            if (colors := dialog.result) is None:
                return
            colors = colors.hex
        
        self.set_property(
            r1, c1, r2, c2,
            property_=field,
            values=colors,
            redraw=redraw,
            undo=undo,
            trace=trace
        )
    
    def set_foregroundcolor(self, *args, **kwargs):
        self._set_color(*args, field='foreground', **kwargs)
    
    def set_backgroundcolor(self, *args, **kwargs):
        self._set_color(*args, field='background', **kwargs)
    
    def _selection_insert_cells(
            self, axis:Optional[int]=None, mode:str='ahead', undo:bool=False):
        assert axis in (None, 0, 1), axis
        assert mode in ('ahead', 'behind'), mode
        
        r1, c1, r2, c2 = self._selection_rcs
        rcs = [(r1, r2), (c1, c2)] = [sorted([r1, r2]), sorted([c1, c2])]
        max_r, max_c = [ s - 1 for s in self.values.shape ]
        if axis is None:
            if (r1 == 0) and (r2 >= max_r):  # cols selected
                axis = 1
            elif (c1 == 0) and (c2 >= max_c):  # rows selected
                axis = 0
            else:
                raise ValueError(
                    'Inserting new cells requires a entire row(s)/col(s) being '
                    'selected. However, the selected row(s) and col(s) indices '
                    f'are: {r1} <= r <= {r2} and {c1} <= c <= {c2}'
                )
        else:
            (i1, i2), max_i = rcs[axis-1], [max_r, max_c][axis-1]
            assert (i1 == 0) and (i2 >= max_i), (axis, i1, i2, max_i)
        
        if mode == 'ahead':
            i = rcs[axis][0]
        else:
            i = rcs[axis][1] + 1
        self.insert_cells(i, axis=axis, undo=undo, trace=False)
    
    def _selection_delete_cells(self, undo:bool=False):
        r1, c1, r2, c2 = self._selection_rcs
        rcs = [(r1, r2), (c1, c2)] = [sorted([r1, r2]), sorted([c1, c2])]
        max_r, max_c = [ s - 1 for s in self.values.shape ]
        if (r1 == 0) and (r2 >= max_r):  # cols selected
            axis = 1
        elif (c1 == 0) and (c2 >= max_c):  # rows selected
            axis = 0
        else:
            raise ValueError(
                'Inserting new cells requires a entire row(s)/col(s) being '
                'selected. However, the selected row(s) and col(s) indices are: '
                f'{r1} <= r <= {r2} and {c1} <= c <= {c2}'
            )
        
        i1, i2 = rcs[axis]
        with self._history.add_sequence() as sequences:
            for i in range(i2, i1-1, -1):  # from higher index to lower index
                self.delete_cells(i, axis=axis, redraw=False, undo=undo)
            set_selection = lambda: self._set_selection(
                **(dict(r1=i1, r2=i2), dict(c1=i1, c2=i2))[axis])
            sequences["forward"].append(set_selection)
            sequences["forward"].append(self.refresh)
            sequences["backward"].insert(0, set_selection)
            sequences["backward"].insert(0, self.refresh)
        
        # Redraw
        self.refresh(scrollbar=('y', 'x')[axis], trace=False)
    
    def _selection_erase_values(self, undo:bool=False):
        rcs = self._selection_rcs
        self.erase_values(*rcs, undo=undo, trace=False)
    
    def _selection_copy_values(self):
        rcs = self._selection_rcs
        self.copy_values(*rcs)
    
    def _selection_paste_values(self, undo:bool=False):
        df = pd.read_clipboard(
            sep='\t',
            header=None,
            index_col=False,
            dtype=str,
            skip_blank_lines=False,
            keep_default_na=False
        )
        n_rows, n_cols = df.shape
        r1, c1, r2, c2 = self._selection_rcs
        r_low, c_low = (min(r1, r2), min(c1, c2))
        r_high, c_high = (r_low + n_rows - 1, c_low + n_cols - 1)
        
        idc = (slice(r_low, r_high + 1), slice(c_low, c_high + 1))
        n_rows_current, n_cols_current = self.values.iloc[idc].shape
        with self._history.add_sequence() as seq:
            # Insert new rows/cols before pasting if the table to be paste has 
            # a larger shape
            if (n_rows_insert := n_rows - n_rows_current):
                for r in range(r_low + 1, r_low + n_rows_insert + 1):
                    self.insert_cells(r, axis=0, redraw=False, undo=True)
            if (n_cols_insert := n_cols - n_cols_current):
                for c in range(c_low + 1, c_low + n_cols_insert + 1):
                    self.insert_cells(c, axis=1, redraw=False, undo=True)
            
            # Set the values
            self.set_values(
                r_high, c_high, r_low, c_low, values=df, redraw=False, undo=True)
            
            seq["forward"].append(self.refresh)
            seq["backward"].insert(0, self.refresh)
        
        self.refresh(trace=False)
    
    def _selection_set_property(
            self, property_:str, values=None, undo:bool=False):
        rcs = self._selection_rcs
        self.set_property(
            *rcs, property_=property_, values=values, undo=undo, trace=False)
    
    def _selection_set_font(self, choose:bool=True, undo:bool=False):
        rcs = self._selection_rcs
        self.set_font(*rcs, choose=choose, undo=undo, trace=False)
    
    def _selection_set_foregroundcolor(self, choose:bool=True, undo:bool=False):
        rcs = self._selection_rcs
        self.set_foregroundcolor(*rcs, choose=choose, undo=undo, trace=False)
    
    def _selection_set_backgroundcolor(self, choose:bool=True, undo:bool=False):
        rcs = self._selection_rcs
        self.set_backgroundcolor(*rcs, choose=choose, undo=undo, trace=False)


class Book(ttk.Frame):
    @property
    def sheets(self) -> Dict[str, Sheet]:
        return { title: props["sheet"]
                 for title, props in self._sheets_props.items() }
    
    @property
    def sheet(self) -> Union[Sheet, None]:
        return self._sheet
    
    def __init__(self, master, **kwargs):
        super().__init__(master)
        
        ## Create switch styles
        ttkstyle = ttk.Style.get_instance()
        dummy_btn = ttk.Button(self, bootstyle='link-primary')
        dummy_switch = ttk.Radiobutton(self, bootstyle='toolbutton-primary')
        dummy_entry = ttk.Entry(self)
        self._tb_btn_style = 'Book.toolbar.' + dummy_btn["style"]
        self._switch_btn_style = 'Book.switch.' + dummy_btn["style"]
        self._switch_style = 'Book.switch.' + dummy_switch["style"]
        self._entry_style = 'Book.entry.' + dummy_entry["style"]
        ttkstyle.configure(self._tb_btn_style, padding=0)
        ttkstyle.configure(self._switch_btn_style, padding=0)
        ttkstyle.configure(self._switch_style, anchor='w', padding=[5, 2])
        ttkstyle.configure(self._entry_style, padding=[5, 2])
        dummy_btn.destroy()
        dummy_switch.destroy()
        dummy_entry.destroy()
        
        # Build toolbar
        self._toolbar = tb = ttk.Frame(self)
        self._toolbar.pack(fill='x', padx=9)
        self._sidebar_hidden = True
        self._sidebar_switch = ttk.Button(
            tb,
            style=self._tb_btn_style,
            text='Sidebar',
            command=self._toggle_sidebar,
            takefocus=0
        )
        self._sidebar_switch.pack(side='left', padx=[0, 12])
        self._sidebar_undo = ttk.Button(
            tb,
            style=self._tb_btn_style,
            text='Undo',
            command=lambda: self.sheet.undo(),
            takefocus=0
        )
        self._sidebar_undo.pack(side='left', padx=[0, 3])
        self._sidebar_redo = ttk.Button(
            tb,
            style=self._tb_btn_style,
            text='Redo',
            command=lambda: self.sheet.redo(),
            takefocus=0
        )
        self._sidebar_redo.pack(side='left', padx=[0, 3])
        
        # Build inputbar
        self._inputbar = ib = ttk.Frame(self)
        self._inputbar.pack(fill='x', padx=9, pady=[9, 6])
        self._inputbar.grid_columnconfigure(0, minsize=130)
        self._inputbar.grid_columnconfigure(1, weight=1)
        self._rc_label = rc_label = ttk.Label(
            ib,
            text='R12, C123',
            width=-12,
            font=('TkDefaultFont', 10)
        )
        self._rc_label.grid(row=0, column=0, sticky='sw')
        self._entry = en = ttk.Entry(ib, style=self._entry_style)
        self._entry.grid(row=0, column=1, sticky='nesw', padx=[12, 0])
        en.bind('<FocusIn>', lambda e: self.sheet._refresh_entry())
        en.bind('<KeyPress>', lambda e: self.sheet._on_key_press_entry(e))
        
        ttk.Separator(self, takefocus=0).pack(fill='x')
        
        # Build sidebar and sheet
        self._panedwindow = pw = ttk.Panedwindow(self, orient='horizontal')
        self._panedwindow.pack(fill='both', expand=1)
        
        ## Sidebar
        self._sidebar_width = 150
        self._sidebar_fm = sbfm = ScrolledFrame(pw, scroll_orient='vertical')
        self._sidebar = sb = ButtonTriggerOrderlyContainer(sbfm, cursor='arrow')
        self._sidebar.pack(fill='both', expand=1)
        self._sidebar.set_dnd_end_callback(self._focus_on_sheet)
        self._panedwindow.add(sbfm.container)
        
        def _show_sidebar(event=None):
            assert self._sidebar_hidden, self._sidebar_hidden
            pw.unbind('<Map>')
            self._toggle_sidebar()
        
        pw.bind('<Map>', _show_sidebar)
        
        # Sheet padding frame
        self._sheet_pad_fm = spfm = ttk.Frame(pw, padding=[3, 3, 0, 0])
        self._panedwindow.add(spfm)
        
        # Build the first sheet
        self._sheet_kw = kwargs.copy()
        self._sheet_var = tk.StringVar(self, value='Sheet 1')
        self._sheet_var.trace_add('write', self._switch_sheet)
        self._sheets_props:Dict[str, list] = dict()
        self._sheet:Union[Sheet, None] = None
        self._sheets_props:Dict[str, list] = self.insert_sheet(0)
        
        # Sizegrip
        ttk.Separator(self, takefocus=0).pack(fill='x')
        
        # Focus on current sheet if any of the frames or canvas is clicked
        for widget in [self, tb, ib, rc_label, sbfm, sbfm.cropper, sb]:
            widget.configure(takefocus=0)
            widget.bind('<ButtonPress-1>', self._focus_on_sheet)
    
    def _focus_on_sheet(self, *_, **__):
        self.sheet._be_focus()
    
    def _toggle_sidebar(self):
        if self._sidebar_hidden:  # show sidebar
            self._panedwindow.insert(0, self._sidebar_fm.container)
            self._panedwindow.sashpos(0, self._sidebar_width)
        else:  # hide sidebar
            self._sidebar_width = self._panedwindow.sashpos(0)
            self._panedwindow.forget(0)
        self._sidebar_hidden = not self._sidebar_hidden
    
    def _switch_sheet(self, *_):
        title = self._sheet_var.get()
        
        last_sheet = self._sheet
        self._sheet = new_sheet = self._sheets_props[title]["sheet"]
        
        if last_sheet:
            last_sheet.pack_forget()
        new_sheet.pack(fill='both', expand=1)
        new_sheet._be_focus()
        
        self._entry.configure(textvariable=new_sheet._focus_var)
        
        return new_sheet
    
    def switch_sheet(self, index_or_title:Union[int, str]) -> Sheet:
        if isinstance(index_or_title, str):
            title = index_or_title
        else:
            title = list(self._sheets_props)[index_or_title]
        
        self._sheet_var.set(title)
        
        return self._sheet
    
    def insert_sheet(self, index:int, title:Optional[str]=None, **kwargs):
        assert isinstance(index, int), index
        assert isinstance(title, (str, type(None))), title
        
        sheet_kw = self._sheet_kw.copy()
        sheet_kw.update(kwargs)
        
        # Check title
        sheets_props = self._sheets_props
        if title is None:
            i, title = (1, 'Sheet 1')
            while title in sheets_props:
                i += 1
                title = f'Sheet {i}'
        else:
            i, _title = (1, title)
            while title in sheets_props:
                i += 1
                title = _title + f' ({i})'
        
        # Build a new sheet widget and sidebar button
        sheet = Sheet(self._sheet_pad_fm, **sheet_kw)
        frame = ttk.Frame(self._sidebar)
        ttk.Button(
            frame,
            style=self._switch_btn_style,
            text='::',
            takefocus=0
        ).pack(side='left', padx=[0, 6])
        switch = ttk.Radiobutton(
            frame,
            style=self._switch_style,
            text=title,
            value=title,
            variable=self._sheet_var,
            takefocus=0
        )
        switch.pack(side='left', fill='x', expand=1)
        
        # Modify the sheet dict
        titles, props = (list(sheets_props.keys()), list(sheets_props.values()))
        titles.insert(index, title)
        props.insert(
            index, {"sheet": sheet, "switch": switch, "switch_frame": frame})
        self._sheets_props = dict(zip(titles, props))
        
        # Remap the radio buttons
        self._sidebar.delete('all')
        self._sidebar.dnd_put(
            [ ps["switch_frame"] for ps in props ],
            sticky='we',
            expand=(True, False),
            padding=9,
            ipadding=4
        )
        self._sidebar_fm._on_map_child()
        
        self.switch_sheet(title)
        
        return self._sheets_props


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    root = ttk.Window(themename='cyborg', position=(100, 100), size=(800, 500))
    
    """
    ss = Sheet(root, bootstyle_scrollbar='light-round')
    ss.pack(fill='both', expand=1)
    
    ss.set_foregroundcolor(5, 3, 5, 3, colors='#FF0000', undo=True)
    ss.set_backgroundcolor(5, 3, 5, 3, colors='#2A7AD5', undo=True)
    ss.resize_cells(5, axis=0, size=80, trace=False, undo=True)
    
    def _set_value_method1():
        ss.set_values(4, 3, 4, 3, values='r4, c3 (method 1)')
    
    def _set_value_method2():
        ss.values.iat[5, 3] = 'R5, C3 (method 2)'
        ss.redraw_cells(5, 3, 5, 3)
    
    ss.after(1000, _set_value_method1)
    ss.after(2000, _set_value_method2)
    
    """
    
    book = Book(root)
    book.pack(fill='both', expand=1)
    
    book.insert_sheet(1, 'index = 1')
    book.insert_sheet(0, 'index = 0')
    book.insert_sheet(1, 'index = 1')
    book.insert_sheet(-1, 'index = -1')
    
    #"""
    
    root.mainloop()
    
    