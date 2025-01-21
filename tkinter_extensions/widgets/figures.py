#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:39:51 2024

@author: tungchentsai
"""

from __future__ import annotations
from typing import Any, Literal
import tkinter as tk
from tkinter.font import Font
from copy import deepcopy
from itertools import cycle as Cycle

import numpy as np
from numpy.typing import NDArray, ArrayLike
import ttkbootstrap as ttk

from .. import variables as vrb
from ..constants import Int, IntFloat, Float, Dimension
from ._others import UndockedFrame
from ._figure_config import STYLES

_anchors = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
# =============================================================================
# ---- Helpers
# =============================================================================
def _cleanup_tk_attributes(obj):
    for name, attr in list(vars(obj).items()):
        if isinstance(attr, (tk.Variable, tk.Image, _BaseElement)):
            delattr(obj, name)


def _to_px(
        root: tk.Tk,
        dimension: Dimension | ArrayLike | None
) -> float | tuple[float, ...]:
    assert isinstance(root, tk.Tk), root
    
    to_pixels = root.winfo_fpixels
    
    if dimension is None:
        return None
    
    if isinstance(dimension, (str, IntFloat)):
        return None if dimension is None else to_pixels(dimension)
    return tuple( d if d is None else to_pixels(d) for d in dimension )


def _get_sticky_p(
        direction: Literal['x', 'y'],
        start: IntFloat,
        stop: IntFloat,
        sticky: str,
        pad: IntFloat | tuple[IntFloat, IntFloat]
) -> tuple[Int, Literal['n', 'e', 's', 'w', '']]:  # returns (position, anchor)
    assert direction in ('x', 'y'), direction
    assert isinstance(start, IntFloat), start
    assert isinstance(stop, IntFloat), stop
    assert start <= stop, (start, stop)
    assert isinstance(sticky, str), sticky
    assert sticky != '', sticky
    assert sticky == 'center' or set(sticky).issubset('nesw'), sticky
    assert isinstance(pad, (IntFloat, tuple)), pad
    
    lower, upper = ('w', 'e') if direction == 'x' else ('n', 's')
    if isinstance(pad, IntFloat):
        pad = (pad, pad)
    else:  # tuple
        assert len(pad) == 2 and all( isinstance(p, IntFloat) for p in pad ), pad
    
    start2 = start + pad[0]
    stop2 = stop - pad[1]
    if start2 <= stop2:
        start, stop = start2, stop2
    
    if start > stop:
        return (start + stop) / 2., ''
    
    if sticky == 'center':
        return (start + stop) / 2., ''
    
    if lower in sticky:
        if upper in sticky:
            return (start + stop) / 2., ''
        else:
            return start, lower
    else:
        if upper in sticky:
            return stop, upper
        else:
            return (start + stop) / 2., ''


def _get_sticky_xy(  # returns (x, y, anchor)
    xys: tuple[Dimension, Dimension, Dimension, Dimension],
    sticky: str,
    padx: Dimension | tuple[Dimension, Dimension],
    pady: Dimension | tuple[Dimension, Dimension]
) -> tuple[IntFloat, IntFloat, Literal['n', 'e', 's', 'w', '']]:
    x1, y1, x2, y2 = xys
    x, anchor_x = _get_sticky_p(
        'x', x1, x2, sticky=sticky, pad=padx
    )
    y, anchor_y = _get_sticky_p(
        'y', y1, y2, sticky=sticky, pad=pady
    )
    if (anchor := anchor_y + anchor_x) == '':
        anchor = 'center'
    
    return (x, y, anchor)


def _drop_consecutive_duplicates(
        *ps: ArrayLike
) -> IntFloat | NDArray[IntFloat]:
    assert len(ps) > 0, ps
    
    ps = np.asarray(ps)  # [[x1, x2, ...], [y1, y2, ...], ...]
    assert ps.ndim in (1, 2), ps.shape    
    
    if ps.size == 0:
        return ps
    
    retain = np.diff(ps, axis=1).any(axis=0)
    
    return np.concat((ps[:, :1], ps[:, 1:][:, retain]), axis=1)


class ZorderNotFoundError(RuntimeError):
    pass


class _Transform1D:  # 1D transformation
    def __init__(
            self,
            x_lims: ArrayLike = [-np.inf, np.inf],
            y_lims: ArrayLike = [-np.inf, np.inf],
            bound: bool = True
    ):
        assert isinstance(bound, bool), bound
        x_lims = np.asarray(x_lims)
        y_lims = np.asarray(y_lims)
        assert x_lims.shape == y_lims.shape == (2,), [x_lims.shape, y_lims.shape]
        t = x_lims.dtype
        assert any( np.issubdtype(t, d) for d in IntFloat.__args__ ), x_lims.dtype
        t = y_lims.dtype
        assert any( np.issubdtype(t, d) for d in IntFloat.__args__ ), y_lims.dtype
        
        self._bound: bool = bound
        self._x_min, self._x_max = sorted(x_lims)
        self._y_min, self._y_max = sorted(y_lims)
    
    def __eq__(self, obj):
        raise NotImplementedError
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat]
    ) -> Float | NDArray[Float]:
        assert isinstance(xs, (IntFloat, np.ndarray)), xs
        if np.ndim(xs) == 0:
            xs = np.float64(xs)
        dt = xs.dtype
        assert any( np.issubdtype(dt, d) for d in IntFloat.__args__ ), xs.dtype
        
        if self._bound:
            return xs[self.bound_x(xs)]  # 1D array
        return xs  # 0D or 1D array
    
    def bound_x(
            self, xs: IntFloat | NDArray[IntFloat]
    ) -> IntFloat | NDArray[IntFloat]:
        assert isinstance(xs, (IntFloat, np.ndarray)), xs
        if np.ndim(xs) == 0:
            xs = np.float64(xs)
        return (self._x_min <= xs) & (xs <= self._x_max)
    
    def bound_y(
            self, ys: IntFloat | NDArray[IntFloat]
    ) -> IntFloat | NDArray[IntFloat]:
        assert isinstance(ys, (IntFloat, np.ndarray)), ys
        if np.ndim(ys) == 0:
            ys = np.float64(ys)
        return (self._y_min <= ys) & (ys <= self._y_max)
    
    def get_x_lims(self) -> tuple[IntFloat, IntFloat]:
        return self._x_min, self._x_max
    
    def get_y_lims(self) -> tuple[IntFloat, IntFloat]:
        return self._y_min, self._y_max
    
    def get_inverse(self):
        raise NotImplementedError


class _FirstOrderPolynomial(_Transform1D):
    def __init__(self, c0: IntFloat = 0., c1: IntFloat = 1., *args, **kwargs):
        assert isinstance(c0, IntFloat), c0
        assert isinstance(c1, IntFloat), c1
        super().__init__(*args, **kwargs)
        self._c0 = np.float64(c0)
        self._c1 = np.float64(c1)
    
    def __eq__(self, obj):
        if type(self) != type(obj):
            return False
        return self._c0 == obj._c0 and self._c1 == obj._c1
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat],
            round_: bool = False
    ) -> Float | NDArray[Float]:
        xs = super().__call__(xs)
        ys = self._c0 + self._c1 * xs  # y(x) = c0 + c1 * x (0D or 1D array)
        if round_:
            ys = ys.round()
        if self._bound:
            ys = ys[self.bound_y(ys)]  # 1D array
        return ys
    
    @classmethod
    def from_points(
            cls, xs: ArrayLike, ys: ArrayLike, *args, **kwargs
    ) -> _FirstOrderPolynomial:
        assert np.shape(xs) == np.shape(ys) == (2,), (np.shape(xs), np.shape(ys))
        
        # y(x) = c0 + c1 * x
        c1 = (ys[1] - ys[0]) / (xs[1] - xs[0])  # c1 = (y1 - y0) / (x1 - x0)
        c0 = ys[0] - c1 * xs[0]  # c0 = y0 - c1 * x0
        
        return cls(c0, c1, *args, x_lims=xs, y_lims=ys, **kwargs)
    
    def get_inverse(self) -> _FirstOrderPolynomial:
        c1_inv = 1. / self._c1  # c1_inv = 1 / c1
        c0_inv = -self._c0 / self._c1  # c0_inv = -c0 / c1
        return _FirstOrderPolynomial(
            c0=c0_inv, c1=c1_inv, x_lims=self.y_lims, y_lims=self.x_lims
        )


class _Transform2D:  # 2D transformation
    def __init__(
            self,
            inp_xs: ArrayLike = [0., 1.],
            inp_ys: ArrayLike = [0., 1.],
            out_xs: ArrayLike = [0., 1.],
            out_ys: ArrayLike = [0., 1.],
            x_transform: type = _FirstOrderPolynomial,
            y_transform: type = _FirstOrderPolynomial,
            bound: bool = True
    ):
        assert issubclass(x_transform, _Transform1D), x_transform
        assert issubclass(y_transform, _Transform1D), y_transform
        assert isinstance(bound, bool), bound
        
        self._x_tf: _Transform1D = x_transform.from_points(
            inp_xs, out_xs, bound=False
        )
        self._y_tf: _Transform1D = y_transform.from_points(
            inp_ys, out_ys, bound=False
        )
        self._inp_x_min, self._inp_x_max = self._x_tf.get_x_lims()
        self._inp_y_min, self._inp_y_max = self._y_tf.get_x_lims()
        self._out_x_min, self._out_x_max = self._x_tf.get_y_lims()
        self._out_y_min, self._out_y_max = self._y_tf.get_y_lims()
        self._bound: bool = bound
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat],
            ys: IntFloat | NDArray[IntFloat],
            round_: bool = False
    ) -> NDArray[Float]:
        assert isinstance(xs, (IntFloat, np.ndarray)), xs
        assert isinstance(ys, (IntFloat, np.ndarray)), ys
        
        if np.ndim(xs) == 0:
            xs = np.float64(xs)
        if np.ndim(ys) == 0:
            ys = np.float64(ys)
        
        if not self._bound:
            xs = self._x_tf(xs, round_=round_)
            ys = self._y_tf(ys, round_=round_)
            return xs, ys
        
        valid_inp = (self._inp_x_min <= xs) & (xs <= self._inp_x_max) \
                  & (self._inp_y_min <= ys) & (ys <= self._inp_y_max)
        xs = self._x_tf(xs[valid_inp], round_=round_)
        ys = self._y_tf(ys[valid_inp], round_=round_)
        valid_out = (self._out_x_min <= xs) & (xs <= self._out_x_max) \
                  & (self._out_y_min <= ys) & (ys <= self._out_y_max)
        xs, ys = xs[valid_out], ys[valid_out]  # two 1D arrays
        
        return np.asarray([xs, ys])


class _BaseElement:
    def __init__(self, canvas: _Plot | _Suptitle, tag: str = ''):
        assert isinstance(canvas, (_Plot, _Suptitle)), canvas
        assert isinstance(tag, str), tag
        
        root = canvas._root()
        self._canvas: _Plot | _Suptitle = canvas
        self._figure: Figure = canvas._figure
        self._to_px = lambda dim: _to_px(root, dim)
        self._tag: str = tag
    
    def __del__(self):
        _cleanup_tk_attributes(self)
    
    def draw(self):
        raise NotImplementedError


# =============================================================================
# ---- Figure Artists
# =============================================================================
class _BaseArtist(_BaseElement):#TODO: antialiasing
    _name: str
    
    def __init__(
            self,
            *args,
            transform: _Transform2D = _Transform2D(bound=False),
            x_side: Literal['b', 't'] | None = None,
            y_side: Literal['l', 'r'] | None = None,
            **kwargs
    ):
        assert x_side in ('b', 't', None), x_side
        assert y_side in ('l', 'r', None), y_side
        
        super().__init__(*args, **kwargs)
        
        tag_length = len(self._tag)
        subtags = self._tag.split('.')
        tags = [
            '.'.join(subtags[:i]) for i in range(1, tag_length)
        ]
        tags = tuple(dict.fromkeys(tags))  # unique elements
        
        self._stale: bool
        self._tags: tuple[str, ...] = tags
        self._x_side: Literal['b', 't'] | None = x_side
        self._y_side: Literal['l', 'r'] | None = y_side
        self._req_transform: _Transform2D = transform
        self._req_coords: list[Dimension] = []
        self._req_zorder: float | None = None
        self._req_style: dict[str, Any] = {}
    
    @property
    def _root_default_style(self) -> dict[str, Any]:
        return self._figure._default_style[self._name]
    
    @property
    def _default_style(self) -> dict[str, Any]:
        return self._figure._default_style[self._tag]
    
    def __del__(self):
        try:
            self.delete()
        except tk.TclError:
            pass
        self._canvas._zorder_tags.pop(self, None)
    
    def configure(self, *args, **kwargs) -> Any:
        return self._canvas.itemconfigure(self._id, *args, **kwargs)
    
    def cget(self, *args, **kwargs) -> Any:
        return self._canvas.itemcget(self._id, *args, **kwargs)
    
    def coords(self, *args, **kwargs) -> list[float]:
        return self._canvas.coords(self._id, *args, **kwargs)
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._canvas.bbox(self._id)
    
    def delete(self):
        self._canvas.delete(self._id)
    
    def hide(self):
        self._canvas.itemconfigure(self._id, state='hidden')
    
    def show(self):
        self._canvas.itemconfigure(self._id, state='normal')
    
    def set_transform(self, transform: _Transform2D | None):
        assert isinstance(transform, (_Transform2D, type(None))), transform
        
        if transform is not None and transform != self._req_transform:
            self._req_transform = transform
            self._stale = True
    
    def get_transform(self) -> _Transform2D:
        return self._req_transform
    
    def set_coords(self, *ps: Dimension):
        assert all( isinstance(p, (Dimension, type(None))) for p in ps ), ps
        
        if ps and ps != self._req_coords:
            self._req_coords[:] = ps
            self._stale = True
    
    def get_coords(self) -> list[float]:
        return self.coords()
    
    def set_zorder(self, zorder: IntFloat | None = None):
        assert isinstance(zorder, IntFloat), zorder
        
        if zorder is not None and zorder != self._req_zorder:
            self._req_zorder = float(zorder)
            self._stale = True
    
    def get_zorder(self) -> float:
        for tag in self._canvas.gettags(self._id):
            if tag.startswith('zorder='):
                return float(tag[7:])
        raise ZorderNotFoundError('Zorder has not been initialized yet.')
    
    def _update_zorder(self):
        try:
            old_zorder = self.get_zorder()
        except ZorderNotFoundError:
            pass
        else:
            self._canvas.dtag(self._id, f'zorder={old_zorder}')
        
        zorder = self._default_style["zorder"] if self._req_zorder is None \
            else self._req_zorder
        new_tag = f'zorder={zorder}'
        self._canvas.addtag_withtag(new_tag, self._id)
        self._canvas._zorder_tags[self] = new_tag
    
    def set_style(self, *args, **kwargs):
        raise NotImplementedError


class _Text(_BaseArtist):
    _name = 'text'
    
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            text: str,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            font: Font | None = None,
            **kwargs
    ):
        assert isinstance(text, str), text
        
        super().__init__(canvas=canvas, **kwargs)
        self._req_bounds: dict[str, Any] = {}
        self._padx: tuple[float, float] = (0., 0.)
        self._pady: tuple[float, float] = (0., 0.)
        self._font: Font = Font() if font is None else font
        self._id: int = canvas.create_text(
            -1, -1, anchor='se', text=text, font=self._font, tags=self._tags
        )
        self.set_style(
            text=text,
            color=color,
            angle=angle,
            family=family,
            size=size,
            weight=weight,
            slant=slant,
            underline=underline,
            overstrike=overstrike
        )
    
    def draw(self):
        if not self._stale:
            return
        
        root_defaults = self._root_default_style
        defaults = self._default_style
        cf = self._req_style.copy()
        cf.update({
            k: defaults.get(k, root_defaults[k])
            for k, v in cf.items() if v is None
        })
        
        # Update style
        self._font.configure(
            family=cf["family"],
            size=cf["size"],
            weight=cf["weight"],
            slant=cf["slant"],
            underline=cf["underline"],
            overstrike=cf["overstrike"]
        )
        self.configure(text=cf["text"], fill=cf["color"], angle=cf["angle"])
        
        # Update position
        if self._req_coords:
            x, y = self._req_coords
            anchor = 'center'
            padx, pady = (0., 0.), (0., 0.)
        else:
            cf = self._req_bounds.copy()
            cf.update({
                k: defaults.get(k, root_defaults[k])
                for k, v in cf.items() if v is None
            })
            
            # Get text size
            if bbox := self.bbox():
                tx1, ty1, tx2, ty2 = bbox
                tw, th = (tx2 - tx1), (ty2 - ty1)
            else:  # empty
                tw, th = (0, 0)
            
            (x1, y1, x2, y2), sticky = cf["xys"], cf["sticky"]
            padx, pady = self._to_px(cf["padx"]), self._to_px(cf["pady"])
            if x1 is None:
                x1 = x2 - tw - sum(padx)
            elif x2 is None:
                x2 = x1 + tw + sum(padx)
            if y1 is None:
                y1 = y2 - th - sum(pady)
            elif y2 is None:
                y2 = y1 + th + sum(pady)
            
            # `anchor` must be 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw', or
            # 'center'
            x, y, anchor = _get_sticky_xy(
                (x1, y1, x2, y2), sticky=sticky, padx=padx, pady=pady
            )
            if anchor != 'center':
                # Roll the anchor. e.g. 0 deg => 1 step, 45 deg => 2 step, ...
                angle = float(self.cget('angle'))
                assert 0.0 <= angle < 360.0, angle
                shift = int((angle + 22.5) // 45)  # step with 45 deg
                mapping = dict(zip(_anchors, _anchors[shift:] + _anchors[:shift]))
                anchor = ''.join( mapping[x] for x in anchor )  # rolling
            x, y = self._req_transform(x, y, round_=True)
        
        self.configure(anchor=anchor)  # update anchor
        self.coords(x, y)  # update position
        self._update_zorder()
        self._padx = padx
        self._pady = pady
        self._stale = False
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        if self.cget('text') and (bbox := super().bbox()):
            x1, y1, x2, y2 = bbox
            x1 -= self._padx[0]
            x2 += self._padx[1]
            y1 -= self._pady[0]
            y2 += self._pady[1]
            return (x1, y1, x2, y2)
        return None
    
    def set_bounds(
            self,
            xys: tuple[Dimension, Dimension, Dimension, Dimension] | None = None,
            sticky: str | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None
    ):
        assert isinstance(xys, (tuple, type(None))), xys
        assert isinstance(sticky, (str, type(None))), sticky
        assert isinstance(padx, (Dimension, tuple, type(None))), padx
        assert isinstance(pady, (Dimension, tuple, type(None))), pady
        if xys is not None:
            assert all( isinstance(p, (Dimension, type(None))) for p in xys ), xys
        
        if xys is not None:
            x1, y1, x2, y2 = self._to_px(xys)
            if x1 is not None and x2 is not None and x2 < x1:
                x1 = x2 = (x1 + x2) / 2.
            if y1 is not None and y2 is not None and y2 < y1:
                y1 = y2 = (y1 + y2) / 2.
            xys = (x1, y1, x2, y2)
        
        padding = []
        for pad in [padx, pady]:
            if isinstance(pad, Dimension):
                padx = (pad, pad)
            elif isinstance(pad, tuple):
                assert len(pad) == 2, [padx, pady]
                assert all( isinstance(p, Dimension) for p in pad ), [padx, pady]
            padding.append(pad)
        
        old = self._req_bounds
        new = {
            "xys": xys,
            "sticky": sticky,
            "padx": padding[0],
            "pady": padding[1]
        }
        new.update({ k: old.get(k, None) for k, v in new.items() if v is None })
        if new != old:
            self._req_bounds = new
            self._req_coords.clear()
            self._stale = True
    
    def set_style(
            self,
            text: str | None = None,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None
    ):
        assert isinstance(text, (str, type(None))), text
        assert isinstance(color, (str, type(None))), color
        assert isinstance(angle, (IntFloat, type(None))), angle
        assert isinstance(family, (str, type(None))), family
        assert isinstance(size, (Int, type(None))), size
        assert isinstance(weight, (str, type(None))), weight
        assert isinstance(slant, (str, type(None))), slant
        assert isinstance(underline, (bool, type(None))), underline
        assert isinstance(overstrike, (bool, type(None))), overstrike
        
        old = self._req_style
        new = {
            "text": text,
            "color": color,
            "angle": angle if angle is None else angle % 360.0,
            "family": family,
            "size": size,
            "weight": weight,
            "slant": slant,
            "underline": underline,
            "overstrike": overstrike
        }
        new.update({ k: old.get(k, None) for k, v in new.items() if v is None })
        if new != old:
            self._req_style = new
            self._stale = True
    
    def get_style(self) -> dict[str, Any]:
        return {
            "text": self.cget(self._id, 'text'),
            "color": self.cget(self._id, 'fill'),
            "angle": self.cget(self._id, 'angle'),
            **self._font.actual()
        }


class _Line(_BaseArtist):
    _name = 'line'
    
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            x: NDArray[IntFloat],  # data x
            y: NDArray[IntFloat],  # data y
            color: str | None = None,
            width: Dimension | None = None,
            smooth: bool | None = None,
            **kwargs
    ):
        super().__init__(canvas=canvas, **kwargs)
        self._xlims: NDArray[Float] = np.array([0., 1.], float)
        self._ylims: NDArray[Float] = np.array([0., 1.], float)
        self._req_xy: NDArray[Float] = np.array([[]], dtype=float)
        self._id: int = self._canvas.create_line(
            -1, -1, -1, -1, fill='', width='0p', tags=self._tags
        )
        self.set_data(x=x, y=y)
        self.set_style(color=color, width=width, smooth=smooth)
    
    def draw(self):
        if not self._stale:
            return
        
        root_defaults = self._root_default_style
        defaults = self._default_style
        cf = self._req_style.copy()
        cf.update({
            k: defaults.get(k, root_defaults[k])
            for k, v in cf.items() if v is None
        })
        self.configure(
            fill=cf["color"], width=cf["width"], smooth=cf["smooth"]
        )
        
        import time
        t0 = time.monotonic()
        if self._req_coords:
            xys = self._req_coords
        else:
            xy = self._req_transform(*self._req_xy, round_=True)
            xy = _drop_consecutive_duplicates(*xy)
            xys = xy.T.ravel()  # x0, y0, x1, y1, x2, y2, ...
        
        if len(xys) < 4:
            xys = (-100, -100, -100, -100)
        self.coords(*xys)
        t1 = time.monotonic()
        print(t1-t0)#???
        self._update_zorder()
        self._stale = False
    
    def set_style(
            self,
            color: str | None = None,
            width: Dimension | None = None,
            smooth: bool | None = None
    ):
        assert isinstance(color, (str, type(None))), color
        assert isinstance(width, (Dimension, type(None))), width
        assert isinstance(smooth, (bool, type(None))), smooth
        
        old = self._req_style
        new = {
            "color": color, "width": width, "smooth": smooth
        }
        new.update({ k: old.get(k, None) for k, v in new.items() if v is None })
        if new != old:
            self._req_style = new
            self._stale = True
    
    def get_style(self) -> dict[str, Any]:
        return {
            "color": self.cget('color'),
            "width": self.cget('width'),
            "smooth": self.cget('smooth')
        }
    
    def set_data(
            self,
            x: NDArray[IntFloat] | None = None,
            y: NDArray[IntFloat] | None = None
    ):
        assert isinstance(x, (np.ndarray, type(None))), x
        assert isinstance(y, (np.ndarray, type(None))), y
        if isinstance(x, np.ndarray):
            xtyp = x.dtype
            assert any( np.issubdtype(xtyp, d) for d in IntFloat.__args__ ), xtyp
        if isinstance(y, np.ndarray):
            ytyp = y.dtype
            assert any( np.issubdtype(ytyp, d) for d in IntFloat.__args__ ), ytyp
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            assert x.shape == y.shape, [x.shape, y.shape]
            assert x.ndim == y.ndim == 1, [x.shape, y.shape]
        
        if x is None and y is None:
            return
        
        length = y.size if x is None else x.size
        xy = np.empty((2, length), dtype=float)
        if x is None:
            if self._req_xy is not None:
                xy[0] = self._req_xy[0]
        else:
            xy[0] = x
        if y is None:
            if self._req_xy is not None:
                xy[1] = self._req_xy[1]
        else:
            xy[1] = y
        
        if not np.array_equal(xy, self._req_xy):
            xmin, ymin = xy.min(axis=1)
            xmax, ymax = xy.max(axis=1)
            self._req_xy = xy
            self._xlims = np.array([xmin, xmax])
            self._ylims = np.array([ymin, ymax])
            self._req_coords.clear()
            self._stale = True
    
    def get_data(self) -> tuple[NDArray[Float], NDArray[Float]]:
        return tuple(self._req_xy)


class _Rectangle(_BaseArtist):
    _name = 'rect'
    
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            facecolor: str | None = None,
            edgecolor: str | None = None,
            width: Dimension | None = None,
            **kwargs
    ):
        super().__init__(canvas=canvas, **kwargs)
        self._id: int = self._canvas.create_rectangle(
            -1, -1, -1, -1, fill='', outline='', width='0p', tags=self._tags
        )
        self.set_style(facecolor=facecolor, edgecolor=edgecolor, width=width)
    
    def draw(self):
        if not self._stale:
            return
        
        root_defaults = self._root_default_style
        defaults = self._default_style
        cf = self._req_style.copy()
        cf.update({
            k: defaults.get(k, root_defaults[k])
            for k, v in cf.items() if v is None
        })
        self.configure(
            fill=cf["facecolor"], outline=cf["edgecolor"], width=cf["width"]
        )
        
        xys = self._req_coords
        
        if len(xys) < 4:
            xys = (-100, -100, -100, -100)
        self.coords(*xys)
        self._update_zorder()
        self._stale = False
    
    def set_style(
            self,
            facecolor: str | None = None,
            edgecolor: str | None = None,
            width: Dimension | None = None
    ):
        assert isinstance(facecolor, (str, type(None))), facecolor
        assert isinstance(edgecolor, (str, type(None))), edgecolor
        assert isinstance(width, (Dimension, type(None))), width
        
        old = self._req_style
        new = {
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "width": width
        }
        new.update({ k: old.get(k, None) for k, v in new.items() if v is None })
        if new != old:
            self._req_style = new
            self._stale = True
    
    def get_style(self) -> dict[str, Any]:
        return {
            "facecolor": self.cget('fill'),
            "edgecolor": self.cget('outline'),
            "width": self.cget('width')
        }


# =============================================================================
# ---- Figure Regions
# =============================================================================
class _BaseRegion(_BaseElement):
    @property
    def _default_style(self) -> dict[str, Any]:
        return self._figure._default_style
    
    def bbox(self):
        raise NotImplementedError


class _Axis(_BaseRegion):
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            side: Literal['t', 'b', 'l', 'r'],
            *args, 
            **kwargs
    ):
        super().__init__(canvas=canvas, *args, **kwargs)
        self._side: Literal['t', 'b', 'l', 'r'] = side
        self._label: _Text = _Text(
            canvas=self._canvas, text='', tag=f'{self._tag}.label.text'
        )
    
    def draw(self):
        self._label.draw()
    
    def delete(self):
        self._label.delete()
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._label.bbox()
    
    def set_bounds(self, *args, sticky: str | None = None, **kwargs):
        invalid = {"t": 's', "b": 'n', "l": 'e', "r": 'w'}[self._side]
        
        if sticky is not None and (invalid in sticky or sticky == 'center'):
            raise ValueError(
                f"`sticky` for taxis label must not include {invalid} and not "
                f"equal to 'center' but got {sticky}."
            )
        
        self._label.set_bounds(*args, sticky=sticky, **kwargs)
    
    def set_label(self, *args, **kwargs):
        self._label.set_style(*args, **kwargs)
    
    def get_label(self) -> _Text:
        return self._label


class _Ticks(_BaseRegion):
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            side: Literal['t', 'b', 'l', 'r'],
            *args, 
            **kwargs
    ):
        super().__init__(canvas=canvas, *args, **kwargs)
        self._side: Literal['t', 'b', 'l', 'r'] = side
        self._grow: int = {"r": 0, "b": 1, "l": 2, "t": 3}[side]
        
        self._req_enable_labels: bool = False
        self._req_scientific: Int | None = None
        self._req_xys: tuple[Dimension, Dimension, Dimension, Dimension] | None \
            = None
        self._req_transform: _Transform1D = _FirstOrderPolynomial(bound=False)
        self._req_limits: list[IntFloat | None] = [None, None]
        self._req_padding: list[Dimension | None] = [None, None]
        
        self._limits: list[IntFloat] = [1., 1e+200]
        self._padding: list[float] = [0., 0.]
        self._update_labels: bool = True
        self._label_size: IntFloat = 0
        self._vertical_label: bool = False
        self._n_labels: int = 0
        self._labels: list[_Text] = []
        self._dummy_label: _Text = _Text(
            canvas=self._canvas, text='', tag=f'{self._tag}.labels.text'
        )
    
    def draw(
            self
    ) -> tuple[dict[str, Any], _Transform1D] | None:
        self._dummy_label.hide()
        if not self._req_enable_labels:
            for label in self._labels:
                label.delete()
            self._labels.clear()
            return
        
        if not self._update_labels:
            return
        
        texts, positions = self._make_labels()
        canvas = self._canvas
        tag = f'{self._tag}.labels.text'
        font = self._dummy_label._font
        style = self._dummy_label._req_style
        style.pop('text', None)
        bounds = self._dummy_label._req_bounds.copy()
        bounds.pop('xys', None)
        
        length_increase = len(texts) - len(self._labels)
        if length_increase < 0:  # delete labels
            for i in range(-length_increase):
                self._labels.pop().delete()
        elif length_increase > 0:  # create labels
            self._labels.extend(
                _Text(canvas, text='', font=font, tag=tag)
                for i in range(length_increase)
            )
        
        for label, text, xys in zip(self._labels, texts, positions):
            label.set_style(text=text, **style)
            label.set_bounds(xys=xys, **bounds)
            label.draw()
    
    def draw_dummy(self, *args, **kwargs):
        if not self._req_enable_labels:
            return
        
        # Temporarily set xys and transform to get the limits' texts and positions
        update_labels = self._update_labels
        xys, transform = self._req_xys, self._req_transform
        try:
            self._set_xys_and_transform(*args, **kwargs)
            self._update_labels = True
            texts, positions = self._make_labels(limits=True)
        finally:
            self._update_labels = update_labels
            self._req_xys, self._req_transform = xys, transform
        
        # Find possibly largest label
        n_chars = [ max(len(t) for t in tx.split('\n', 1)) for tx in texts ]
        i = np.argmax(n_chars)
        text, xys = texts[i], positions[i]
        
        # Update the text item and draw
        label = self._dummy_label
        label.set_style(text=text)
        label.set_bounds(xys=xys)
        label.show()
        label.draw()
        
        # Update current size of the text item
        tx1, ty1, tx2, ty2 = self.bbox(dummy=True)
        w, h = (tx2 - tx1), (ty2 - ty1)
        angle = float(label.cget('angle'))
        vertical = (angle - 90) % 180 == 0  # text direction
        self._n_chars = n_chars[i]
        self._label_size = w if self._side in 'tb' else h
        self._vertical_label = vertical
    
    def delete(self):
        for label in self._labels:
            label.delete()
        self._labels.clear()
    
    def bbox(self, dummy: bool = False) -> tuple[int, int, int, int] | None:
        if dummy:
            return self._dummy_label.bbox()
        elif not self._labels:
            return None
        
        x1y1x2y2 = np.asarray([
            label.bbox() for label in self._labels
        ])
        xs = np.concat((x1y1x2y2[:, 0], x1y1x2y2[:, 2]))
        ys = np.concat((x1y1x2y2[:, 1], x1y1x2y2[:, 3]))
        xs.sort()
        ys.sort()
        x1, y1, x2, y2 = xs[0], ys[0], xs[-1], ys[-1]
        
        return x1, y1, x2, y2
    
    def set_labels(
            self,
            enable: bool = True,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None,
            scientific: Int | None = None
    ):
        self._dummy_label.set_style(
            color=color,
            angle=angle,
            family=family,
            size=size,
            weight=weight,
            slant=slant,
            underline=underline,
            overstrike=overstrike
        )
        self._dummy_label.set_bounds(padx=padx, pady=pady)
        self._req_enable_labels = enable
        self._req_scientific = scientific
    
    def get_labels(self) -> list[_Text]:
        return self._labels
    
    def set_limits(
            self,
            limits: ArrayLike = [None, None],
            padding: ArrayLike | Dimension | None = [None, None]
    ):
        limits = list(limits)
        assert len(limits) == 2, limits
        assert isinstance(limits[0], (IntFloat, type(None))), limits
        assert isinstance(limits[1], (IntFloat, type(None))), limits
        if limits[0] is not None and limits[1] is not None:
            assert limits[0] <= limits[1], limits
        if isinstance(padding, str):
            padding = [padding, padding]
        padding = list(padding)
        assert len(padding) == 2, padding
        assert isinstance(padding[0], (Dimension, type(None))), padding
        assert isinstance(padding[1], (Dimension, type(None))), padding
        
        req_limits = self._req_limits
        for i, lim in enumerate(limits):
            if lim is not None:
                req_limits[i] = lim
        
        req_padding = self._req_padding
        for i, pad in enumerate(padding):
            if pad is not None:
                req_padding[i] = pad
    
    def get_limits(self) -> tuple[list[IntFloat], list[Dimension]]:
        return self._limits, self._padding
    
    def _set_xys_and_transform(
            self,
            xys: tuple[Dimension, Dimension, Dimension, Dimension] | None = None,
            transform: _Transform1D | None = None
    ):
        assert isinstance(xys, tuple) and len(xys) == 4, xys
        assert sum( p is None for p in xys ) == 1, xys
        assert xys[self._grow] is None, (self._side, xys)
        assert isinstance(transform, _Transform1D), transform
        
        if xys is not None and xys != self._req_xys:
            self._req_xys = xys
            self._update_labels = True
        if transform is not None and transform != self._req_transform:
            self._req_transform = transform
            self._update_labels = True
    
    def _fit_labels(
            self, dlimits: ArrayLike, climits: ArrayLike
    ) -> tuple[list[IntFloat], list[IntFloat]]:
        (dmin, dmax), (cmin, cmax) = sorted(dlimits), sorted(climits)
        req_dmin, req_dmax = self._req_limits
        req_pad1, req_pad2 = self._req_padding
        if req_dmin is not None:
            dmin = req_dmin
        if req_dmax is not None:
            dmax = req_dmax
        default_pad1, default_pad2 = self._default_style[f"{self._tag}.padding"]
        pad1 = self._to_px(default_pad1 if req_pad1 is None else req_pad1)
        pad2 = self._to_px(default_pad2 if req_pad2 is None else req_pad2)
        cmin, cmax = (cmin + pad1), (cmax - pad2)
        
        self._limits = [dmin, dmax]
        self._padding = [pad1, pad2]
        
        if not self._req_enable_labels:
            if dlimits[0] < dlimits[1]:
                return [dmin, dmax], [cmin, cmax]
            return [dmax, dmin], [cmin, cmax]
        
        sci = self._default_style[f"{self._tag}.labels.scientific"] \
            if self._req_scientific is None \
            else self._req_scientific
        
        if self._side in 'tb':  # top or bottom
            fixed_size = self._vertical_label
        else:  # left or right
            fixed_size = not self._vertical_label
        size = self._label_size
        if not fixed_size:
            size *= (max(self._n_chars, sci) + 2) / self._n_chars
        size *= 1.2
        
        #TODO: optimize the step value and handle n_labels <= 3
        self._n_labels = max(int((cmax - cmin) // size), 0)
        step = (dmax - dmin) / (self._n_labels - 3)
        print(self._n_labels, dmin, dmax, step)#???
        
        if dlimits[0] < dlimits[1]:
            return [dmin, dmax], [cmin, cmax]
        return [dmax, dmin], [cmin, cmax]
    
    def _make_labels(
            self,
            limits: bool = False
    ) -> tuple[list[str], NDArray[Float]]:
        assert self._update_labels, self._update_labels
        
        x1, y1, x2, y2 = self._req_xys
        transform = self._req_transform
        sci = self._default_style[f"{self._tag}.labels.scientific"] \
            if self._req_scientific is None \
            else self._req_scientific
        dmin, dmax = sorted(transform.get_x_lims())
        
        if limits:
            data = np.asarray([dmin, dmax])
        else:
            data = np.linspace(dmin, dmax, self._n_labels, endpoint=True)
        
        texts = [
            t.replace('e', '\ne') if 'e' in (t := '{0:.{1}g}'.format(d, sci))
                else t + '\n'
            for d in data
        ]
        if self._side != 'b':  # b\ne^a => e^a\nb (put the exponent above the base)
            texts = [ '\n'.join(t.split('\n', 1)[::-1]) for t in texts ]
        
        if self._side in ('t', 'b'):
            positions = [ (x, y1, x, y2) for x in transform(data, round_=True) ]
        else:  # left or right
            positions = [ (x1, y, x2, y) for y in transform(data, round_=True) ]
        
        self._update_labels = False
        
        return texts, positions


class _Frame(_BaseRegion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rect: _Rectangle = _Rectangle(
            self._canvas, tag=f'{self._tag}.rect'
        )
    
    def draw(self):
        self._rect.draw()
    
    def delete(self):
        self._rect.delete()
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._rect.bbox()
    
    def set_coords(self, *args, **kwargs):
        self._rect.set_coords(*args, **kwargs)
    
    def set_rect(self, *args, **kwargs):
        self._rect.set_style(*args, **kwargs)
    
    def get_rect(self) -> _Rectangle:
        return self._rect


# =============================================================================
# ---- Figure Subwidgets
# =============================================================================
class _BaseSubwidget:
    def __init__(self, figure: Figure, **kwargs):
        assert isinstance(figure, Figure), figure
        
        super().__init__(master=figure, **kwargs)
        root = figure._root()
        self._figure = figure
        self._to_px = lambda dim: _to_px(root, dim)
        self._draw_idle_id: str = 'after#'
        self._zorder_tags: dict[_BaseArtist, str] = {}
        self._size: tuple[int, int] = (
            self.winfo_reqwidth(), self.winfo_reqheight()
        )
        self._req_facecolor: str | None = None
        self.bind('<Configure>', self._on_configure, add=True)
    
    @property
    def _default_style(self) -> dict[str, Any]:
        return self._figure._default_style
    
    def _on_configure(self, event: tk.Event):
        self.after_cancel(self._draw_idle_id)
        self._draw_idle_id = self.after_idle(self._resize, event)
    
    def _resize(self, event: tk.Event):
        self._size = (event.width, event.height)
    
    def update_theme(self):
        self.set_facecolor(self._req_facecolor)
        
        for artist in self._zorder_tags:
            artist._stale = True
        
        event = tk.Event()
        event.width, event.height = self._size
        self._on_configure(event)
    
    def draw(self):
        raise NotImplementedError
    
    def draw_idle(self):
        self.after_cancel(self._draw_idle_id)
        self._draw_idle_id = self.after_idle(self.draw)
    
    def _set_facecolor(self, color: str | None = None) -> str:
        assert isinstance(color, (str, type(None))), color
        
        default_color = self._default_style["facecolor"]
        new_color = default_color if color is None else color
        self.configure(bg=new_color)
        self._req_facecolor = color
        
        return new_color
    
    def set_facecolor(self, color: str | None = None) -> str:
        self._set_facecolor(color=color)
    
    def get_facecolor(self) -> str:
        return self["background"]


class _Suptitle(_BaseSubwidget, tk.Canvas):
    _tag: str = 'suptitle'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text: _Text = _Text(self, text='', tag=f'{self._tag}.text')
        self.set_facecolor()
    
    def _resize(self, event: tk.Event):
        super()._resize(event)
        self.set_bounds()
        self._text.draw()
    
    def draw(self):
        self._text.draw()
        xys = self._text.bbox()
        if xys is None:
            self._text.draw()
        else:
            x1, y1, x2, y2 = xys
            self.configure(width=x2 - x1, height=y2 - y1)
            self.update_idletasks()  # triggers `self._on_configure`
    
    def get_title(self) -> _Text:
        return self._text
    
    def set_bounds(
            self,
            sticky: str | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None
    ):
        if sticky is not None and ('s' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for suptitle must not include 's' and not equal to "
                f"'center' but got {sticky}."
            )
        
        w, h = self._size
        self._text.set_bounds(
            (0, 0, w-1, None), sticky=sticky, padx=padx, pady=pady
        )
    
    def set_style(self, *args, **kwargs):
        return self._text.set_style(*args, **kwargs)
    
    def get_style(self) -> dict[str, Any]:
        return self._text.get_style()


class _Plot(_BaseSubwidget, tk.Canvas):
    _tag: str = 'plot'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        artists = {"lines": []}
        self._colors: dict[str, Cycle] = {"lines": self._create_color_cycle()}
        self._tartists: dict[str, list[_BaseArtist]] = deepcopy(artists)
        self._bartists: dict[str, list[_BaseArtist]] = deepcopy(artists)
        self._lartists: dict[str, list[_BaseArtist]] = deepcopy(artists)
        self._rartists: dict[str, list[_BaseArtist]] = deepcopy(artists)
        
        self._title: _Text = _Text(self, text='', tag='title.text')
        self._taxis: _Axis = _Axis(self, side='t', tag='taxis')
        self._baxis: _Axis = _Axis(self, side='b', tag='baxis')
        self._laxis: _Axis = _Axis(self, side='l', tag='laxis')
        self._raxis: _Axis = _Axis(self, side='r', tag='raxis')
        self._tticks: _Ticks = _Ticks(self, side='t', tag='tticks')
        self._bticks: _Ticks = _Ticks(self, side='b', tag='bticks')
        self._lticks: _Ticks = _Ticks(self, side='l', tag='lticks')
        self._rticks: _Ticks = _Ticks(self, side='r', tag='rticks')
        self._frame: _Frame = _Frame(self, tag='frame')
        
        self.set_facecolor()
    
    @property
    def artists(self) -> set:
        artists = []
        for side in 'bt':
            for arts in getattr(self, f'_{side}artists').values():
                artists.extend(arts)
        return artists
    
    def _resize(self, event: tk.Event):
        super()._resize(event)
        self.draw()
    
    def draw(self):
        # Get data limits
        _dlimits = [
            np.asarray([ a._ylims for a in self._rartists["lines"] ]),
            np.asarray([ a._xlims for a in self._bartists["lines"] ]),
            np.asarray([ a._ylims for a in self._lartists["lines"] ]),
            np.asarray([ a._xlims for a in self._tartists["lines"] ])
        ]
        dlimits = np.array([[1, 1e+200]]*4, dtype=float)
        for i, dlims in enumerate(_dlimits):
            if dlims.size:
                dlimits[i] = [dlims[:, 0].min(), dlims[:, 1].max()]
        dlimits[0] = dlimits[0][::-1]  # flip y
        dlimits[2] = dlimits[2][::-1]  # flip y
        del _dlimits
        
        # Draw items and get empty space for frame
        w, h = self._size
        cx1, cy1, cx2, cy2 = (0, 0, w-1, h-1)
        
        ## Draw title
        self._title.set_bounds((cx1, cy1, cx2, cy2))
        self._title.draw()
        ## Update upper bound if title was drawn
        if bbox := self._title.bbox():
            cy1 = bbox[3] + 1
            if cy1 > cy2:
                cy1 = cy2 = round((cy1 + cy2) / 2)
        
        ## Draw axes
        for i, side in enumerate('rblt'):
            xys = [cx1, cy1, cx2, cy2]
            xys[i] = None
            axis = self._get_axis(side)
            axis.set_bounds(tuple(xys))
            axis.draw()
        ## Update empty space
        if bbox := self._raxis.bbox():
            cx2 = bbox[0] - 1
        if bbox := self._baxis.bbox():
            cy2 = bbox[1] - 1
        if bbox := self._laxis.bbox():
            cx1 = bbox[2] + 1
        if bbox := self._taxis.bbox():
            cy1 = bbox[3] + 1
        if cy1 > cy2:
            cy1 = cy2 = round((cy1 + cy2) / 2)
        if cx1 > cx2:
            cx1 = cx2 = round((cx1 + cx2) / 2)
        
        ## Draw dummy ticks and get the empty space
        cxys_backup = (cx1, cy1, cx2, cy2)  # backup space for ticks
        climits = [[cy1, cy2], [cx1, cx2]] * 2
        for i, (side, dlims, clims) in enumerate(zip('rblt', dlimits, climits)):
            tf = _FirstOrderPolynomial.from_points(dlims, clims, bound=False)
            xys = list(cxys_backup)
            xys[i] = None
            ticks = self._get_ticks(side)
            ticks.draw_dummy(tuple(xys), tf)
            
            ## Update empty space
            if bbox := self._rticks.bbox(dummy=True):
                cx2 = bbox[0] - 1
            if bbox := self._bticks.bbox(dummy=True):
                cy2 = bbox[1] - 1
            if bbox := self._lticks.bbox(dummy=True):
                cx1 = bbox[2] + 1
            if bbox := self._tticks.bbox(dummy=True):
                cy1 = bbox[3] + 1
            if cy1 > cy2:
                cy1 = cy2 = round((cy1 + cy2) / 2)
            if cx1 > cx2:
                cx1 = cx2 = round((cx1 + cx2) / 2)
        
        ## Draw ticks and get the empty space
        climits = [[cy1, cy2], [cx1, cx2]] * 2
        for i, (side, dlims, clims) in enumerate(zip('rblt', dlimits, climits)):
            ticks = self._get_ticks(side)
            dlims[:], clims[:] = ticks._fit_labels(dlims, clims)
            tf = _FirstOrderPolynomial.from_points(dlims, clims, bound=False)
            
            xys = [cx1, cy1, cx2, cy2]
            xys[i] = None  # growing bound
            xys[i-2] = cxys_backup[i-2]  # use previous base bound
            ticks._set_xys_and_transform(tuple(xys), tf)
            ticks.draw()
        
        # Draw frame
        self._frame.set_coords(cx1, cy1, cx2, cy2)
        self._frame.draw()
        
        # Draw user defined artists
        dlimits = dict(zip('rblt', dlimits))
        climits = dict(zip('rblt', climits))
        transforms = {}
        for artist in self.artists:
            x_side, y_side = artist._x_side, artist._y_side
            side_pair = x_side + y_side
            if side_pair in transforms:
                tf = transforms[side_pair]
            else:
                tf = _Transform2D(
                    inp_xs=dlimits[x_side], inp_ys=dlimits[y_side],
                    out_xs=climits[x_side], out_ys=climits[y_side]
                )
                transforms[side_pair] = tf
            artist.set_transform(tf)
            artist.draw()
        
        # Apply zorders
        for tag in sorted(set(self._zorder_tags.values())):
            self.tag_raise(tag)
    
    def _create_color_cycle(self) -> Cycle:
        return Cycle(self._default_style["colors"])
    
    def set_title(
            self,
            text: str | None = None,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None,
    ):
        if sticky is not None and ('s' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for title must not include 's' and not equal to "
                f"'center' but got {sticky}."
            )
        
        w, h = self._size
        self._title.set_style(
            text=text,
            color=color,
            angle=angle,
            family=family,
            size=size,
            weight=weight,
            slant=slant,
            underline=underline,
            overstrike=overstrike
        )
        self._title.set_bounds(
            xys=(0, 0, w-1, None),
            sticky=sticky,
            padx=padx,
            pady=pady
        )
    
    def get_title(self) -> _Text:
        return self._title
    
    def _get_axis(self, side: Literal['t', 'b', 'l', 'r']) -> _Axis:
        assert side in ('t', 'b', 'l', 'r'), side
        return getattr(self, f'_{side}axis')
    
    def get_taxis(self) -> _Axis:
        return self._get_axis('t')
    
    def get_baxis(self) -> _Axis:
        return self._get_axis('b')
    
    def get_laxis(self) -> _Axis:
        return self._get_axis('l')
    
    def get_raxis(self) -> _Axis:
        return self._get_axis('r')
    
    def _set_axislabel(
            self,
            side: Literal['t', 'b', 'l', 'r'],
            text: str | None = None,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None
    ):
        axis = self._get_axis(side)
        axis.set_label(
            text=text,
            color=color,
            angle=angle,
            family=family,
            size=size,
            weight=weight,
            slant=slant,
            underline=underline,
            overstrike=overstrike
        )
        axis.set_bounds(
            sticky=sticky,
            padx=padx,
            pady=pady
        )
    
    def set_tlabel(self, *args, **kwargs):
        return self._set_axislabel('t', *args, **kwargs)
    
    def set_blabel(self, *args, **kwargs):
        return self._set_axislabel('b', *args, **kwargs)
    
    def set_llabel(self, *args, **kwargs):
        return self._set_axislabel('l', *args, **kwargs)
    
    def set_rlabel(self, *args, **kwargs):
        return self._set_axislabel('r', *args, **kwargs)
    
    def _get_ticks(self, side: Literal['t', 'b', 'l', 'r']) -> _Ticks:
        assert side in ('t', 'b', 'l', 'r'), side
        return getattr(self, f'_{side}ticks')
    
    def get_tticks(self) -> _Ticks:
        return self._get_ticks('t')
    
    def get_bticks(self) -> _Ticks:
        return self._get_ticks('b')
    
    def get_lticks(self) -> _Ticks:
        return self._get_ticks('l')
    
    def get_rticks(self) -> _Ticks:
        return self._get_ticks('r')
    
    def _set_tickslabels(
            self,
            side: Literal['t', 'b', 'l', 'r'],
            *args,
            **kwargs
    ):
        ticks = self._get_ticks(side)
        ticks.set_labels(*args, **kwargs)
    
    def set_ttickslabels(self, *args, **kwargs):
        return self._set_tickslabels('t', *args, **kwargs)
    
    def set_btickslabels(self, *args, **kwargs):
        return self._set_tickslabels('b', *args, **kwargs)
    
    def set_ltickslabels(self, *args, **kwargs):
        return self._set_tickslabels('l', *args, **kwargs)
    
    def set_rtickslabels(self, *args, **kwargs):
        return self._set_tickslabels('r', *args, **kwargs)
    
    def get_frame(self) -> _Frame:
        return self._frame
    
    def set_tlimits(self, *args, **kwargs):
        self._tticks.set_limits(*args, **kwargs)
    
    def set_blimits(self, *args, **kwargs):
        self._bticks.set_limits(*args, **kwargs)
    
    def set_llimits(self, *args, **kwargs):
        self._lticks.set_limits(*args, **kwargs)
    
    def set_rlimits(self, *args, **kwargs):
        self._rticks.set_limits(*args, **kwargs)
    
    def get_tlimits(self) -> tuple[list[IntFloat], list[Dimension]]:
        return self._tticks.get_limits()
    
    def get_blimits(self) -> tuple[list[IntFloat], list[Dimension]]:
        return self._bticks.get_limits()
    
    def get_llimits(self) -> tuple[list[IntFloat], list[Dimension]]:
        return self._lticks.get_limits()
    
    def get_rlimits(self) -> tuple[list[IntFloat], list[Dimension]]:
        return self._rticks.get_limits()
    
    def plot(
            self,
            b: ArrayLike | None = None,
            l: ArrayLike | None = None,
            t: ArrayLike | None = None,
            r: ArrayLike | None = None,
            color: str | None = None,
            width: Dimension | None = None,
            smooth: bool | None = None,
            label: str = ''
    ) -> _Line:
        assert isinstance(label, str), label
        
        if not ((b is None) ^ (t is None)):
            raise ValueError('Either `b` ot `t` must be a arraylike value.')
        if not ((l is None) ^ (r is None)):
            raise ValueError('Either `l` ot `r` must be a arraylike value.')
        
        if b is not None:  # b
            x_side = 'b'
            x = np.asarray(b, dtype=float)
            x_artists = self._bartists
        else:  # t
            x_side = 't'
            x = np.asarray(t, dtype=float)
            x_artists = self._tartists
        if l is not None:  # l
            y_side = 'l'
            y = np.asarray(l, dtype=float)
            y_artists = self._lartists
        else:  # r
            y_side = 'r'
            y = np.asarray(r, dtype=float)
            y_artists = self._rartists
        assert x.shape == y.shape, [x.shape, y.shape]
        
        color_cycle = self._colors["lines"]
        x_lines = x_artists["lines"]
        y_lines = y_artists["lines"]
        
        line = _Line(
            self,
            x=x.ravel(),
            y=y.ravel(),
            color=next(color_cycle),
            width=width,
            smooth=smooth,
            x_side=x_side,
            y_side=y_side,
            tag='line'
        )
        x_lines.append(line)
        y_lines.append(line)
        
        return line


class _Toolbar(_BaseSubwidget, tk.Frame):
    _tag: str = 'toolbar'
    
    def __init__(
            self,
            figure: Figure,
            var_coord: tk.Variable,
            **kwargs
    ):
        super().__init__(figure=figure, **kwargs)
        
        self._home_bt = ttk.Button(self, text='Home', command=self._home_view)
        self._home_bt.pack(side='left')
        self._prev_bt = ttk.Button(self, text='Prev', command=self._prev_view)
        self._prev_bt.pack(side='left', padx=(3, 0))
        self._next_bt = ttk.Button(self, text='Next', command=self._next_view)
        self._next_bt.pack(side='left', padx=(3, 0))
        self._pan_bt = ttk.Button(self, text='Pan', command=self._pan_view)
        self._pan_bt.pack(side='left', padx=(6, 0))
        self._zoom_bt = ttk.Button(self, text='Zoom', command=self._zoom_view)
        self._zoom_bt.pack(side='left', padx=(3, 0))
        self._xyz_lb = tk.Label(self, textvariable=var_coord)
        self._xyz_lb.pack(side='left', padx=(6, 0))
        
        self.set_facecolor()
    
    def set_facecolor(self, color: str | None = None):
        new_color = super()._set_facecolor(color=color)
        self._xyz_lb.configure(bg=new_color)
    
    def _home_view(self):
        raise NotImplementedError
    
    def _prev_view(self):
        raise NotImplementedError
    
    def _next_view(self):
        raise NotImplementedError
    
    def _pan_view(self):
        raise NotImplementedError
    
    def _zoom_view(self):
        raise NotImplementedError


# =============================================================================
# ---- Figure Widgets
# =============================================================================
class Figure(UndockedFrame):
    def __init__(
            self,
            master: tk.Misc,
            suptitle: str = '',
            toolbar: bool = True,
            width: Dimension | None = None,
            height: Dimension | None = None,
            padx: Dimension = '6p',
            pady: Dimension = '6p',
            **kwargs
    ):
        window_title = suptitle or 'Figure'
        super().__init__(
            master, window_title=window_title, padx=padx, pady=pady, **kwargs
        )
        
        root = self._root()
        self._initialized: bool = False
        self._req_size: tuple[Int, Int]
        self._plots: NDArray[_Plot]
        self._to_px = lambda dim: _to_px(root, dim)
        self._draw_idle_id: str = 'after#'
        self._default_style: dict[str, Any] = STYLES[
            self._root().style.theme.type
        ].copy()
        self._var_coord: vrb.StringVar = vrb.StringVar(self, value='()')
        self.set_size(width=width, height=height)
        
        self._suptitle: _Suptitle
        if suptitle:
            self.set_suptitle(text=suptitle)
        
        self._toolbar: _Toolbar
        if toolbar:
            self.set_toolbar(True)
        
        self.grid_propagate(False)  # allow `self` to be resized
        
        self.bind('<Destroy>', self._on_destroy, add=True)
        self.bind('<<ThemeChanged>>', self._on_theme_changed, add=True)
        
        self.draw_idle()
    
    def _on_destroy(self, event: tk.Event | None = None):
        _cleanup_tk_attributes(self)
    
    def _on_theme_changed(self, event: tk.Event):
        self.update_theme()
        self.draw_idle()
    
    def update_theme(self):
        # Update undock button
        id_ = str(id(self))
        if not (udbt_style := self.undock_button["style"]).startswith(id_):
            udbt_style = f'{id(self)}.{udbt_style}'
        udbt_bg = self._default_style["facecolor"]
        self.undock_button.configure(style=udbt_style)
        style = self._root().style
        style._build_configure(
            udbt_style,
            **dict.fromkeys(
                ['background', 'bordercolor', 'darkcolor', 'lightcolor'],
                udbt_bg
            )
        )
        style.map(
            udbt_style,
            **dict.fromkeys(
                ['background', 'bordercolor', 'darkcolor', 'lightcolor'],
                []
            )
        )
        self.configure(bg=self._default_style["facecolor"])
        
        if not self._initialized:
            return
        
        self._default_style = STYLES[self._root().style.theme.type]
        
        # Update suptitle
        if hasattr(self, '_suptitle'):
            self._suptitle.update_theme()
        
        # Update plots
        if hasattr(self, '_plots'):
            for plot in self._plots.flat:
                plot.update_theme()
        
        # Update toolbar
        if hasattr(self, '_toolbar'):
            self._toolbar.update_theme()
    
    def draw(self):
        self._initialized = True
        
        if hasattr(self, '_suptitle'):
            self._suptitle.draw()
        
        if hasattr(self, '_plots'):
            for plot in self._plots.flat:
                if plot:
                    plot.draw()
    
    def draw_idle(self):
        self.after_cancel(self._draw_idle_id)
        self._draw_idle_id = self.after_idle(self.draw)
    
    def set_size(
            self,
            width: Dimension | None = None,
            height: Dimension | None = None
    ):
        default_width, default_height = self._default_style["size"]
        width = default_width if width is None else self._to_px(width)
        height = default_height if height is None else self._to_px(height)
        new_size = (width, height)
        
        if not hasattr(self, '_req_size') or self._req_size != new_size:
            self._req_size = new_size
            self.configure(width=width, height=height)
    
    def get_size(self) -> tuple[Int, Int]:
        return self._req_size
    
    def set_suptitle(
            self,
            text: str | None = None,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None,
            facecolor: str | None = None
    ) -> _Suptitle | None:
        if text is not None:  # enable suptitle
            if not hasattr(self, '_suptitle'):
                self._suptitle = _Suptitle(self)
                self._suptitle.grid(row=0, column=0, sticky='we')
                if hasattr(self, '_plots'):
                    n_rows, n_cols = self._plots.shape
                    self._suptitle.grid(columnspan=n_cols)
            self._suptitle.set_facecolor(color=facecolor)
            self._suptitle.set_style(
                text=text,
                color=color,
                angle=angle,
                family=family,
                size=size,
                weight=weight,
                slant=slant,
                underline=underline,
                overstrike=overstrike
            )
            self._suptitle.set_bounds(
                sticky=sticky,
                padx=padx,
                pady=pady
            )
            
            return self._suptitle
        
        # Disable suptitle
        if hasattr(self, '_suptitle'):
            self._suptitle.destroy()
            delattr(self, '_suptitle')
    
    def get_suptitle(self) -> _Suptitle:
        return self._suptitle
    
    def set_plots(
            self,
            n_rows: Int = 1,
            n_cols: Int = 1,
            width_ratios: list[Int] = [],
            height_ratios: list[Int] = [],
            padx: Dimension | tuple[Dimension, Dimension] = ('3p', '3p'),
            pady: Dimension | tuple[Dimension, Dimension] = ('3p', '3p')
    ) -> NDArray[_Plot] | _Plot:
        assert isinstance(n_rows, Int) and n_rows >= 1, n_rows
        assert isinstance(n_cols, Int) and n_cols >= 1, n_cols
        assert isinstance(width_ratios, list), width_ratios
        assert all( isinstance(r, Int) for r in width_ratios ), width_ratios
        assert all( r >= 0 for r in width_ratios ), width_ratios
        assert len(width_ratios) in (0, n_cols), (n_cols, width_ratios)
        assert isinstance(height_ratios, list), height_ratios
        assert all( isinstance(r, Int) for r in height_ratios ), height_ratios
        assert all( r >= 0 for r in height_ratios ), height_ratios
        assert len(height_ratios) in (0, n_rows), (n_rows, height_ratios)
        
        width_ratios = width_ratios or [1] * n_cols
        height_ratios = height_ratios or [1] * n_rows
        
        # Clean up old plots
        if hasattr(self, '_plots'):
            for r, row in enumerate(self._plots):
                for c, plot in enumerate(row):
                    plot.grid_forget()
                    self.grid_columnconfigure(c, weight=0)
                self.grid_rowconfigure(r, weight=0)
        
        # Update suptitle's position
        if hasattr(self, '_suptitle'):
            self._suptitle.grid(columnspan=n_cols)
        
        # Update toolbar's position
        if hasattr(self, '_toolbar'):
            self._toolbar.grid(row=n_rows+1, columnspan=n_cols)
        
        # Create plots
        self._plots: NDArray[_Plot] = np.array([
            [ _Plot(self) for c in range(n_cols) ] for r in range(n_rows)
        ])
        for r, row in enumerate(self._plots, 1):  # plots start from 2nd row
            for c, plot in enumerate(row):
                plot.grid(row=r, column=c, sticky='nesw', padx=padx, pady=pady)
                plot.configure(width=0, height=0)  # this makes all space as
                 # extra space which will be distributed to each row and column
                 # with each weight specified in `grid_rowconfigure` and
                 # `grid_columnconfigure` respectively.
        
        # Set the size ratios
        for r, weight in enumerate(height_ratios, 1):  # plots start from 2nd row
            self.grid_rowconfigure(r, weight=weight)
        for c, weight in enumerate(width_ratios):
            self.grid_columnconfigure(c, weight=weight)
        
        if n_rows == 1 and n_cols == 1:  # single plot
            return self._plots[0, 0]
        elif n_rows != 1 and n_cols != 1:  # 2-D array of plots
            return self._plots
        return self._plots.ravel()  # 1-D array of plots
    
    def get_plots(self) -> NDArray[_Plot] | _Plot:
        n_rows, n_cols = self._plots.shape
        
        if n_rows == 1 and n_cols == 1:  # single plot
            return self._plots[0, 0]
        elif n_rows != 1 and n_cols != 1:  # 2-D array of plots
            return self._plots
        return self._plots.ravel()  # 1-D array of plots
    
    def set_toolbar(self, enable: bool = True) -> _Toolbar | None:
        if enable and not hasattr(self, '_toolbar'):
            self._toolbar = _Toolbar(
                self,
                var_coord=self._var_coord
            )
            kw = {"column": 0, "sticky": 'we', "padx": 9, "pady": (9, 0)}
            if hasattr(self, '_plots'):
                n_rows, n_cols = self._plots.shape
                self._toolbar.grid(row=n_rows+1, columnspan=n_cols, **kw)
            else:
                self._toolbar.grid(row=1, **kw)
                 # the toolbar will be `grid` again when `set_plots` is called
                self.grid_columnconfigure(0, weight=1)
        elif not enable and hasattr(self, '_toolbar'):
            self._toolbar.destroy()
            delattr(self, '_toolbar')
        
        if enable:
            return self._toolbar
    
    def get_toolbar(self) -> _Toolbar:
        return self._toolbar


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    from ._others import Window
    
    root = Window(title='Figure Demo')
    
    x = np.arange(0, 10, 1/48000, dtype=float)
    y = np.sin(2*np.pi*1*x)
    #x = np.array([3, 6, 6, 3, 3], dtype=float)
    #y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5], dtype=float)
    
    fig = Figure(root, toolbar=True)
    fig.pack(fill='both', expand=True)
    
    suptitle = fig.set_suptitle('<Suptitle>')
    plt = fig.set_plots(1, 1)
    plt.plot(x, y)
    plt.set_title('<Title>')
    plt.set_tlabel('<top-label>')
    plt.set_blabel('<bottom-label>')
    plt.set_llabel('<left-label>')
    plt.set_rlabel('<right-label>')
    plt.set_btickslabels(True)
    plt.set_ltickslabels(True)
    #plt.set_ttickslabels(True)
    #plt.set_rtickslabels(True)
    
    fig.after(3000, lambda: root.style.theme_use('cyborg'))
    
    root.mainloop()

