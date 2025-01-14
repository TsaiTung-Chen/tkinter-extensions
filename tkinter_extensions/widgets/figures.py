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
from contextlib import contextmanager

import numpy as np
from numpy.dtypes import StringDType
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
        dimension: Dimension | ArrayLike[Dimension] | None
) -> float | tuple[float, ...]:
    assert isinstance(root, tk.Tk), root
    
    to_pixels = root.winfo_fpixels
    
    if dimension is None:
        return None
    
    if len(dimensions := list(dimension)) > 1:
        return tuple( d if d is None else to_pixels(d) for d in dimensions )
    return None if dimension is None else to_pixels(dimension)


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
    
    if start == stop:
        return (start + stop) / 2., ''
    
    start += pad[0]
    stop -= pad[1]
    
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


def _get_transforms(
        inp_xs: ArrayLike,
        inp_ys: ArrayLike,
        out_xs: ArrayLike,
        out_ys: ArrayLike
):
    x_tf = _FirstOrderPolynomial.from_points(
        inp_xs, out_xs, x_interval=inp_xs, y_interval=out_xs
    )
    y_tf = _FirstOrderPolynomial.from_points(
        inp_ys, out_ys, x_interval=inp_ys, y_interval=out_ys
    )
    return x_tf, y_tf


class ZorderNotFoundError(RuntimeError):
    pass


class _BaseTransform:
    def __init__(
            self,
            x_interval: list[IntFloat] = [-np.inf, +np.inf],
            y_interval: list[IntFloat] = [-np.inf, +np.inf]
    ):
        assert isinstance(x_interval, list), x_interval
        assert isinstance(y_interval, list), y_interval
        assert len(x_interval) == len(y_interval) == 2, (x_interval, y_interval)
        assert all( isinstance(x, IntFloat) for x in x_interval ), x_interval
        assert all( isinstance(y, IntFloat) for y in y_interval ), y_interval
        self._x_interval: list[IntFloat] = x_interval
        self._y_interval: list[IntFloat] = y_interval
    
    @property
    def get_x_interval(self) -> tuple[IntFloat, IntFloat]:
        return self._x_interval
    
    @property
    def get_y_interval(self) -> tuple[IntFloat, IntFloat]:
        return self._y_interval
    
    def __eq__(self, obj):
        raise NotImplementedError
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat]
    ) -> Float | NDArray[Float]:
        assert isinstance(xs, (IntFloat, np.ndarray)), xs
        if isinstance(xs, (int, float)):
            xs = np.float64(xs)
        dt = xs.dtype
        assert any( np.issubdtype(dt, d) for d in IntFloat.__args__ ), xs.dtype
        return np.clip(xs, *self._x_interval)
    
    def get_inverse(self):
        raise NotImplementedError


class _FirstOrderPolynomial(_BaseTransform):
    def __init__(self, c0: IntFloat, c1: IntFloat, *args, **kwargs):
        assert isinstance(c0, IntFloat), c0
        assert isinstance(c1, IntFloat), c1
        super().__init__(*args, **kwargs)
        self._c0 = np.float64(c0)
        self._c1 = np.float64(c1)
    
    def __eq__(self, obj):
        if type(self) != type(obj):
            return False
        return self._c0 == obj._c0 and self._c1 == obj._c1 \
            and self.get_x_interval == obj.get_x_interval \
            and self.get_y_interval == obj.get_y_interval
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat]
    ) -> Float | NDArray[Float]:
        xs = super().__call__(xs)
        return np.clip(self._c0 + self._c1 * xs, *self._y_interval)
         # y(x) = c0 + c1 * x
    
    @classmethod
    def from_points(
            cls, xs: ArrayLike, ys: ArrayLike, **kwargs
    ) -> _FirstOrderPolynomial:
        assert np.shape(xs) == np.shape(ys) == (2,), (np.shape(xs), np.shape(ys))
        
        # y(x) = c0 + c1 * x
        c1 = (ys[1] - ys[0]) / (xs[1] - xs[0])  # c1 = (y1 - y0) / (x1 - x0)
        c0 = ys[0] - c1 * xs[0]  # c0 = y0 - c1 * x0
        
        return cls(c0=c0, c1=c1, **kwargs)
    
    def get_inverse(self) -> _FirstOrderPolynomial:
        c1_inv = 1. / self._c1  # c1_inv = 1 / c1
        c0_inv = -self._c0 / self._c1  # c0_inv = -c0 / c1
        return _FirstOrderPolynomial(
            c0=c0_inv,
            c1=c1_inv,
            x_interval=self._y_interval,
            y_interval=self._x_interval
        )


class _BBox:#TODO: delete
    pass

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
class _BaseArtist(_BaseElement):
    _name: str
    
    def __init__(
            self,
            *args,
            transforms: tuple[_BaseTransform, _BaseTransform] \
                = (_FirstOrderPolynomial(0, 1), _FirstOrderPolynomial(0, 1)),
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        tag_length = len(self._tag)
        subtags = self._tag.split('.')
        tags = [
            '.'.join(subtags[:i]) for i in range(1, tag_length)
        ]
        tags = tuple(dict.fromkeys(tags))  # unique elements
        
        self._stale: bool
        self._tags: tuple[str, ...] = tags
        self._req_transforms: tuple[_BaseTransform, _BaseTransform] = transforms
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
        #TODO: delete from Plot
    
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
    
    def set_transforms(
            self,
            x_transform: _BaseTransform | None = None,
            y_transform: _BaseTransform | None = None,
    ):
        assert isinstance(x_transform, _BaseTransform), x_transform
        assert isinstance(y_transform, _BaseTransform), y_transform
        
        if x_transform is not None and x_transform != self._req_transforms[0]:
            self._req_transforms = (x_transform, self._req_transforms[1])
            self._stale = True
        if y_transform is not None and y_transform != self._req_transforms[1]:
            self._req_transforms = (self._req_transforms[0], y_transform)
            self._stale = True
    
    def get_transforms(self) -> tuple[_BaseTransform, _BaseTransform]:
        return self._req_transforms
    
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
            if self.cget('text') == '':  # empty
                itw, ith = (0, 0)
            else:
                itx1, ity1, itx2, ity2 = self.bbox()
                itw, ith = (itx2 - itx1), (ity2 - ity1)
            
            (x1, y1, x2, y2), sticky = cf["xys"], cf["sticky"]
            padx, pady = self._to_px(cf["padx"]), self._to_px(cf["pady"])
            if x1 is None:
                x1 = x2 - itw - sum(padx)
            elif x2 is None:
                x2 = x1 + itw + sum(padx)
            if y1 is None:
                y1 = y2 - ith - sum(pady)
            elif y2 is None:
                y2 = y1 + ith + sum(pady)
            
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
        
        x_transform, y_transform = self._req_transforms
        x = x_transform(x)
        y = y_transform(y)
        
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
        self._xymin: NDArray[Float] = np.array([0., 1.], float)
        self._xymax: NDArray[Float] = np.array([0., 1.], float)
        self._req_xy: NDArray[Float] = np.array([[]], dtype=float)
        self._req_default_color: str = ''
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
        
        if self._req_coords:
            xys = self._req_coords
        else:
            x_transform, y_transform = self._req_transforms
            xys = np.concat(  # x0, y0, x1, y1, x2, y2, ...
                (
                    x_transform(self._req_xy[0])[:, None],
                    y_transform(self._req_xy[1])[:, None]
                ),
                axis=1
            ).ravel()
        
        self.coords(*xys)
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
            self._req_xy = xy
            self._xymin = xy.min(axis=1)
            self._xymax = xy.max(axis=1)
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
        
        self.configure(fill=cf["facecolor"], outline=cf["edgecolor"])
        self.coords(*self._req_coords)
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
    
    def resize(
            self,
            xys: tuple[
                IntFloat | None,
                IntFloat | None,
                IntFloat | None,
                IntFloat | None
            ]
    ):
        raise NotImplementedError
    
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
    
    def get_label(self) -> _Text:
        return self._label
    
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
        self._req_transform: _BaseTransform = _FirstOrderPolynomial(0, 1)
        self._update_labels: bool = True
        self._labels: list[_Text] = []
        self._dummy_label: _Text = _Text(
            canvas=self._canvas, text='', tag=f'{self._tag}.labels.text'
        )
    
    def draw(
            self
    ) -> tuple[dict[str, Any], _BaseTransform] | None:
        self._dummy_label.hide()
        if not self._req_enable_labels:
            for label in self._labels:
                label.delete()
            self._labels.clear()
            return
        
        if not self._update_labels:
            return
        
        texts, positions = self._fit_labels()
        canvas = self._canvas
        tag = f'{self._tag}.labels.text'
        font = self._dummy_label._font
        style = self._dummy_label._req_style
        style.pop('text', None)
        bounds = self._dummy_label._req_bounds.copy()
        bounds.pop('xys', None)
        
        length_increase = len(texts) - len(self._labels)
        if length_increase < 0:
            for label in self._labels[length_increase:]:
                label.delete()
            self._labels[:] = self._labels[:-length_increase]
        elif length_increase > 0:
            self._labels.extend(
                _Text(canvas, text='', font=font, tag=tag)
                for i in range(length_increase)
            )
        
        for label, text, xys in zip(self._labels, texts, positions):
            label.set_style(text=text, **style)
            label.set_bounds(xys=xys, **bounds)
            label.draw()
    
    def draw_dummy(self, *args, **kwargs):
        update_labels = self._update_labels
        xys, transform = self._req_xys, self._req_transform
        try:
            self.set_xys_and_transform(*args, **kwargs)
            self._update_labels = True
            texts, positions = self._fit_labels()
        finally:
            self._update_labels = update_labels
            self._req_xys, self._req_transform = xys, transform
        
        label = self._dummy_label
        label.set_style(text=texts[0])
        label.set_bounds(xys=positions[0])
        label.show()
        label.draw()
    
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
    
    def get_labels(self) -> list[_Text]:
        return self._labels
    
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
    
    def set_xys_and_transform(
            self,
            xys: tuple[Dimension, Dimension, Dimension, Dimension] | None = None,
            transform: _BaseTransform | None = None
    ):
        assert isinstance(xys, tuple) and len(xys) == 4, xys
        assert sum( p is None for p in xys ) == 1, xys
        assert xys[self._grow] is None, (self._side, xys)
        assert isinstance(transform, _BaseTransform), transform
        
        if xys is not None and xys != self._req_xys:
            self._req_xys = xys
            self._update_labels = True
        if transform is not None and transform != self._req_transform:
            self._req_transform = transform
            self._update_labels = True
    
    def _fit_labels(
            self
    ) -> tuple[list[str], NDArray[Float]]:
        assert self._update_labels, self._update_labels
        
        x1, y1, x2, y2 = self._req_xys
        transform = self._req_transform
        sci = self._default_style[f"{self._tag}.labels.scientific"] \
            if self._req_scientific is None \
            else self._req_scientific
        
        data = np.asarray(transform.get_x_interval)#TODO
        texts = [
            t.replace('e', '\ne') if 'e' in (t := '{0:.{1}g}'.format(d, sci))
                else t + '\n'
            for d in data
        ]
        
        if self._side in ('t', 'b'):
            positions = [ (x, y1, x, y2) for x in transform(data) ]
        else:  # left or right
            positions = [ (x1, y, x2, y) for y in transform(data) ]
        
        self._update_labels = False
        
        return texts, positions


class _Frame(_BaseRegion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rect: _Rectangle = _Rectangle(
            self._canvas,
            default_style=self._default_style,
            tags=(f'{self._tag}.rect',)
        )
    
    def resize(
            self,
            xys: tuple[
                Dimension | None,
                Dimension | None,
                Dimension | None,
                Dimension | None
            ]
    ):
        self.set_bbox(xys)
    
    def draw(self):
        self._rect.draw()
        self._bbox = self._rect.get_bbox().copy()
    
    def delete(self):
        self._rect.delete()
    
    def set_bbox(
            self,
            xys: tuple[
                Dimension | None,
                Dimension | None,
                Dimension | None,
                Dimension | None
            ]
    ) -> _BBox:
        bbox = self._rect.set_bbox(
            _BBox(xys, sticky='nesw', padx='0p', pady='0p', convert=True)
        )
        self._req_bbox = bbox.copy()
        return self._req_bbox
    
    def get_bbox(self) -> _BBox:
        return self._bbox
    
    def set_facecolor(self, *args, **kwargs) -> str | None:
        return self._rect.set_facecolor(*args, **kwargs)
    
    def get_facecolor(self) -> str:
        return self._rect.get_facecolor()
    
    def set_edgecolor(self, *args, **kwargs) -> str | None:
        return self._rect.set_edgecolor(*args, **kwargs)
    
    def get_edgecolor(self) -> str:
        return self._rect.get_edgecolor()


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
    
    def _on_configure(self, event: tk.Event):
        super()._on_configure(event)
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
        self._x_transform: _BaseTransform
        self._y_transform: _BaseTransform
        self._title: _Text = _Text(self, text='', tag='title.text')
        self._taxis: _Axis = _Axis(self, side='t', tag='taxis')
        self._baxis: _Axis = _Axis(self, side='b', tag='baxis')
        self._laxis: _Axis = _Axis(self, side='l', tag='laxis')
        self._raxis: _Axis = _Axis(self, side='r', tag='raxis')
        self._tticks: _Ticks = _Ticks(self, side='t', tag='tticks')
        self._bticks: _Ticks = _Ticks(self, side='b', tag='bticks')
        self._lticks: _Ticks = _Ticks(self, side='l', tag='lticks')
        self._rticks: _Ticks = _Ticks(self, side='r', tag='rticks')
        '''#TODO
        self._frame: _Frame
        self._frame.set_bbox(self._get_xys_for_frame())
        '''    
        self.set_facecolor()
        self.set_btickslabels(True)
        self.set_ltickslabels(True)
    
    @property
    def _artists(self) -> set:
        artists = []
        for side in 'rblt':
            for arts in getattr(self, f'_{side}artists').values():
                artists.extend(arts)
        return set(artists)
    
    def _on_configure(self, event: tk.Event):
        super()._on_configure(event)
        self.draw()
    
    def draw(self):
        # Get data limits
        dxss, dyss = [], []
        for side in 'rblt':
            if lines := getattr(self, f'_{side}artists')["lines"]:
                dx1, dy1 = np.asarray([ a._xymin for a in lines ]).min(axis=0)
                dx2, dy2 = np.asarray([ a._xymax for a in lines ]).max(axis=0)
            else:
                dx1, dx2 = dy1, dy2 = (1., 1e9)
            dxss.append([dx1, dx2])
            dyss.append([dy1, dy2])
        
        # Draw items and get empty space for frame
        w, h = self._size
        cx1, cy1, cx2, cy2 = (0, 0, w-1, h-1)
        
        ## Draw title
        self._title.set_bounds((cx1, cy1, cx2, cy2))
        self._title.draw()
        ## Update upper bound if title was drawn
        if bbox := self._title.bbox():
            cy1 = bbox[3] + 1
        
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
        
        ## Draw dummy ticks and get the empty space
        _cxys = (cx1, cy1, cx2, cy2)  # backup space for ticks
        for i, (side, dxs, dys) in enumerate(zip('rblt', dxss, dyss)):
            x_tf, y_tf = _get_transforms(dxs, dys, [cx1, cx2], [cy1, cy2])
            xys = list(_cxys)
            xys[i] = None
            ticks = self._get_ticks(side)
            ticks.draw_dummy(tuple(xys), x_tf if side in 'bt' else y_tf)
            ## Update empty space
            if bbox := self._rticks.bbox(dummy=True):
                cx2 = bbox[0] - 1
            if bbox := self._bticks.bbox(dummy=True):
                cy2 = bbox[1] - 1
            if bbox := self._lticks.bbox(dummy=True):
                cx1 = bbox[2] + 1
            if bbox := self._tticks.bbox(dummy=True):
                cy1 = bbox[3] + 1
        
        ## Draw ticks and get the empty space
        for i, (side, dxs, dys) in enumerate(zip('rblt', dxss, dyss)):
            x_tf, y_tf = _get_transforms(dxs, dys, [cx1, cx2], [cy1, cy2])
            xys = [cx1, cy1, cx2, cy2]
            xys[i] = None  # growing bound
            xys[i-2] = _cxys[i-2]  # use previous base bound
            ticks = self._get_ticks(side)
            ticks.set_xys_and_transform(
                tuple(xys), x_tf if side in 'bt' else y_tf
            )
            ticks.draw()
        
        '''#TODO
        self._frame.set_bbox(self._get_xys_for_frame())
        self._frame.draw()
        '''
        
        # Draw user defined artists
        for side, dxs, dys in zip('rblt', dxss, dyss):
            x_tf, y_tf = _get_transforms(dxs, dys, [cx1, cx2], [cy1, cy2])
            for artists in getattr(self, f'_{side}artists').values():
                for artist in artists:
                    artist.set_transforms(x_tf, y_tf)
                    artist.draw()
        
        # Apply zorders
        for tag in sorted(self._zorder_tags.values()):
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
        #TODO: handle t and r
        
        if b is not None:  # b
            x = np.asarray(b, dtype=float)
            x_artists = self._bartists
        else:  # t
            x = np.asarray(t, dtype=float)
            x_artists = self._tartists
        if l is not None:  # l
            y = np.asarray(l, dtype=float)
            y_artists = self._lartists
        else:  # r
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
            tag='line'
        )
        x_lines.append(line)
        y_lines.append(line)
        #TODO: set sides
        
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
        
        self.after_idle(self.draw)
    
    def _on_destroy(self, event: tk.Event | None = None):
        _cleanup_tk_attributes(self)
    
    def _on_theme_changed(self, event: tk.Event):
        self.update_theme()
        self.draw()
    
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
            padx: Dimension | tuple[Dimension, Dimension] = ('1p', '1p'),
            pady: Dimension | tuple[Dimension, Dimension] = ('1p', '1p')
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
    
    fig.after(3000, lambda: root.style.theme_use('cyborg'))
    
    root.mainloop()

