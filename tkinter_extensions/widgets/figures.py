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


class ZorderNotFoundError(RuntimeError):
    pass


class _BaseTransform:
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat]
    ):
        assert isinstance(xs, (IntFloat, np.ndarray)), xs
        if isinstance(xs, (int, float)):
            xs = np.float64(xs)
        dt = xs.dtype
        assert any( np.issubdtype(dt, d) for d in IntFloat.__args__ ), xs.dtype
    
    def get_inverse(self):
        raise NotImplementedError


class _FirstOrderPolynomial(_BaseTransform):
    def __init__(self, c0: IntFloat, c1: IntFloat):
        assert isinstance(c0, IntFloat), c0
        assert isinstance(c1, IntFloat), c1
        self._c0 = np.float64(c0)
        self._c1 = np.float64(c1)
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat]
    ) -> IntFloat | NDArray[Float]:
        super().__call__(xs)
        return self._c0 + self._c1 * xs  # y(x) = c0 + c1 * x
    
    @classmethod
    def from_points(
            cls,
            xs: NDArray[IntFloat],
            ys: NDArray[IntFloat]
    ) -> _FirstOrderPolynomial:
        assert isinstance(xs, np.ndarray), xs
        assert isinstance(ys, np.ndarray), ys
        assert xs.shape == ys.shape, (xs.shape, ys.shape)
        assert xs.size == ys.size == 2, (xs.shape, ys.shape)
        dt = xs.dtype
        assert any( np.issubdtype(dt, d) for d in IntFloat.__args__ ), xs.dtype
        dt = ys.dtype
        assert any( np.issubdtype(dt, d) for d in IntFloat.__args__ ), ys.dtype
        
        # y(x) = c0 + c1 * x
        c1 = (ys[1] - ys[0]) / (xs[1] - xs[0])  # c1 = (y1 - y0) / (x1 - x0)
        c0 = ys[0] - c1 * xs[0]  # c0 = y0 - c1 * x0
        
        return cls(c0=c0, c1=c1)
    
    def get_inverse(self) -> _FirstOrderPolynomial:
        c1_inv = 1. / self._c1  # c1_inv = 1 / c1
        c0_inv = -self._c0 / self._c1  # c0_inv = -c0 / c1
        return _FirstOrderPolynomial(c0=c0_inv, c1=c1_inv)


class _BBox:#TODO: delete
    def __init__(
            self,
            xys: tuple[Dimension, Dimension, Dimension, Dimension],
            sticky: str | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None,
            convert: bool = False
    ):
        root = tk._get_default_root('_BBox.__init__')
        self._xys: tuple[IntFloat, IntFloat, IntFloat, IntFloat]
        self._sticky: str | None
        self._padx: tuple[IntFloat, IntFloat] | None
        self._pady: tuple[IntFloat, IntFloat] | None
        self._to_px = lambda dim: _to_px(root, dim)
        self.set_xys(xys, convert=convert)
        self.set_sticky(sticky)
        self.set_padx(padx, convert=convert)
        self.set_pady(pady, convert=convert)
    
    def __eq__(self, bbox: _BBox):
        assert isinstance(bbox, _BBox), _BBox
        return self.get_all() == bbox.get_all()
    
    def copy(self) -> _BBox:
        return type(self)(**self.get_all())
    
    def set_xys(
            self,
            xys: tuple[Dimension, Dimension, Dimension, Dimension],
            convert: bool = False
    ):
        assert isinstance(xys, tuple), xys
        assert all( isinstance(p, (Dimension, type(None))) for p in xys ), xys
        
        if convert:
            x1, y1, x2, y2 = self._to_px(xys)
        else:
            x1, y1, x2, y2 = xys
        
        if x1 is not None and x2 is not None and x2 < x1:
            x1 = x2 = (x1 + x2) / 2.
        if y1 is not None and y2 is not None and y2 < y1:
            y1 = y2 = (y1 + y2) / 2.
        
        self._xys = (x1, y1, x2, y2)
    
    def get_xys(self) -> tuple[IntFloat, IntFloat, IntFloat, IntFloat]:
        return self._xys
    
    def set_sticky(self, sticky: str | None):
        assert isinstance(sticky, (str, type(None))), sticky
        self._sticky = sticky
    
    def get_sticky(self) -> str | None:
        return self._sticky
    
    def set_padx(
            self,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            convert: bool = False
    ):
        assert isinstance(padx, (Dimension, tuple, type(None))), padx
        
        if isinstance(padx, Dimension):
            padx = (padx, padx)
        elif isinstance(padx, tuple):
            assert len(padx) == 2, padx
            assert all( isinstance(p, Dimension) for p in padx ), padx
        if convert:
            padx = self._to_px(padx)
        
        self._padx = padx
    
    def get_padx(self) -> tuple[IntFloat, IntFloat] | None:
        return self._padx
    
    def set_pady(
            self,
            pady: Dimension | tuple[Dimension, Dimension] | None = None,
            convert: bool = False
    ):
        assert isinstance(pady, (Dimension, tuple, type(None))), pady
        
        if isinstance(pady, Dimension):
            pady = (pady, pady)
        elif isinstance(pady, tuple):
            assert len(pady) == 2, pady
            assert all( isinstance(p, Dimension) for p in pady ), pady
        if convert:
            pady = self._to_px(pady)
        
        self._pady = pady
    
    def get_pady(self) -> tuple[IntFloat, IntFloat] | None:
        return self._pady
    
    def get(self) -> tuple[IntFloat, IntFloat, IntFloat, IntFloat]:
        return self.get_xys()
    
    def get_all(self) -> dict[str, Any]:
        return {
            "xys": self.get_xys(),
            "sticky": self.get_sticky(),
            "padx": self.get_padx(),
            "pady": self.get_pady()
        }


class _BaseElement:
    def __init__(self, canvas: _Plot | _Suptitle, tag: str = ''):
        assert isinstance(canvas, (_Plot, _Suptitle)), canvas
        assert isinstance(tag, str), tag
        
        root = canvas._root()
        self._canvas: _Plot | _Suptitle = canvas
        self._figure: Figure = canvas._figure
        self._to_px = lambda dim: _to_px(root, dim)
        self._tag: str = tag
        self._default_style: dict[str, Any]
        self._root_default_style: dict[str, Any]
        self.update_theme()
    
    def __del__(self):
        _cleanup_tk_attributes(self)
    
    def update_theme(self):
        raise NotImplementedError
    
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
    
    def __del__(self):
        try:
            self.delete()
        except tk.TclError:
            pass
        self._canvas._zorder_tags.pop(self, None)
    
    def update_theme(self):
        self._root_default_style = self._figure._default_style[self._name]
        self._default_style = self._figure._default_style[self._tag]
        self._stale = True
    
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
    
    def set_transforms(
            self,
            x_transform: _BaseTransform | None = None,
            y_transform: _BaseTransform | None = None,
    ):
        assert isinstance(x_transform, _BaseTransform), x_transform
        assert isinstance(y_transform, _BaseTransform), y_transform
        
        if x_transform is not None and x_transform != self._req_transforms[0]:
            self._req_transforms[0] = x_transform
            self._stale = True
        if y_transform is not None and y_transform != self._req_transforms[1]:
            self._req_transforms[1] = y_transform
            self._stale = True
    
    def get_transforms(self) -> tuple[_BaseTransform, _BaseTransform]:
        return self._req_transforms
    
    def set_coords(self, *ps: Dimension):
        assert all( isinstance(p, (Dimension, type(None))) for p in ps ), ps
        
        if ps and ps != self._req_coords:
            self._req_coords[:] = ps
            self._stale = True
    
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
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        if self.cget('text'):
            return super().bbox()
        return None
    
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
        self._stale = False
    
    def set_bounds(
            self,
            xys: tuple[Dimension, Dimension, Dimension, Dimension] | None = None,
            sticky: str | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None
    ):
        assert isinstance(xys, (tuple, type(None))), xys
        assert all( isinstance(p, (Dimension, type(None))) for p in xys ), xys
        assert isinstance(sticky, (str, type(None))), sticky
        assert isinstance(padx, (Dimension, tuple, type(None))), padx
        assert isinstance(pady, (Dimension, tuple, type(None))), pady
        
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
            default_color: str = '',
            **kwargs
    ):
        super().__init__(canvas=canvas, **kwargs)
        self._xymin: NDArray[Float]
        self._xymax: NDArray[Float]
        self._req_xy: NDArray[Float] = np.array([[]], dtype=float)
        self._id: int = self._canvas.create_line(
            -1, -1, -1, -1, fill='', width='0p', tags=self._tags
        )
        self.set_default_color(default_color=default_color)
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
            xys = np.asarray([  # x0, y0, x1, y1, x2, y2, ...
                x_transform(self._req_xy[0])[:, None],
                y_transform(self._req_xy[1])[:, None]
            ]).ravel()
        
        self.coords(*xys)
        self._update_zorder()
        self._stale = False
    
    def set_default_color(self, color: str | None = None):
        assert isinstance(color, (str, type(None))), color
        
        if color is not None and color != self._default_color:
            self._default_color = color
            self._stale = True
    
    def get_default_color(self) -> str:
        return self._default_color
    
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
            self._xymin = self._req_xy.min(axis=0)
            self._xymax = self._req_xy.max(axis=0)
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
    def update_theme(self):
        self._default_style = self._figure._default_style
    
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


class _Title(_BaseRegion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text: _Text = _Text(self._canvas, text='', tag=f'{self._tag}.text')
    
    def resize(
            self,
            xys: tuple[
                Dimension | None,
                Dimension | None,
                Dimension | None,
                Dimension | None
            ]
    ):
        self.set_bounds(xys=xys)
    
    def draw(self):
        self._text.draw()
    
    def delete(self):
        self._text.delete()
    
    def get_text(self) -> _Text:
        return self._text
    
    def set_bounds(self, *args, **kwargs):
        self._text.set_bounds(*args, **kwargs)
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._text.bbox()
    
    def set_style(self, *args, **kwargs):
        self._text.set_style(*args, **kwargs)
    
    def get_style(self) -> dict[str, Any]:
        return self._text.get_style()


class _Axis(_BaseRegion):#TODO
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            side: Literal['t', 'b', 'l', 'r'],
            **kwargs
    ):
        super().__init__(canvas=canvas, **kwargs)
        self._side: Literal['t', 'b', 'l', 'r'] = side
        self._extend: int = dict([('r', 0), ('b', 1), ('l', 2), ('t', 3)])[
            side
        ]
        self._title: _Text = _Text(
            self._canvas,
            default_style=self._default_style["title"]["text"],
            tags=(f'{self._tag}.title.text',)
        )
        self._ticks: list[_Text] = []
        self._req_title_text: dict[str, Any] = {}
        self._req_title_font: dict[str, Any] = {}
        self._req_title_bbox: dict[str, Any] = {}
        self._req_ticks: bool = False
        self._req_ticks_text: dict[str, Any] = {}
        self._req_ticks_font: dict[str, Any] = {}
        self._req_ticks_bbox: dict[str, Any] = {}
        self._req_ticks_scientific: Int | None = None
        self._ticks_values: NDArray[Float] | None = None
        self._ticks_positions: list[tuple[IntFloat]] | None = None
    
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
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self._title.update_theme(self._default_style["title"]["text"])
        for tick in self._ticks:
            tick.update_theme(self._default_style["tick"]["text"])
    
    def draw(self):
        bbox = _BBox(self._req_bbox.get(), **self._req_title_bbox)
        title = self._title
        title.set_bbox(bbox)
        title.set_font(**self._req_title_font)
        title.set_text(**self._req_title_text)
        title.draw()
        
        for tick in self._ticks:
            tick.delete()
        self._ticks.clear()
        
        if self._ticks_values is None:
            xys = title.get_bbox().get()
            self._bbox = _BBox(xys)
        else:
            default_style = self._default_style["tick"]["text"]
            text_cf = self._req_ticks_text
            font_cf = self._req_ticks_font
            bbox_cf = self._req_ticks_bbox
            scientific = self._default_style["tick"]["scientific"] \
                if self._req_ticks_scientific is None \
                else self._req_ticks_scientific
            tags = (f'{self._tag}.tick.text',)
            font = Font()
            
            for value, xys in zip(self._ticks_values, self._ticks_positions):
                text = '{0:.{1}g}'.format(value, scientific).replace('e', '\ne')
                tick = _Text(
                    self._canvas,
                    font=font,
                    default_style=default_style,
                    tags=tags
                )
                tick.set_font(**font_cf)
                tick.set_text(text=text, **text_cf)
                tick.set_bbox(_BBox(xys, **bbox_cf))
                tick.draw()
                self._ticks.append(tick)
    
    def delete(self):
        self._title.delete()
        for tick in self._ticks:
            tick.delete()
    
    def set_bbox(
            self, xys: tuple[IntFloat, IntFloat, IntFloat, IntFloat]
    ) -> _BBox:
        assert xys[self._extend] is None, (self._side, xys)
        self._req_bbox = _BBox(xys)
        return self._req_bbox
    
    def get_bbox(self) -> _BBox:
        return self._bbox
    
    def set_title(
            self,
            text: str = '',
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
    ) -> dict[str, Any]:
        req_title_text = {
            "text": text,
            "color": color,
            "angle": angle
        }
        req_title_font = {
            "family": family,
            "size": size,
            "weight": weight,
            "slant": slant,
            "underline": underline,
            "overstrike": overstrike
        }
        req_title_bbox = {
            "sticky": sticky,
            "padx": padx,
            "pady": pady
        }
        self._req_title_text = req_title_text
        self._req_title_font = req_title_font
        self._req_title_bbox = req_title_bbox
        
        return self._req_title_text, self._req_title_font, self._req_title_bbox
    
    def get_title(self) -> _Text:
        return self._text
    
    def set_ticks(
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
    ) -> dict[str, Any]:
        req_ticks_text = {
            "color": color,
            "angle": angle
        }
        req_ticks_font = {
            "family": family,
            "size": size,
            "weight": weight,
            "slant": slant,
            "underline": underline,
            "overstrike": overstrike
        }
        req_ticks_bbox = {
            "padx": padx,
            "pady": pady
        }
        self._req_ticks = enable
        self._req_ticks_text = req_ticks_text
        self._req_ticks_font = req_ticks_font
        self._req_ticks_bbox = req_ticks_bbox
        self._req_ticks_scientific = scientific
        
        return (
            self._req_ticks,
            self._req_ticks_text,
            self._req_ticks_font,
            self._req_ticks_bbox,
            self._req_ticks_scientific
        )
    
    def get_ticks(self) -> list[_Text]:
        return self._ticks
    
    def _update_bbox(self) -> _BBox:
        self.draw()
        self._ticks_values = self._ticks_positions = None
        
        if not self._req_ticks:
            return
        
        tx1, ty1, tx2, ty2 = self._title.get_bbox().get()
        
        tick = _Text(
            self._canvas,
            text='0123456789',
            default_style=self._default_style["tick"]["text"],
            tags=(f'{self._tag}.tick.text',)
        )
        tick.set_font(**self._req_ticks_font)
        tick.set_text(**self._req_ticks_text)
        tick.set_bbox(_BBox((0, 0, 0, 0), **self._req_ticks_bbox))
        tick.draw()
        tick.delete()
        kx1, ky1, kx2, ky2 = tick.get_bbox().get()
        kw, kh = (kx2 - kx1 + 1), (ky2 - ky1 + 1)
        
        if self._side == 'r':
            xys = (tx1 - kw, ty1, tx2, ty2)
        elif self._side == 'b':
            xys = (tx1, ty1 - kh, tx2, ty2)
        elif self._side == 'l':
            xys = (tx1, ty1, tx2 + kw, ty2)
        else:
            xys = (tx1, ty1, tx2, ty2 + kh)
        
        self._bbox = _BBox(xys)
    
    def _set_ticks_values(
            self,
            values: NDArray[Float],
            positions: NDArray[Float]
    ):
        assert self._req_ticks, self._req_ticks
        assert isinstance(values, np.ndarray), (type(values), values)
        assert np.issubdtype(values.dtype, np.floating), values.dtype
        assert isinstance(positions, np.ndarray), (type(positions), positions)
        assert np.issubdtype(positions.dtype, np.floating), positions.dtype
        assert values.shape == positions.shape, (values.shape, positions.shape)
        assert values.ndim == 1, values.ndim
        
        tx1, ty1, tx2, ty2 = self._title.get_bbox().get()
        if self._side == 'r':
            positions = [ (None, p, tx1 - 1, p) for p in positions ]
        elif self._side == 'b':
            positions = [ (p, None, p, ty1 - 1) for p in positions ]
        elif self._side == 'l':
            positions = [ (tx2 + 1, p, None, p) for p in positions ]
        else:
            positions = [ (p, ty2 + 1, p, None) for p in positions ]
        
        self._ticks_values = values
        self._ticks_positions = positions


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
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self._rect.update_theme(self._default_style)
    
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
        self._default_style: dict[str, Any]
        self._root_default_style: dict[str, Any]
        self._zorder_tags: dict[_BaseArtist, str] = {}
        self._size: tuple[int, int] = (
            self.winfo_reqwidth(), self.winfo_reqheight()
        )
        self._req_facecolor: str | None = None
        self._update_theme()
        self.bind('<Configure>', self._on_configure, add=True)
    
    def _on_configure(self, event: tk.Event):
        self._size = (event.width, event.height)
    
    def _update_theme(self):
        self._default_style = self._figure._default_style
    
    def update_theme(self):
        self._update_theme()
        self.set_facecolor(self._req_facecolor)
    
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
        self._title: _Title = _Title(self, tag=f'{self._tag}')
        self.set_facecolor()
    
    def _on_configure(self, event: tk.Event):
        super()._on_configure(event)
        w, h = self._size
        self._title.resize((0, 0, w-1, None))
        self._title.draw()
    
    def update_theme(self):
        super().update_theme()
        self._title.update_theme()
        event = tk.Event()
        event.width, event.height = self._size
        self._on_configure(event)
    
    def draw(self):
        self._title.draw()
        x1, y1, x2, y2 = self._title.bbox()
        self.configure(width=x2 - x1, height=y2 - y1)
         # triggers `self._on_configure`
    
    def get_title(self) -> _Title:
        return self._title
    
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
        self._title.set_bounds(
            (0, 0, w-1, None), sticky=sticky, padx=padx, pady=pady
        )
    
    def set_style(self, *args, **kwargs):
        return self._title.set_style(*args, **kwargs)
    
    def get_style(self) -> dict[str, Any]:
        return self._title.get_style()


class _Plot(_BaseSubwidget, tk.Canvas):
    _tag: str = 'plot'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._artists: dict[str, list[_BaseArtist]] = {"line": []}
        self._x_transform: _FirstOrderPolynomial
        self._y_transform: _FirstOrderPolynomial
        self._req_facecolor: str | None
        self._taxis: _Axis
        self._baxis: _Axis
        self._laxis: _Axis
        self._raxis: _Axis
        
        self._title: _Title = _Title(
            self,
            default_style=self._root_default_style["title"],
            tag=f'{self._tag}.title'
        )
        for side in ['t', 'b', 'l', 'r']:
            name = f'{side}axis'
            axis: _Axis = _Axis(
               self,
               side=side,
               default_style=self._root_default_style[name],
               tag=f'{self._tag}.{name}'
            )
            setattr(self, f'_{name}', axis)
            self._set_axislabel(side)
        self._frame: _Frame = _Frame(
            self,
            default_style=self._root_default_style["frame"],
            tag=f'{self._tag}.frame'
        )
        self._frame.set_bbox(self._get_xys_for_frame())
        self.set_facecolor()
        self.set_title('')
    
    def _on_configure(self, event: tk.Event):
        super()._on_configure(event)
        w, h = self._size
        
        self._title.resize((0, 0, w-1, None))
        self._title.draw()
        
        if hasattr(self, '_legend'):
            self._legend.resize(self._get_xys_for_legend())
            self._legend.draw()
        
        for side in ['t', 'b', 'l', 'r']:
            axis = self._get_axis(side)
            axis.resize(
                self._get_xys_for_axis(side, draw_dependencies=False)
            )
        
        self.draw()
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self.set_facecolor(self._req_facecolor)
        
        self._title.update_theme(self._root_default_style["title"])
        
        if hasattr(self, '_legend'):
            self._legend.update_theme(self._root_default_style["legend"])
        
        for side in ['t', 'b', 'l', 'r']:
            self._get_axis(side).update_theme(
                self._root_default_style[f'{side}axis']
            )
        
        self._frame.update_theme(self._root_default_style["frame"])
        
        for artists in self._artists.values():
            for artist in artists:
                artist.update_theme(self._root_default_style[artist._name])
        
        event = tk.Event()
        event.width, event.height = self._size
        self._on_configure(event)
    
    def draw(self):#TODO: transform
        self._title.draw()
        
        self._baxis.set_ticks(True)#???
        for side in ['t', 'b', 'l', 'r']:
            self._get_axis(side)._update_bbox()
        cx1, cy1, cx2, cy2 = self._get_xys_for_frame()
        dx1, dy1 = np.asarray(
            [ a._xymin for a in self._artists["line"] ]
        ).min(axis=0)
        dx2, dy2 = np.asarray(
            [ a._xymax for a in self._artists["line"] ]
        ).max(axis=0)
        self._x_transform = x_tf = _FirstOrderPolynomial.from_points(
            np.array([dx1, dx2]), np.array([cx1, cx2])
        )
        self._y_transform = y_tf = _FirstOrderPolynomial.from_points(
            np.array([dy1, dy2]), np.array([cy1, cy2])
        )
        ticks_values = np.asarray([dx1, dx2])
        ticks_positions = x_tf(ticks_values)
        if self._baxis._req_ticks is not None:
            self._baxis._set_ticks_values(ticks_values, ticks_positions)
            self._baxis.draw()
        
        self._frame.set_bbox(self._get_xys_for_frame())
        self._frame.draw()
        
        for artists in self._artists.values():
            for artist in artists:
                artist.draw()
        
        for tag in sorted(self._zorder_tags.values()):
            self.tag_raise(tag)
    
    def _create_color_cycle(self) -> Cycle:
        return Cycle(self._root_default_style["colors"])
    
    def set_facecolor(self, color: str | None = None) -> str | None:
        assert isinstance(color, (str, type(None))), color
        
        # Get default style and update it with the new values
        default_color = self._root_default_style["facecolor"]
        new_color = default_color if color is None else color
        self.configure(bg=new_color)
        self._req_facecolor = color
        
        return self._req_facecolor
    
    def get_facecolor(self) -> str:
        return self["background"]
    
    def set_title(
            self,
            text: str = '',
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
    ) -> _Title:
        if sticky is not None and ('s' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for title must not include 's' and not equal to "
                f"'center' but got {sticky}."
            )
        
        w, h = self._size
        self._title.set_text(
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
        self._title.set_bbox(
            xys=(0, 0, w-1, None),
            sticky=sticky,
            padx=padx,
            pady=pady
        )
        
        return self._title
    
    def get_title(self) -> _Title:
        return self._title
    
    def _get_xys_for_legend(
            self
    ) -> tuple[
        Dimension | None, Dimension | None, Dimension | None, Dimension | None
    ]:
        raise NotImplementedError
    
    def _get_xys_for_axis(
            self,
            side: Literal['t', 'b', 'l', 'r'],
            draw_dependencies: bool = True
    ) -> tuple[
        Dimension | None, Dimension | None, Dimension | None, Dimension | None
    ]:
        def _get_boundary(
                which: Literal['_title', '_legend']
        ) -> Dimension | None:
            if (region := getattr(self, which, None)) is None:
                return
            
            assert which in ('_title', '_legend'), which
            
            if draw_dependencies:
                region.draw()
            _x1, _y1, _x2, _y2 = region.get_bbox().get()
            
            if which in '_title':
                assert _y2 is not None, _y2
                return _y2 + 1  # new y1
            assert _x1 is not None, _x1
            return _x1 - 1  # new x2
        #
        
        assert side in ('t', 'b', 'l', 'r'), side
        
        w, h = self._size
        x1, y1, x2, y2 = (0, 0, w-1, h-1)
        if side == 't':  # top axis
            y2 = None
            y1 = _get_boundary('_title')
        elif side == 'b':  # bottom axis
            y1 = None
        elif side == 'l':  # left axis
            x2 = None
            y1 = _get_boundary('_title')
        else:  # right axis
            x1 = None
            y1 = _get_boundary('_title')
            if (x2_bound := _get_boundary('_legend')) is not None:
                x2 = x2_bound
        
        return (x1, y1, x2, y2)
    
    def _get_xys_for_frame(
            self,
    ) -> tuple[
        Dimension | None, Dimension | None, Dimension | None, Dimension | None
    ]:
        x1, _, _, _ = self._raxis.get_bbox().get()
        _, y1, _, _ = self._baxis.get_bbox().get()
        _, _, x2, _ = self._laxis.get_bbox().get()
        _, _, _, y2 = self._taxis.get_bbox().get()
        
        return (x2+1, y2+1, x1-1, y1-1)
    
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
            text: str = '',
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
    ) -> _Axis:
        axis = self._get_axis(side)
        axis.set_title(
            text=text,
            color=color,
            angle=angle,
            family=family,
            size=size,
            weight=weight,
            slant=slant,
            underline=underline,
            overstrike=overstrike,
            sticky=sticky,
            padx=padx,
            pady=pady
        )
        return axis
    
    def set_tlabel(self, *args, sticky: str | None = None, **kwargs) -> _Axis:
        if sticky is not None and ('s' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for taxis label must not include 's' and not equal to "
                f"'center' but got {sticky}."
            )
        return self._set_axislabel('t', *args, sticky=sticky, **kwargs)
    
    def set_blabel(self, *args, sticky: str | None = None, **kwargs) -> _Axis:
        if sticky is not None and ('n' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for baxis label must not include 'n' and not equal to "
                f"'center' but got {sticky}."
            )
        return self._set_axislabel('b', *args, sticky=sticky, **kwargs)
    
    def set_llabel(self, *args, sticky: str | None = None, **kwargs) -> _Axis:
        if sticky is not None and ('e' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for laxis label must not include 'e' and not equal to "
                f"'center' but got {sticky}."
            )
        return self._set_axislabel('l', *args, sticky=sticky, **kwargs)
    
    def set_rlabel(self, *args, sticky: str | None = None, **kwargs) -> _Axis:
        if sticky is not None and ('w' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for raxis label must not include 'w' and not equal to "
                f"'center' but got {sticky}."
            )
        return self._set_axislabel('r', *args, sticky=sticky, **kwargs)
    
    def _get_axislabel(self, side: Literal['t', 'b', 'l', 'r']) -> _Title:
        return self._get_axis(side)._title
    
    def get_tlabel(self, *args, **kwargs) -> _Axis:
        return self._get_axislabel('t', *args, **kwargs)
    
    def get_blabel(self, *args, **kwargs) -> _Axis:
        return self._get_axislabel('b', *args, **kwargs)
    
    def get_llabel(self, *args, **kwargs) -> _Axis:
        return self._get_axislabel('l', *args, **kwargs)
    
    def get_rlabel(self, *args, **kwargs) -> _Axis:
        return self._get_axislabel('r', *args, **kwargs)
    
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
        
        x = np.asarray(b, dtype=float)
        y = np.asarray(l, dtype=float)
        assert x.shape == y.shape, [x.shape, y.shape]
        
        cycle = self._create_color_cycle()
        
        lines = self._artists["line"]
        for line in lines:
            line.set_default_color(next(cycle))
        
        line = _Line(
            self,
            x=x.ravel(),
            y=y.ravel(),
            default_color=next(cycle),
            color=color,
            width=width,
            smooth=smooth,
            default_style=self._root_default_style["line"],
            tags=(f'{self._tag}.line',)
        )
        lines.append(line)
        
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
            text: str = '',
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
        if text:  # enable suptitle
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
            [ _Plot(self, default_style=self._default_style["plot"])
              for c in range(n_cols) ]
            for r in range(n_rows)
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
                self.grid_rowconfigure(0, weight=1)
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
    
    x = np.arange(48000. * 10.) / 3.
    y = x
    
    fig = Figure(root, toolbar=True)
    fig.pack(fill='both', expand=True)
    
    suptitle = fig.set_suptitle('<Suptitle>')
    '''
    plt = fig.set_plots(1, 1)
    plt.plot(x, y)
    plt.set_title('<Title>')
    plt.set_tlabel('<top-label>')
    plt.set_blabel('<bottom-label>')
    plt.set_llabel('<left-label>')
    plt.set_rlabel('<right-label>')
    '''
    fig.after(3000, lambda: root.style.theme_use('cyborg'))
    
    root.mainloop()

