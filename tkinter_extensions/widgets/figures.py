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

import numpy as np
from numpy.typing import NDArray, ArrayLike
import ttkbootstrap as ttk

from .. import utils
from .. import variables as vrb
from ..constants import Int, IntFloat, Float
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


def _get_sticky_p(
        direction: Literal['x', 'y'],
        start: Int,
        stop: Int,
        sticky: str,
        pad: Int | tuple[Int, Int]
) -> tuple[Int, Literal['n', 'e', 's', 'w', '']]:  # returns (position, anchor)
    assert direction in ('x', 'y'), direction
    assert isinstance(start, Int) and isinstance(stop, Int), (start, stop)
    assert start <= stop, (start, stop)
    assert isinstance(sticky, str), sticky
    assert sticky != '', sticky
    assert sticky == 'center' or set(sticky).issubset('nesw'), sticky
    assert isinstance(pad, (Int, tuple)), pad
    
    lower, upper = ('w', 'e') if direction == 'x' else ('n', 's')
    if isinstance(pad, Int):
        pad = (pad, pad)
    else:  # tuple
        assert len(pad) == 2 and all( isinstance(p, Int) for p in pad ), pad
    
    start += pad[0]
    stop = max(stop - pad[1], start)
    
    if sticky == 'center':
        return (start + stop) // 2, ''
    
    if lower in sticky:
        if upper in sticky:
            return (start + stop) // 2, ''
        else:
            return start, lower
    else:
        if upper in sticky:
            return stop, upper
        else:
            return (start + stop) // 2, ''


def _get_sticky_xy(
    bbox: _CanvasBBox
) -> tuple[Int, Int, Literal['n', 'e', 's', 'w', '']]:  # returns (x, y, anchor)
    x1, y1, x2, y2 = bbox.get_xys()
    sticky = bbox.get_sticky()
    x, anchor_x = _get_sticky_p(
        'x', x1, x2, sticky=sticky, pad=bbox.get_padx()
    )
    y, anchor_y = _get_sticky_p(
        'y', y1, y2, sticky=sticky, pad=bbox.get_pady()
    )
    if (anchor := anchor_y + anchor_x) == '':
        anchor = 'center'
    return (x, y, anchor)


def _get_xy_trasforms(
        inp: _BaseBBox, out: _BaseBBox
) -> tuple[_FirstOrderPolynomial]:
    assert isinstance(inp, _BBox) or isinstance(out, _BBox), (inp, out)
    assert isinstance(inp, _CanvasBBox) or isinstance(out, _CanvasBBox), (inp, out)
    
    ix1, iy1, ix2, iy2 = inp.get()
    ox1, oy1, ox2, oy2 = out.get()
    
    return (
        _FirstOrderPolynomial.from_points(
            np.array([ix1, ix2], dtype=float),
            np.array([ox1, ox2], dtype=float)
        ),
        _FirstOrderPolynomial.from_points(
            np.array([iy1, iy2], dtype=float),
            np.array([oy1, oy2], dtype=float)
        )
    )


class ZorderNotFoundError(RuntimeError):
    pass


class _BaseTransformation:
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat]
    ):
        assert isinstance(xs, (IntFloat, np.ndarray)), xs
        dt = xs.dtype
        assert any( np.issubdtype(dt, d) for d in IntFloat.__args__ ), xs.dtype
    
    def get_inverse(self):
        raise NotImplementedError


class _FirstOrderPolynomial(_BaseTransformation):
    def __init__(self, c0: IntFloat, c1: IntFloat):
        assert isinstance(c0, IntFloat), c0
        assert isinstance(c1, IntFloat), c1
        self._c0 = c0
        self._c1 = c1
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat]
    ) -> IntFloat | NDArray[IntFloat]:
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


class _BaseBBox:
    def get(self) -> tuple[IntFloat, IntFloat, IntFloat, IntFloat]:
        return self._xys


class _BBox(_BaseBBox):
    def __init__(
            self,
            xys: tuple[IntFloat, IntFloat, IntFloat, IntFloat]
    ):
        assert isinstance(xys, tuple), xys
        assert all( isinstance(p, IntFloat) for p in xys ), xys
        x1, y1, x2, y2 = xys
        
        if x2 < x1:
            x1 = x2 = (x1 + x2) / 2.
        if y2 < y1:
            y1 = y2 = (y1 + y2) / 2.
        
        self._xys: tuple[IntFloat, IntFloat, IntFloat, IntFloat] \
            = (x1, y1, x2, y2)
    
    def __eq__(self, bbox: _BaseBBox):
        assert isinstance(bbox, _BaseBBox), bbox
        return type(bbox) == type(self) and self.get() == bbox.get()
    
    def copy(self) -> _BBox:
        return type(self)(self.get())


class _CanvasBBox(_BaseBBox):
    def __init__(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None],
            sticky: str | None = None,
            padx: Int | tuple[Int, Int] | None = None,
            pady: Int | tuple[Int, Int] | None = None
    ):
        self._xys: tuple[Int | None, Int | None, Int | None, Int | None]
        self._sticky: str | None
        self._padx: Int | tuple[Int, Int] | None
        self._pady: Int | tuple[Int, Int] | None
        self.set_xys(xys)
        self.set_sticky(sticky)
        self.set_padx(padx)
        self.set_pady(pady)
    
    def __eq__(self, bbox: _CanvasBBox):
        assert isinstance(bbox, _CanvasBBox), bbox
        return self.get_all() == bbox.get_all()
    
    def copy(self) -> _CanvasBBox:
        return type(self)(**self.get_all())
    
    def set_xys(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None]
    ):
        assert isinstance(xys, tuple), xys
        assert all( isinstance(p, (Int, type(None))) for p in xys ), xys
        x1, y1, x2, y2 = xys
        assert x1 is not None or x2 is not None, xys
        assert y1 is not None or y2 is not None, xys
        
        if x1 is not None and x2 is not None and x2 < x1:
            x1 = x2 = (x1 + x2) // 2
        if y1 is not None and y2 is not None and y2 < y1:
            y1 = y2 = (y1 + y2) // 2
        
        self._xys = (x1, y1, x2, y2)
    
    def get_xys(self) -> tuple[Int | None, Int | None, Int | None, Int | None]:
        return self._xys
    
    def set_sticky(self, sticky: str | None):
        assert isinstance(sticky, (str, type(None))), sticky
        self._sticky = sticky
    
    def get_sticky(self) -> str | None:
        return self._sticky
    
    def set_padx(self, padx: Int | tuple[Int, Int] | None):
        assert isinstance(padx, (Int, tuple, type(None))), padx
        
        if isinstance(padx, Int):
            padx = (padx, padx)
        elif isinstance(padx, tuple):
            assert len(padx) == 2, padx
            assert all( isinstance(p, Int) for p in padx ), padx
        
        self._padx = padx
    
    def get_padx(self) -> tuple[Int, Int] | None:
        return self._padx
    
    def set_pady(self, pady: Int | tuple[Int, Int] | None):
        assert isinstance(pady, (Int, tuple, type(None))), pady
        
        if isinstance(pady, Int):
            pady = (pady, pady)
        elif isinstance(pady, tuple):
            assert len(pady) == 2, pady
            assert all( isinstance(p, Int) for p in pady ), pady
        
        self._pady = pady
    
    def get_pady(self) -> tuple[Int, Int] | None:
        return self._pady
    
    def get_all(self) -> dict[str, Any]:
        return {
            "xys": self.get_xys(),
            "sticky": self.get_sticky(),
            "padx": self.get_padx(),
            "pady": self.get_pady()
        }


class _BaseElement:
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            default_style: dict[str, Any] | None = None
    ):
        assert isinstance(canvas, (_Plot, _Suptitle)), canvas
        assert isinstance(default_style, (dict, type(None))), default_style
        
        self._canvas: _Plot | _Suptitle = canvas
        self._figure: Figure = canvas._figure
        self._default_style: dict[str, Any]
        self._bbox: _CanvasBBox = _CanvasBBox((0, 0, 0, 0))
        self._req_bbox: _CanvasBBox = _CanvasBBox((0, 0, 0, 0))
        self._update_theme(default_style)
    
    def __del__(self):
        _cleanup_tk_attributes(self)
    
    def _update_theme(self, default_style: dict[str, Any]):
        self._default_style = default_style.copy()
        self._root_default_style = self._figure._default_style.copy()
    
    def update_theme(self, default_style: dict[str, Any]):
        self._update_theme(default_style=default_style)
    
    def draw(self):
        raise NotImplementedError


# =============================================================================
# ---- Figure Artists
# =============================================================================
class _BaseArtist(_BaseElement):
    _name: str
    
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            default_style: dict[str, Any] | None = None,
            tags: tuple[str, ...] = ()
    ):
        super().__init__(canvas=canvas, default_style=default_style)
        assert isinstance(tags, tuple) and len(tags) >= 1, tags
        
        tag_list = []
        for tag in tags:
            if '.' not in tag:
                tag_list.append(tag)
                continue
            subtags = tag.split('.')
            tag_list.extend(
                '.'.join(subtags[:i]) for i in range(1, len(tag))
            )
        tag_list.append(self._name)
        tags = tuple(dict.fromkeys(tag_list))
        
        self._tags = tags
        self._req_bbox: _CanvasBBox = _CanvasBBox((0, 0, 0, 0))
        self._req_zorder: float | None = None
        self._stale: bool = True
    
    def __del__(self):
        self._canvas._zorders.pop(self, None)
    
    def _update_theme(self, default_style: dict[str, Any]):
        super()._update_theme(default_style)
        self._stale = True
    
    def set_bbox(self, bbox: _CanvasBBox) -> _CanvasBBox:
        assert isinstance(bbox, _CanvasBBox), bbox
        
        if bbox != self._req_bbox:
            self._req_bbox = bbox.copy()
            self._stale = True
        
        return self._req_bbox
    
    def get_bbox(self) -> _CanvasBBox:
        return self._bbox
    
    def set_zorder(self, zorder: IntFloat | None = None) -> float:
        assert isinstance(zorder, IntFloat), zorder
        
        if zorder != self._req_zorder:
            self._req_zorder = float(zorder)
            self._stale = True
        
        return self._req_zorder
    
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
        self._canvas._zorders[self] = new_tag


class _Text(_BaseArtist):
    _name = 'text'
    
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            text: str = '',
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            **kwargs
    ):
        super().__init__(canvas=canvas, **kwargs)
        self._req_font: dict[str, Any] = {}
        self._req_text: dict[str, Any] = {}
        self._font: Font = Font()
        self._id: Int = canvas.create_text(
            0, 0, anchor='se', text=text, font=self._font, tags=self._tags
        )
        self.set_font(
            family=family,
            size=size,
            weight=weight,
            slant=slant,
            underline=underline,
            overstrike=overstrike
        )
        self.set_text(text=text, color=color, angle=angle)
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self.set_font(**self._req_font)
        self.set_text(**self._req_text)
        self.set_bbox(self._req_bbox)
    
    def draw(self):
        if not self._stale:
            return
        
        # Update font
        defaults = self._default_style
        cf = self._req_font.copy()
        cf.update({ k: defaults[k] for k, v in cf.items() if v is None })
        self._font.configure(**cf)
        
        # Update text style
        cf = self._req_text.copy()
        cf.update({ k: defaults[k] for k, v in cf.items() if v is None })
        self._canvas.itemconfigure(
            self._id, text=cf["text"], fill=cf["color"], angle=cf["angle"]
        )
        
        # Update position
        cf = self._req_bbox.get_all()
        cf.update({ k: defaults[k] for k, v in cf.items() if v is None })
        
        ## Get text size
        if not self._canvas.itemcget(self._id, 'text'):  # empty
            itw, ith = (0, 0)
        else:
            itx1, ity1, itx2, ity2 = self._canvas.bbox(self._id)
            itw, ith = (itx2 - itx1), (ity2 - ity1)
        
        x1, y1, x2, y2 = cf["xys"]
        sticky, padx, pady = cf["sticky"], cf["padx"], cf["pady"]
        if x1 is None:
            x1 = x2 - itw - sum(padx)
        elif x2 is None:
            x2 = x1 + itw + sum(padx)
        if y1 is None:
            y1 = y2 - ith - sum(pady)
        elif y2 is None:
            y2 = y1 + ith + sum(pady)
        bbox = _CanvasBBox((x1, y1, x2, y2), sticky=sticky, padx=padx, pady=pady)
        
        # `anchor` must be 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw', or 'center'
        x, y, anchor = _get_sticky_xy(bbox)
        if anchor != 'center':
            # Roll the anchor. e.g. 0 deg => 1 step, 45 deg => 2 step, ...
            angle = float(self._canvas.itemcget(self._id, 'angle'))
            assert 0.0 <= angle < 360.0, angle
            shift = int((angle + 22.5) // 45)  # 8 directions with a 45 deg step
            mapping = dict(zip(_anchors, _anchors[shift:] + _anchors[:shift]))
            anchor = ''.join( mapping[x] for x in anchor )  # rolling
        
        self._canvas.coords(self._id, x, y)  # update position
        self._canvas.itemconfigure(self._id, anchor=anchor)  # update anchor
        self._update_zorder()
        self._bbox = bbox
        self._stale = False
    
    def set_font(
            self,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None
    ) -> dict[str, Any]:
        # Update None with default values
        req_font = {
            "family": family,
            "size": size,
            "weight": weight,
            "slant": slant,
            "underline": underline,
            "overstrike": overstrike
        }
        if req_font != self._req_font:
            self._req_font = req_font
            self._stale = True
        
        return self._req_font
    
    def get_font(self) -> Font:
        return self._font
    
    def set_text(
            self,
            text: str,
            color: str | None = None,
            angle: IntFloat | None = None,
    ) -> dict[str, Any]:
        # Get default style and update it with the new values
        angle = angle if angle is None else angle % 360.0
        req_text = {
            "text": text,
            "color": color,
            "angle": angle
        }
        if req_text != self._req_text:
            self._req_text = req_text
            self._stale = True
        
        return self._req_text
    
    def get_text(self) -> dict[str, Any]:
        canvas = self._canvas
        return {
            "text": canvas.itemcget(self._id, 'text'),
            "color": canvas.itemcget(self._id, 'fill'),
            "angle": canvas.itemcget(self._id, 'angle')
        }


class _Line(_BaseArtist):
    _name = 'line'
    
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            x: NDArray[IntFloat],
            y: NDArray[IntFloat],
            color: str | None = None,
            width: IntFloat | None = None,
            smooth: bool | None = None,
            default_color: str = '',
            **kwargs
    ):
        super().__init__(canvas=canvas, **kwargs)
        self._xy: NDArray[float]
        self._default_color: str = default_color
        self._req_color: str | None = None
        self._req_width: IntFloat | None = None
        self._req_smooth: bool | None = None
        self._id: Int = self._canvas.create_line(
            0, 0, 0, 0, fill='', width=1, tags=self._tags
        )
        self.set_default_color(default_color)
        self.set_xy(x, y)
        self.set_color(color)
        self.set_width(width)
        self.set_smooth(smooth)
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self.set_color(self._req_color)
        self.set_width(self._req_width)
        self.set_smooth(self._req_smooth)
    
    def draw(self):
        if not self._stale:
            return
        
        dfts = self._default_style
        color = self._default_color if self._req_color is None \
            else self._req_color
        width = dfts["width"] if self._req_width is None \
            else self._req_width
        smooth = dfts["smooth"] if self._req_smooth is None \
            else self._req_smooth
        
        self._canvas.itemconfigure(
            self._id, fill=color, width=width, smooth=smooth
        )
        self._canvas.coords(self._id, *self._xy.ravel())
        self._update_zorder()
        self._stale = False
    
    def set_default_color(self, color: str = '') -> str:
        assert isinstance(color, str), color
        if color != self._default_color:
            self._default_color = color
            self._stale = True
        return self._default_color
    
    def get_default_color(self) -> str:
        return self._default_color
    
    def set_xy(#TODO: integer pixels
            self, x: NDArray[IntFloat], y: NDArray[IntFloat]
    ) -> NDArray[float]:
        assert isinstance(x, np.ndarray), x
        assert isinstance(y, np.ndarray), y
        assert x.shape == y.shape, [x.shape, y.shape]
        
        x_dt, y_dt = x.dtype, y.dtype
        assert any( np.issubdtype(x_dt, d) for d in IntFloat.__args__ ), x_dt
        assert any( np.issubdtype(y_dt, d) for d in IntFloat.__args__ ), y_dt
        
        self._xy = np.asarray([x.ravel(), y.ravel()], dtype=float).T
        
        return self._xy
    
    def get_xy(self) -> NDArray[float]:
        return tuple(self._xy.T)
    
    def set_color(self, color: str | None = None) -> str | None:
        assert isinstance(color, (str, type(None))), color
        if color != self._req_color:
            self._req_color = color
            self._stale = True
        return self._req_color
    
    def get_color(self) -> str:
        return self._canvas.itemcget(self._id, 'fill')
    
    def set_width(self, width: IntFloat | None = None) -> IntFloat:
        assert isinstance(width, (IntFloat, type(None))), width
        
        if width != self._req_width:
            self._req_width = width
            self._stale = True
        
        return self._req_width
    
    def get_width(self) -> int:
        return int(self._canvas.itemcget(self._id, 'width'))
    
    def set_smooth(self, smooth: bool | None = None) -> bool | None:
        assert isinstance(smooth, (bool, type(None))), smooth
        
        if smooth != self._req_smooth:
            self._req_smooth = smooth
            self._stale = True
        
        return self._req_smooth
    
    def get_smooth(self) -> bool:
        return bool(self._canvas.itemcget(self._id, 'smooth'))


class _Rectangle(_BaseArtist):
    _name = 'rect'
    
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            facecolor: str | None = None,
            edgecolor: str | None = None,
            **kwargs
    ):
        super().__init__(canvas=canvas, **kwargs)
        self._req_facecolor: str | None = None
        self._req_edgecolor: str | None = None
        self._id: Int = self._canvas.create_rectangle(
            0, 0, 1, 1, fill='', outline='', width=1, tags=self._tags
        )
        self.set_facecolor(facecolor)
        self.set_edgecolor(edgecolor)
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self.set_facecolor(self._req_facecolor)
        self.set_edgecolor(self._req_edgecolor)
        self.set_bbox(self._req_bbox)
    
    def draw(self):
        if not self._stale:
            return
        
        dfts = self._default_style
        facecolor = dfts["facecolor"] if self._req_facecolor is None \
            else self._req_facecolor
        edgecolor = dfts["edgecolor"] if self._req_edgecolor is None \
            else self._req_edgecolor
        if not edgecolor:
            edgecolor = facecolor  # edge => face
        
        self._canvas.itemconfigure(self._id, fill=facecolor, outline=edgecolor)
        self._canvas.coords(self._id, *self._req_bbox.get_xys())
        self._update_zorder()
        self._bbox = self._req_bbox.copy()
        self._stale = False
    
    def set_facecolor(self, color: str | None = None) -> str | None:
        assert isinstance(color, (str, type(None))), color
        if color != self._req_facecolor:
            self._req_facecolor = color
            self._stale = True
        return self._req_facecolor
    
    def get_facecolor(self) -> str:
        return self._canvas.itemcget(self._id, 'fill')
    
    def set_edgecolor(self, color: str | None = None) -> str | None:
        assert isinstance(color, (str, type(None))), color
        if color != self._req_edgecolor:
            self._req_edgecolor = color
            self._stale = True
        return self._req_edgecolor
    
    def get_edgecolor(self) -> str:
        return self._canvas.itemcget(self._id, 'outline')


# =============================================================================
# ---- Figure Regions
# =============================================================================
class _BaseRegion(_BaseElement):
    def __init__(
            self,
            canvas: _Plot | _Suptitle,
            tag: str,
            **kwargs
    ):
        assert isinstance(tag, str), tag
        super().__init__(canvas=canvas, **kwargs)
        self._tag = tag
    
    def on_resize(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None]
    ):
        raise NotImplementedError
    
    def get_bbox(self):
        raise NotImplementedError


class _Title(_BaseRegion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text: _Text = _Text(
            self._canvas,
            text='',
            default_style=self._default_style["text"],
            tags=(f'{self._tag}.text',)
        )
    
    def on_resize(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None]
    ):
        bbox = self._req_bbox.copy()
        bbox.set_xys(xys)
        self.set_bbox(**bbox.get_all())
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self._text.update_theme(default_style["text"])
    
    def draw(self):
        self._text.draw()
        self._bbox = self._text.get_bbox().copy()
    
    def set_bbox(self, *args, **kwargs) -> _CanvasBBox:
        bbox = self._text.set_bbox(_CanvasBBox(*args, **kwargs))
        self._req_bbox = bbox.copy()
        return self._req_bbox
    
    def get_bbox(self) -> _CanvasBBox:
        return self._bbox
    
    def set_text(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None],
            text: str,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Int | tuple[Int, Int] | None = None,
            pady: Int | tuple[Int, Int] | None = None
    ) -> _Text:
        self._text.set_font(
            family=family,
            size=size,
            weight=weight,
            slant=slant,
            underline=underline,
            overstrike=overstrike
        )
        self._text.set_text(text=text, color=color, angle=angle)
        self._text.set_bbox(
            _CanvasBBox(xys=xys, sticky=sticky, padx=padx, pady=pady)
        )
        
        return self._text
    
    def get_text(self) -> _Text:
        return self._text


class _Axis(_BaseRegion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._title: _Title
        self._ticks: NDArray[_Text]
    
    def on_resize(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None]
    ):
        if hasattr(self, '_title'):
            self._title.on_resize(xys)
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        if hasattr(self, '_title'):
            self._title.update_theme(self._default_style["title"])
    
    def draw(self):
        xs, ys = [], []
        
        if hasattr(self, '_title'):
            self._title.draw()
            x1, y1, x2, y2 = self._title.get_bbox().get_xys()
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        
        if xs:
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1, y1, x2, y2 = (0, 0, 0, 0)
        
        self._bbox = _CanvasBBox((x1, y1, x2, y2))
    
    def get_bbox(self) -> _CanvasBBox:
        return self._bbox
    
    def set_title(self, text: str, *args, **kwargs) -> _Title | None:
        if text:  # enable title
            if not hasattr(self, '_title'):
                self._title = _Title(
                   self._canvas,
                   default_style=self._default_style["title"],
                   tag=f'{self._tag}.title'
               )
            self._title.set_text(*args, text=text, **kwargs)
            
            return self._title
        
        # Disable title
        if hasattr(self, '_title'):
            delattr(self, '_title')
    
    def get_title(self) -> _Title:
        return self._title
    
    def set_ticks(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None],
            text: str,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Int | tuple[Int, Int] | None = None,
            pady: Int | tuple[Int, Int] | None = None
    ) -> NDArray[_Text] | None:#TODO: texts and positions
        if text:  # enable ticks
            if not hasattr(self, '_ticks'):
                self._ticks = np.array([
                     _Text(
                        self._canvas,
                        text='',
                        default_style=self._default_style["tick"]["text"],
                        tags=(f'{self._tag}.tick.text',)
                    )
               ])
            for tick in self._ticks:
                tick.set_font(
                    family=family,
                    size=size,
                    weight=weight,
                    slant=slant,
                    underline=underline,
                    overstrike=overstrike
                )
                tick.set_text(text=text, color=color, angle=angle)
                tick.set_bbox(xys=xys, sticky=sticky, padx=padx, pady=pady)
            
            return self._ticks
        
        # Disable title
        if hasattr(self, '_ticks'):
            delattr(self, '_ticks')
    
    def get_ticks(self) -> NDArray[_Text]:
        return self._ticks


class _Frame(_BaseRegion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rect: _Rectangle = _Rectangle(
            self._canvas,
            default_style=self._default_style,
            tags=(f'{self._tag}.rect',)
        )
    
    def on_resize(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None]
    ):
        self.set_bbox(xys)
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self._rect.update_theme(self._default_style)
    
    def draw(self):
        self._rect.draw()
        self._bbox = self._rect.get_bbox().copy()
    
    def set_bbox(
            self,
            xys: tuple[Int | None, Int | None, Int | None, Int | None]
    ) -> _CanvasBBox:
        bbox = self._rect.set_bbox(
            _CanvasBBox(xys, sticky='nesw', padx=0, pady=0)
            )
        self._req_bbox = bbox.copy()
        return self._req_bbox
    
    def get_bbox(self) -> _CanvasBBox:
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
    def __init__(
            self,
            figure: Figure,
            default_style: dict[str, Any] | None = None,
            **kwargs
    ):
        super().__init__(master=figure, **kwargs)
        self._figure = figure
        self._default_style: dict[str, Any]
        self._root_default_style: dict[str, Any]
        self._zorders: dict[_BaseArtist, str] = {}
        self._size: tuple[Int, Int] = (
            self.winfo_reqwidth(), self.winfo_reqheight()
        )
        self._update_theme(default_style)
        self.bind('<Configure>', self._on_configure, add=True)
    
    def _on_configure(self, event: tk.Event):
        self._size = (event.width, event.height)
    
    def _update_theme(self, default_style: dict[str, Any]):
        self._default_style = default_style.copy()
        self._root_default_style = self._figure._default_style.copy()
    
    def update_theme(self, default_style: dict[str, Any]):
        self._update_theme(default_style=default_style)
    
    def draw(self):
        raise NotImplementedError


class _Suptitle(_BaseSubwidget, tk.Canvas):
    _tag: str = 'suptitle'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._req_facecolor: str | None
        self._title: _Title = _Title(
            self, tag=self._tag, default_style=self._default_style
        )
        self.set_facecolor()
    
    def _on_configure(self, event: tk.Event):
        super()._on_configure(event)
        w, h = self._size
        self._title.on_resize((0, 0, w-1, None))
        self._title.draw()
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self.set_facecolor(self._req_facecolor)
        
        self._title.update_theme(default_style)
        
        event = tk.Event()
        event.width, event.height = self._size
        self._on_configure(event)
    
    def draw(self):
        self._title.draw()
        x1, y1, x2, y2 = self._title.get_bbox().get_xys()
        w, h = (x2 - x1 + 1), (y2 - y1 + 1)
        self.configure(width=w, height=h)  # triggers `self._on_configure`
    
    def set_facecolor(self, color: str | None = None):
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
            text: str,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Int | tuple[Int, Int] | None = None,
            pady: Int | tuple[Int, Int] | None = None,
    ) -> _Title:
        if sticky is not None and ('s' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for suptitle must not include 's' and not equal to "
                f"'center' but got {sticky}."
            )
        
        if not hasattr(self, '_title'):
            self._title = _Title(
               self,
               default_style=self._root_default_style["title"],
               tag=f'{self._tag}.title'
           )
        w, h = self._size
        self._title.set_text(
            xys=(0, 0, w-1, None),
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
        
        return self._title
    
    def get_title(self) -> _Title:
        return self._title


class _Plot(_BaseSubwidget, tk.Canvas):
    _tag: str = 'plot'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._artists: dict[str, list[_BaseArtist]] = {}
        self._color_cycles: dict[type, Cycle] = {}
        self._req_facecolor: str | None
        self._title: _Title
        self._taxis: _Axis
        self._baxis: _Axis
        self._laxis: _Axis
        self._raxis: _Axis
        self._frame: _Frame = _Frame(
            self,
            default_style=self._root_default_style["frame"],
            tag=f'{self._tag}.frame'
        )
        self._frame.set_bbox(self._get_xys_for_frame())
        self.set_facecolor()
        self.set_taxis(True)
        self.set_baxis(True)
        self.set_laxis(True)
        self.set_raxis(True)
    
    def _on_configure(self, event: tk.Event):
        super()._on_configure(event)
        w, h = self._size
        
        if hasattr(self, '_title'):
            self._title.on_resize((0, 0, w-1, None))
            self._title.draw()
        
        if hasattr(self, '_legend'):
            self._legend.on_resize(self._get_xys_for_legend())
            self._legend.draw()
        
        for side in ['t', 'b', 'l', 'r']:
            if hasattr(self, f'_{side}axis'):
                axis = self._get_axis(side)
                axis.on_resize(
                    self._get_xys_for_axis(side, draw_dependencies=False)
                )
                axis.draw()
        
        self._frame.on_resize(self._get_xys_for_frame())
        
        self.draw()
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self.set_facecolor(self._req_facecolor)
        
        if hasattr(self, '_title'):
            self._title.update_theme(self._root_default_style["title"])
        
        if hasattr(self, '_legend'):
            self._legend.update_theme(self._root_default_style["legend"])
        
        for side in ['t', 'b', 'l', 'r']:
            if hasattr(self, f'_{side}axis'):
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
    
    def draw(self):
        if hasattr(self, '_title'):
            self._title.draw()
        
        for side in ['t', 'b', 'l', 'r']:
            if hasattr(self, f'_{side}axis'):
                self._get_axis(side).draw()
        
        self._frame.draw()
        
        for artists in self._artists.values():
            for artist in artists:
                artist.draw()
        
        for tag in sorted(self._zorders.values()):
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
            text: str,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Int | tuple[Int, Int] | None = None,
            pady: Int | tuple[Int, Int] | None = None,
    ) -> _Title | None:
        if sticky is not None and ('s' in sticky or sticky == 'center'):
            raise ValueError(
                "`sticky` for title must not include 's' and not equal to "
                f"'center' but got {sticky}."
            )
        
        if text:  # enable title
            if not hasattr(self, '_title'):
                self._title = _Title(
                   self,
                   default_style=self._root_default_style["title"],
                   tag=f'{self._tag}.title'
               )
            w, h = self._size
            self._title.set_text(
                xys=(0, 0, w-1, None),
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
            
            return self._title
        
        # Disable title
        if hasattr(self, '_title'):
            delattr(self, '_title')
    
    def get_title(self) -> _Title:
        return self._title
    
    def _get_xys_for_legend(
            self
    ) -> tuple[Int | None, Int | None, Int | None, Int | None]:
        raise NotImplementedError
    
    def _get_xys_for_axis(
            self,
            side: Literal['t', 'b', 'l', 'r'],
            draw_dependencies: bool = True
    ) -> tuple[Int | None, Int | None, Int | None, Int | None]:
        def _get_boundary(which: Literal['_title', '_legend']) -> Int | None:
            if (region := getattr(self, which, None)) is None:
                return
            
            assert which in ('_title', '_legend'), which
            
            if draw_dependencies:
                region.draw()
            _x1, _y1, _x2, _y2 = region.get_bbox().get_xys()
            
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
            if (y1_bound := _get_boundary('_title')) is not None:
                y1 = y1_bound
        elif side == 'b':  # bottom axis
            y1 = None
        elif side == 'l':  # left axis
            x2 = None
            if (y1_bound := _get_boundary('_title')) is not None:
                y1 = y1_bound
        else:  # right axis
            x1 = None
            if (y1_bound := _get_boundary('_title')) is not None:
                y1 = y1_bound
            if (x2_bound := _get_boundary('_legend')) is not None:
                x2 = x2_bound
        
        return (x1, y1, x2, y2)
    
    def _get_xys_for_frame(
            self
    ) -> tuple[Int | None, Int | None, Int | None, Int | None]:
        w, h = self._size
        _, y1, _, _ = self._get_xys_for_axis('t')
        _, _, _, x2 = self._get_xys_for_axis('r')
        xys = [0, y1, w-1, x2]
        for i, (side, j) in enumerate({"l": 2, "t": 3, "r": 0, "b": 1}.items()):
            if hasattr(self, f'_{side}axis'):
                axis = self._get_axis(side)
                axis.draw()
                _xys = axis.get_bbox().get_xys()
                assert (p :=_xys[j]) is not None, _xys
                if any( _p != p for _p in _xys ):  # non-empty axis
                    xys[i] = p
        
        return tuple(xys)
    
    def _set_axis(
            self,
            side: Literal['t', 'b', 'l', 'r'],
            enable: bool = True
    ) -> tuple[str, _Axis | None]:
        assert side in ('t', 'b', 'l', 'r'), side
        
        axis_name = f'{side}axis'
        attr_name = f'_{side}axis'
        
        if enable and not hasattr(self, attr_name):
            axis = _Axis(
               self,
               default_style=self._root_default_style[axis_name],
               tag=f'{self._tag}.{axis_name}'
            )
            setattr(self, attr_name, axis)
        elif not enable and hasattr(self, attr_name):
            delattr(self, attr_name)
        
        if enable:
            return (attr_name, self._get_axis(side))
        return (attr_name, None)
    
    def set_taxis(self, *args, **kwargs) -> _Axis | None:
        return self._set_axis('t', *args, **kwargs)[-1]
    
    def set_baxis(self, *args, **kwargs) -> _Axis | None:
        return self._set_axis('b', *args, **kwargs)[-1]
    
    def set_laxis(self, *args, **kwargs) -> _Axis | None:
        return self._set_axis('l', *args, **kwargs)[-1]
    
    def set_raxis(self, *args, **kwargs) -> _Axis | None:
        return self._set_axis('r', *args, **kwargs)[-1]
    
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
            text: str,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Int | tuple[Int, Int] | None = None,
            pady: Int | tuple[Int, Int] | None = None
    ) -> _Axis:
        attr_name, axis = self._set_axis(side, enable=True)
        self._get_axis(side).set_title(
            xys=self._get_xys_for_axis(side),
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
            x: ArrayLike | None = None,
            y: ArrayLike | None = None,
            color: str | None = None,
            width: IntFloat | None = None,
            smooth: bool | None = None,
            label: str = ''
    ) -> _Line:
        assert isinstance(label, str), label
        
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        assert x.shape == y.shape, [x.shape, y.shape]
        
        cycle = self._create_color_cycle()
        
        lines = self._artists.setdefault('line', [])
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
        
        self._req_facecolor: str | None
        self.set_facecolor()
    
    def update_theme(self, default_style: dict[str, Any]):
        super().update_theme(default_style)
        self.set_facecolor(self._req_facecolor)
    
    def set_facecolor(self, color: str | None = None) -> str | None:
        assert isinstance(color, (str, type(None))), color
        
        # Get default style and update it with the new values
        default_color = self._root_default_style["facecolor"]
        new_color = default_color if color is None else color
        self.configure(bg=new_color)
        self._xyz_lb.configure(bg=new_color)
        self._req_facecolor = color
        
        return self._req_facecolor
    
    def get_facecolor(self) -> str:
        return self["background"]
    
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
            master: tk.BaseWidget,
            suptitle: str = '',
            toolbar: bool = True,
            width: Int | None = None,
            height: Int | None = None,
            padx: Int = 6,
            pady: Int = 6,
            **kwargs
    ):
        window_title = suptitle or 'Figure'
        super().__init__(
            master, window_title=window_title, padx=padx, pady=pady, **kwargs
        )
        
        self._initialized: bool = False
        self._req_size: tuple[Int, Int]
        self._plots: NDArray[_Plot]
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
        udbt_bg = self._default_style["toolbar"]["facecolor"]
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
            self._suptitle.update_theme(self._default_style["suptitle"])
        
        # Update plots
        if hasattr(self, '_plots'):
            for plot in self._plots.flat:
                plot.update_theme(self._default_style["plot"])
        
        # Update toolbar
        if hasattr(self, '_toolbar'):
            self._toolbar.update_theme(self._default_style["toolbar"])
    
    def draw(self):
        self._initialized = True
        
        if hasattr(self, '_suptitle'):
            self._suptitle.draw()
        
        if hasattr(self, '_plots'):
            for plot in self._plots.flat:
                if plot:
                    plot.draw()
    
    def set_size(self, width: Int | None = None, height: Int | None = None):
        default_width, default_height = self._default_style["size"]
        width = default_width if width is None else width
        height = default_height if height is None else height
        new_size = (width, height)
        
        if not hasattr(self, '_req_size') or self._req_size != new_size:
            self._req_size = new_size
            self.configure(width=width, height=height)
    
    def get_size(self) -> tuple[Int, Int]:
        return self._req_size
    
    def set_suptitle(
            self,
            text: str,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            sticky: str | None = None,
            padx: Int | tuple[Int, Int] | None = None,
            pady: Int | tuple[Int, Int] | None = None,
            facecolor: str | None = None
    ) -> _Suptitle | None:
        if text:  # enable suptitle
            if not hasattr(self, '_suptitle'):
                self._suptitle = _Suptitle(
                    self,
                    default_style=self._default_style["suptitle"]
                )
                self._suptitle.grid(row=0, column=0, sticky='we')
                if hasattr(self, '_plots'):
                    n_rows, n_cols = self._plots.shape
                    self._suptitle.grid(columnspan=n_cols)
            self._suptitle.set_facecolor(facecolor)
            self._suptitle.set_title(
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
            padx: Int | tuple[Int] = (1, 1),
            pady: Int | tuple[Int] = (1, 1)
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
        self.grid_propagate(False)  # allow `self` to be resized
        
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
                var_coord=self._var_coord,
                default_style=self._default_style["toolbar"]
            )
            kw = {"column": 0, "sticky": 'we', "padx": 9, "pady": (9, 0)}
            if hasattr(self, '_plots'):
                n_rows, n_cols = self._plots.shape
                self._toolbar.grid(row=n_rows+1, columnspan=n_cols, **kw)
            else:
                self._toolbar.grid(row=1, **kw)
                 # the toolbar will be `grid` again when `set_plots` is called
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
    plt = fig.set_plots(1, 1)
    plt.plot(x, y)
    plt.set_title('<Title>')
    plt.set_tlabel('<top-label>')
    plt.set_blabel('<bottom-label>')
    plt.set_llabel('<left-label>')
    plt.set_rlabel('<right-label>')
    
    fig.after(3000, lambda: root.style.theme_use('cyborg'))
    
    root.mainloop()

