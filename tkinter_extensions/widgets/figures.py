#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:39:51 2024

@author: tungchentsai
"""

from __future__ import annotations
from typing import Any, Literal, Callable, Iterable, NamedTuple
import tkinter as tk
from tkinter.font import Font
from copy import deepcopy
from functools import wraps
from itertools import cycle as Cycle
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

import numpy as np
from numpy.typing import NDArray, ArrayLike
import ttkbootstrap as ttk

from tkinter_extensions import variables as vrb
from tkinter_extensions.constants import (
    Int, IntFloat, Float, Dimension,
    MLEFTPRESS, MLEFTMOTION, MLEFTRELEASE,
    MRIGHTPRESS, MRIGHTMOTION, MRIGHTRELEASE, MMOTION,
    DRAWSTARTED, DRAWENDED
)
from tkinter_extensions.utils import defer, unbind, contrast_color
from tkinter_extensions.widgets.scrolled import ScrolledCanvas
from tkinter_extensions.widgets._others import UndockedFrame
from tkinter_extensions.widgets._figure_config import STYLES

_ANCHORS = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
_NP_FLOAT = np.float64
_NP_FINFO = np.finfo(_NP_FLOAT)
_PMIN, _MIN, _MAX = (_NP_FINFO.smallest_normal, _NP_FINFO.min, _NP_FINFO.max)
_DMIN, _DMAX = (_NP_FLOAT(-1e+10), _NP_FLOAT(+1e+10))
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
) -> float | None | tuple[float | None, ...]:
    assert isinstance(root, tk.Tk), root
    
    _winfo_fpixels = root.winfo_fpixels
    _to_pixels = lambda d: round(_winfo_fpixels(d))
    
    if dimension is None:
        return None
    elif isinstance(dimension, (str, IntFloat)):
        return _to_pixels(dimension)
    return tuple( None if d is None else _to_pixels(d) for d in dimension )


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
    
    if start > stop or sticky == 'center':
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
        xs: NDArray[IntFloat], ys: NDArray[IntFloat]
) -> NDArray[IntFloat]:
    xy = np.asarray([xs, ys])
    assert xy.ndim == 2, xy.shape
    
    if xy.shape[1] < 3:
        return xy
    
    retain = np.diff(xy[:, :-1], axis=1).any(axis=0)
    
    return np.concat(
        (xy[:, :1], xy[:, 1:-1][:, retain], xy[:, -1:]),
        axis=1
    )


def _drop_linearly_redundant_points(xy: NDArray[IntFloat]) -> NDArray[IntFloat]:
    """
    Simplifies a series of points based on the perpendicular distance. Drop the
    points having a 0 perpendicular distance.
    
    Ref: https://psimpl.sourceforge.net/perpendicular-distance.html
    """
    assert isinstance(xy, np.ndarray), type(xy)
    assert xy.ndim == 2, xy.shape
    assert xy.shape[0] == 2, xy.shape
    
    if xy.shape[1] < 3:
        return xy
    
    # Get xy
    x0s, y0s = xy[:, :-2]  # front points
    x1s, y1s = xy1 = xy[:, 1:-1]  # middle points
    x2s, y2s = xy2 = xy[:, 2:]  # back points
    
    # Calculate the perpendicular distances from middle points to front-back lines
    numerator = np.abs((y2s - y0s)*x1s - (x2s - x0s)*y1s + x2s*y0s - y2s*x0s)
    denominator = np.sqrt((x2s - x0s)**2 + (y2s - y0s)**2)
    with np.errstate(divide='ignore', invalid='ignore'):  # address 1/0 or 0/0
        dists = numerator / denominator
    
    round_trip = denominator == 0  # x0 == x2 and y0 == y2
    x1s_round, y1s_round = xy1[:, round_trip]
    x2s_round, y2s_round = xy2[:, round_trip]
    dists[round_trip] = np.sqrt(
        (x2s_round - x1s_round)**2 + (y2s_round - y1s_round)**2
    )  # from x1 to x2
    
    retain = dists > 0
    
    return np.concat(
        (xy[:, :1], xy[:, 1:-1][:, retain], xy[:, -1:]),
        axis=1
    )


def _cutoff_z_patterns(xy: NDArray[IntFloat]) -> NDArray[IntFloat]:
    """
    Z patterns usually result from rounding the points (x, y) from a tilt
    line segment. For example,
        a size-1 vertical line in between two horizontal lines:
        -...-
             |
             -...-
        
        or a size-1 horizontal line in between two vertical lines:
        |
        .
        .
        .
        |
        -
         |
         .
         .
         .
         |
    
    We simplify the z patterns by dropping the turning points.
    """
    def _find_z_patterns(dup1, dun1, dvp, dvn):
        du1_idc = []
        for du1 in [dup1, dun1]:
            _du1_idc = du1.nonzero()[0]
            if _du1_idc.size and _du1_idc[0] == 0:
                _du1_idc = _du1_idc[1:]
            if _du1_idc.size and _du1_idc[-1] == dup1.size - 1:
                _du1_idc = _du1_idc[:-1]
            du1_idc.append(_du1_idc)
        du1_idc = np.concat(du1_idc)
        
        z_pattern = (dvp[du1_idc - 1] & dvp[du1_idc + 1]) \
                  | (dvn[du1_idc - 1] & dvn[du1_idc + 1])
        
        return du1_idc[z_pattern]
    #> end of _find_z_patterns()
    
    assert isinstance(xy, np.ndarray), type(xy)
    assert xy.ndim == 2, xy.shape
    assert xy.shape[0] == 2, xy.shape
    
    if xy.shape[1] < 4:
        return xy
    
    dx, dy = np.diff(xy, axis=1)
    dx0, dy0 = (dx == 0), (dy == 0)
    dxp, dxn = dy0 & (dx > 0), dy0 & (dx < 0)
    dyp, dyn = dx0 & (dy > 0), dx0 & (dy < 0)
    dxp1, dxn1 = (dx == 1), (dx == -1)
    dyp1, dyn1 = (dy == 1), (dy == -1)
    
    z_pattern_idc_x = _find_z_patterns(dxp1, dxn1, dyp, dyn)
    z_pattern_idc_y = _find_z_patterns(dyp1, dyn1, dxp, dxn)
    
    retain = np.ones(dx.size, dtype=bool)
    retain[z_pattern_idc_x - 1] = False  # drop front points
    retain[z_pattern_idc_y] = False  # drop back points
    
    return np.concat((xy[:, :1], xy[:, 1:][:, retain]), axis=1)


class ZorderNotFoundError(RuntimeError):
    pass


class _PlotView(NamedTuple):
    plot: _Plot
    view: dict[str, tuple[Any, Any]]
    
    # The `__new__()` of `NamedTuple` can't be overwritten, so we define `make()`
    # instead. `make()` will be called when a instance is created.
    @classmethod
    def make(cls, plot: _Plot, view: dict[str, tuple[Any, Any]]):
        assert isinstance(plot, _Plot), plot
        assert isinstance(view, dict), view
        assert tuple(view) == ('r', 'b', 'l', 't'), (view)
        
        for limits, margins in view.values():
            assert len(limits) == 2, limits
            assert isinstance(limits[0], (IntFloat, type(None))), limits
            assert isinstance(limits[1], (IntFloat, type(None))), limits
            assert len(margins) == 2, margins
            assert isinstance(margins[0], (Dimension, type(None))), margins
            assert isinstance(margins[1], (Dimension, type(None))), margins
        
        return cls(plot, view)


class _ViewSet(tuple):
    def __new__(cls, iterable: Iterable[_PlotView]):
        self = super().__new__(cls, iterable)
        assert all( isinstance(v, _PlotView) for v in self ), iterable
        return self


class _ViewSetHistory:
    @property
    def step(self) -> int:
        return self._step
    
    @property
    def backable(self) -> bool:
        return self._step > 0
    
    @property
    def forwardable(self) -> bool:
        return 0 <= self._step < len(self._stack) - 1
    
    def __init__(self, figure: Figure):
        self._figure: Figure = figure
        self._step: int = -1
        self._stack: list[_ViewSet] = []
        self._update_toolbar_buttons()
    
    def __bool__(self) -> bool:
        return bool(self._stack)
    
    def __getitem__(self, i: Int) -> _ViewSet:
        assert isinstance(i, Int), i
        return self._stack[i]
    
    def _update_toolbar_buttons(self):
        if not (tb := getattr(self._figure, '_toolbar', None)):
            return
        
        state = 'normal' if self.backable else 'disabled'
        tb._prev_bt.configure(state=state)
        
        state = 'normal' if self.forwardable else 'disabled'
        tb._next_bt.configure(state=state)
    
    def add(self, v: _ViewSet):
        assert isinstance(v, _ViewSet), v
        
        self._stack[:] = self._stack[:self._step+1] + [v]
        self._step += 1
        self._update_toolbar_buttons()
    
    def replaced_by(self, iterable: Iterable):
        new = list(iterable)
        assert all( isinstance(v, _ViewSet) for v in new ), new
        self._stack[:] = new
        self._step = len(self._stack) - 1
        self._update_toolbar_buttons()
    
    def clear(self):
        self.__init__(self._figure)
    
    def drop_future(self):
        self._stack[:] = self._stack[:self._step+1]
        self._update_toolbar_buttons()
    
    def forward(self) -> Any:
        assert self.forwardable, (self._step, self._stack)
        self._step += 1
        self._update_toolbar_buttons()
        return self._stack[self._step]
    
    def back(self) -> Any:
        assert self.backable, (self._step, self._stack)
        self._step -= 1
        self._update_toolbar_buttons()
        return self._stack[self._step]


class _BaseTransform1D:  # 1D transformation
    def __eq__(self, obj) -> bool:
        if type(self) != type(obj):
            return False
        return True
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat]
    ) -> NDArray[Float]:
        assert isinstance(xs, (IntFloat, np.ndarray)), xs
        
        xs = np.asarray(xs)
        dt = xs.dtype
        assert any( np.issubdtype(dt, d) for d in IntFloat.__args__ ), xs.dtype
        
        if not np.issubdtype(dt, np.floating):
            xs = xs.astype(float)
        
        return xs
    
    def copy(self, *args, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def from_points(cls, *args, **kwargs):
        raise NotImplementedError
    
    def get_inverse(self):
        raise NotImplementedError


class _FirstOrderPolynomial(_BaseTransform1D):
    def __init__(self, c0: IntFloat = 0., c1: IntFloat = 1.):
        assert isinstance(c0, IntFloat), c0
        assert isinstance(c1, IntFloat), c1
        
        self._c0: np.float64 = np.float64(c0)
        self._c1: np.float64 = np.float64(c1)
    
    def __eq__(self, obj) -> bool:
        return super().__eq__(obj) and self._c0 == obj._c0 and self._c1 == obj._c1
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat],
            round_: bool = False
    ) -> Float | NDArray[Float]:
        xs = super().__call__(xs)
        ys = self._c0 + self._c1 * xs  # y(x) = c0 + c1 * x (0D or 1D array)
        
        if round_:
            out = ys if isinstance(ys, np.ndarray) else None
            ys = ys.round(out=out)
        
        return ys
    
    def copy(self) -> _FirstOrderPolynomial:
        return type(self)(c0=self._c0, c1=self._c1)
    
    @classmethod
    def from_points(cls, xs: ArrayLike, ys: ArrayLike) -> _FirstOrderPolynomial:
        xs, ys = np.asarray(xs), np.asarray(ys)
        assert xs.shape == ys.shape == (2,), [xs.shape, ys.shape]
        
        # y(x) = c0 + c1 * x
        c1 = (ys[1] - ys[0]) / (xs[1] - xs[0])  # c1 = (y1 - y0) / (x1 - x0)
        c0 = ys[0] - c1 * xs[0]  # c0 = y0 - c1 * x0
        
        return cls(c0=c0, c1=c1)
    
    def get_inverse(self) -> _FirstOrderPolynomial:
        assert self._c1 != 0., self._c1
        
        c1_inv = 1. / self._c1  # c1_inv = 1 / c1
        c0_inv = -self._c0 / self._c1  # c0_inv = -c0 / c1
        
        return _FirstOrderPolynomial(c0=c0_inv, c1=c1_inv)


class _Logarithm(_BaseTransform1D):
    def __init__(
            self,
            base: IntFloat = 10.,
            c0: IntFloat = 0.,
            c1: IntFloat = 1.
    ):
        assert isinstance(base, IntFloat), base
        assert isinstance(c0, IntFloat), c0
        assert isinstance(c1, IntFloat), c1
        assert base > 0., base
        
        self._c0: np.float64 = np.float64(c0)
        self._c1: np.float64 = np.float64(c1)
        self._base: np.float64 = np.float64(base)
        self._log2_base: np.float64 = np.log2(self._base)
    
    def __eq__(self, obj) -> bool:
        return super().__eq__(obj) and self._base == obj._base and \
            self._c0 == obj._c0 and self._c1 == obj._c1
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat],
            round_: bool = False
    ) -> NDArray[Float]:
        xs = super().__call__(xs)
        
        if self._base == 2.:
            ys = self._c1 * np.log2(xs) + self._c0
        elif self._base == np.e:
            ys = self._c1 * np.log(xs) + self._c0
        elif self._base == 10.:
            ys = self._c1 * np.log10(xs) + self._c0
        else:
            ys = self._c1 * np.log2(xs) / self._log2_base + self._c0
        
        if round_:
            out = ys if isinstance(ys, np.ndarray) else None
            ys = ys.round(out=out)
        
        return ys
    
    def copy(self) -> _Logarithm:
        return type(self)(base=self._base, c0=self._c0, c1=self._c1)
    
    @classmethod
    def from_points(
            cls, xs: ArrayLike, ys: ArrayLike, base: IntFloat = 10.
    ) -> _Logarithm:
        assert base > 0., base
        
        # y(x) = c1 * log(x) / log(base) + c0
        if base == 2.:
            log_xs = np.log2(xs)
        elif base == np.e:
            log_xs = np.log(xs)
        elif base == 10.:
            log_xs = np.log10(xs)
        else:
            log_xs = np.log2(xs) / np.log2(base)
        
        c1 = (ys[1] - ys[0]) / (log_xs[1] - log_xs[0])  # c1 = (y1 - y0) / (x1 - x0)
        c0 = ys[0] - c1 * log_xs[0]  # c0 = y0 - c1 * x0
        
        return cls(base=base, c0=c0, c1=c1)
    
    def get_inverse(self) -> _InverseLogarithm:
        return _InverseLogarithm(base=self._base, c0=self._c0, c1=self._c1)


class _InverseLogarithm(_BaseTransform1D):
    def __init__(
            self,
            base: IntFloat = 10.,
            c0: IntFloat = 0.,
            c1: IntFloat = 1.
    ):
        assert isinstance(base, IntFloat), base
        assert isinstance(c0, IntFloat), c0
        assert isinstance(c1, IntFloat), c1
        assert base > 0., base
        assert c1 != 0., c1
        
        self._c0: np.float64 = np.float64(c0)
        self._c1: np.float64 = np.float64(c1)
        self._base: np.float64 = np.float64(base)
    
    def __eq__(self, obj) -> bool:
        return super().__eq__(obj) and self._base == obj._base and \
            self._c0 == obj._c0 and self._c1 == obj._c1
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat],
            round_: bool = False
    ) -> NDArray[Float]:
        xs = super().__call__(xs)
        ys = self._base ** ((xs - self._c0) / self._c1)
        
        if round_:
            out = ys if isinstance(ys, np.ndarray) else None
            ys = ys.round(out=out)
        
        return ys
    
    def copy(self) -> _InverseLogarithm:
        return type(self)(base=self._base, c0=self._c0, c1=self._c1)
    
    def get_inverse(self) -> _Logarithm:
        return _Logarithm(base=self._base, c0=self._c0, c1=self._c1)


class _Transform2D:  # 2D transformation
    def __init__(
            self,
            x_transform: _BaseTransform1D = _FirstOrderPolynomial(),
            y_transform: _BaseTransform1D = _FirstOrderPolynomial(),
            inp_xbounds: ArrayLike | None = None,   # input xlimits
            inp_ybounds: ArrayLike | None = None,   # input ylimits
            out_xbounds: ArrayLike = [-np.inf, np.inf],   # output xlimits
            out_ybounds: ArrayLike = [-np.inf, np.inf]    # output ylimits
    ):
        assert isinstance(x_transform, _BaseTransform1D), x_transform
        assert isinstance(y_transform, _BaseTransform1D), y_transform
        
        if inp_xbounds is None:
            if isinstance(x_transform, _Logarithm):
                inp_xbounds = [_PMIN, np.inf]
            else:
                inp_xbounds = [-np.inf, np.inf]
        if inp_ybounds is None:
            if isinstance(x_transform, _Logarithm):
                inp_ybounds = [_PMIN, np.inf]
            else:
                inp_ybounds = [-np.inf, np.inf]
        
        inp_xbounds = np.array(inp_xbounds, dtype=float)
        if isinstance(x_transform, _Logarithm):
            inp_xbounds[inp_xbounds < _PMIN] = _PMIN
        inp_ybounds = np.array(inp_ybounds, dtype=float)
        if isinstance(y_transform, _Logarithm):
            inp_ybounds[inp_ybounds < _PMIN] = _PMIN
        
        self._x_tf: _BaseTransform1D = x_transform
        self._y_tf: _BaseTransform1D = y_transform
        self._inp_xlimits = _, _ = sorted(inp_xbounds)
        self._inp_ylimits = _, _ = sorted(inp_ybounds)
        self._out_xlimits = _, _ = sorted(out_xbounds)
        self._out_ylimits = _, _ = sorted(out_ybounds)
    
    def __call__(
            self,
            xs: IntFloat | NDArray[IntFloat],
            ys: IntFloat | NDArray[IntFloat],
            round_: bool = False,
            clip: bool = True
    ) -> NDArray[Float]:
        xs, ys = np.array(xs), np.array(ys)  # copy
        assert isinstance(xs, np.ndarray), xs
        assert isinstance(xs, np.ndarray), xs
        assert xs.shape == ys.shape, [xs.shape, ys.shape]
        assert xs.ndim <= 1, xs.shape
        
        scalar = xs.ndim == 0
        
        if clip and not scalar:
            xs, ys = self._clip(xs, ys, limits='input')
        xs = self._x_tf(xs, round_=round_)
        ys = self._y_tf(ys, round_=round_)
        
        xy = np.asarray([xs, ys])
        if scalar:
            return xy.ravel()
        return xy
    
    def _clip(
            self,
            xs: IntFloat | NDArray[IntFloat],
            ys: IntFloat | NDArray[IntFloat],
            limits: Literal['input', 'output']
    ) -> tuple[NDArray[Float], NDArray[Float]]:
        def _restrict(
                xs: NDArray[Float], ys: NDArray[Float], u: Literal['x', 'y']
        ) -> tuple[NDArray[Float], NDArray[Float]]:
            assert u in ('x', 'y'), u
            
            if u == 'x':
                us, vs = xs.copy(), ys.copy()
                umin, umax = (xmin, xmax)
                _interp = lambda i, x: (ys[i+1]-ys[i])/(xs[i+1]-xs[i]) \
                    * (x-xs[i]) + ys[i]
            else:  # u == 'y'
                us, vs = ys.copy(), xs.copy()
                umin, umax = (ymin, ymax)
                _interp = lambda i, y: (xs[i+1]-xs[i])/(ys[i+1]-ys[i]) \
                    * (y-ys[i]) + xs[i]
            
            us, vs = _interpolate(
                us, vs, retain=umin <= us, ulimit=umin, interp=_interp
            )
            
            if u == 'x':
                xs, ys = us.copy(), vs.copy()
            else:  # u == 'y'
                xs, ys = vs.copy(), us.copy()
            
            us, vs = _interpolate(
                us, vs, retain=us <= umax, ulimit=umax, interp=_interp
            )
            
            if u == 'x':
                return us, vs
            # u == 'y'
            return vs, us
        #> end of _restrict()
        
        def _interpolate(
                us: NDArray[Float],
                vs: NDArray[Float],
                retain: NDArray[bool],
                ulimit: Float,
                interp: Callable
        ) -> tuple[NDArray[Float], NDArray[Float]]:
            crossing = np.diff(retain.astype(int))
            
            us_inserts, vs_inserts = [], []
            for j in (crossing == +1).nonzero()[0]:
                # index: j        k = j+1
                # valid: False    True
                us[j] = ulimit
                vs[j] = interp(j, ulimit)
                retain[j] = True
            for i in (crossing == -1).nonzero()[0]:
                j = i + 1
                if retain[j]:
                    # index: i       j = i+1            k = i+2
                    # valid: True    (False =>) True    True
                    j2 = retain[:j].sum()
                    us_inserts.append([j2, ulimit])
                    vs_inserts.append([j2, interp(i, ulimit)])
                else:
                    # index: i       j = i+1            k = i+2
                    # valid: True    False (=> True)    False
                    us[j] = ulimit
                    vs[j] = interp(i, ulimit)
                    retain[j] = True
            
            us, vs = us[retain], vs[retain]
            if us_inserts:
                us = np.insert(us, *zip(*us_inserts))
                vs = np.insert(vs, *zip(*vs_inserts))
            
            return us, vs
        #> end of _interpolate()
        
        assert limits in ('input', 'output'), limits
        assert xs.shape == ys.shape, [xs.shape, ys.shape]
        
        if limits == 'input':
            xmin, xmax = self._inp_xlimits
            ymin, ymax = self._inp_ylimits
        else:  # limits == 'input'
            xmin, xmax = self._out_xlimits
            ymin, ymax = self._out_ylimits
        
        xs, ys = _restrict(xs, ys, 'x')
        xs, ys = _restrict(xs, ys, 'y')
        
        return xs, ys
    
    def get_inverse(self) -> _Transform2D:
        return _Transform2D(
            x_transform=self._x_tf.get_inverse(),
            y_transform=self._y_tf.get_inverse(),
            inp_xbounds=self._out_xlimits,
            inp_ybounds=self._out_ylimits,
            out_xbounds=self._inp_xlimits,
            out_ybounds=self._inp_ylimits
        )


class _BaseElement:
    def __init__(self, canvas: tk.Canvas, tag: str = ''):
        assert isinstance(canvas, tk.Canvas), canvas
        assert isinstance(canvas._figure, Figure), canvas._figure
        assert isinstance(tag, str), tag
        
        self._root = canvas._root
        self._canvas: tk.Canvas = canvas
        self._figure: Figure = canvas._figure
        self._tag: str = tag
    
    def __del__(self):
        _cleanup_tk_attributes(self)
    
    def _to_px(self, dimension: Dimension) -> float | tuple[float, ...]:
        return _to_px(self._root(), dimension)
    
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
            state: Literal['normal', 'hidden', 'disabled'] = 'normal',
            label: str | None = None,
            antialias: bool = False,
            antialias_bg: Callable[[], str] | None = None,
            hover: bool = False,
            transform: _Transform2D = _Transform2D(),
            movable: bool = False,
            resizable: bool = False,
            extra_tags: tuple[str, ...] = (),
            x_side: Literal['b', 't'] | None = None,
            y_side: Literal['l', 'r'] | None = None,
            **kwargs
    ):
        assert x_side in ('b', 't', None), x_side
        assert y_side in ('l', 'r', None), y_side
        assert isinstance(label, (str, type(None))), label
        assert isinstance(movable, bool), movable
        assert isinstance(resizable, bool), resizable
        assert isinstance(extra_tags, tuple), extra_tags
        assert all( isinstance(t, str) for t in extra_tags ), extra_tags
        assert antialias_bg is None or callable(antialias_bg), antialias_bg
        assert isinstance(hover, bool), hover
        
        super().__init__(*args, **kwargs)
        
        self._req_label: str | None = label
        self._req_transform: _Transform2D = transform
        self._req_coords: list[Dimension] = []
        self._req_state: Literal['normal', 'hidden', 'disabled'] = 'normal'
        self._req_zorder: float | None = None
        self._req_style: dict[str, Any] = {}
        
        subtags = self._tag.split('.')
        tags = [
            '.'.join(subtags[:i]) for i in range(1, len(subtags)+1)
        ]
        tags.append(f'movable={movable}')
        tags.append(f'resizable={resizable}')
        tags.extend(extra_tags)
        tags = tuple(dict.fromkeys(tags))  # ordered and unique elements
        
        self._tags: tuple[str, ...] = tags
        self._x_side: Literal['b', 't'] | None = x_side
        self._y_side: Literal['l', 'r'] | None = y_side
        self._antialias_enabled: bool = bool(antialias)
        self._antialias_bg: Callable[[], str] | None = antialias_bg
        self._hover: bool = hover
        self._id: int
        self._id_aa: int
        self._stale: bool
        self.set_state(state)
    
    @property
    def _root_default_style(self) -> dict[str, Any]:
        return self._figure._default_style[self._name]
    
    @property
    def _default_style(self) -> dict[str, Any]:
        return self._figure._default_style[self._tag]
    
    def __del__(self):
        self.delete()
    
    def coords(self, *args, **kwargs) -> list[float]:
        return tk.Canvas.coords(self._canvas, self._id, *args, **kwargs)
    
    def move(self, *args, **kwargs):
        tk.Canvas.move(self._canvas, self._id, *args, **kwargs)
    
    def moveto(self, *args, **kwargs):
        tk.Canvas.move(self._canvas, self._id, *args, **kwargs)
    
    def lift(self, *args, **kwargs):
        tk.Canvas.tag_raise(self._canvas, self._id, *args, **kwargs)
    
    def lower(self, *args, **kwargs):
        tk.Canvas.tag_lower(self._canvas, self._id, *args, **kwargs)
    
    def itemconfigure(self, *args, **kwargs) -> Any:
        return tk.Canvas.itemconfigure(self._canvas, self._id, *args, **kwargs)
    
    def cget(self, *args, **kwargs) -> Any:
        return tk.Canvas.itemcget(self._canvas, self._id, *args, **kwargs)
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        return tk.Canvas.bbox(self._canvas, self._id)
    
    def delete(self):
        for side in (self._x_side, self._y_side):
            if side is not None:
                artists = getattr(self._canvas, f'_{side}artists')
                try:
                    artists[f'{self._name}s'].remove(self)
                except ValueError:
                    pass
        
        self._canvas._zorder_tags.pop(self, None)
        
        try:
            tk.Canvas.delete(self._canvas, self._id)
        except tk.TclError:
            pass
    
    def set_transform(self, transform: _Transform2D | None):
        assert isinstance(transform, (_Transform2D, type(None))), transform
        
        if transform is not None and transform != self._req_transform:
            self._req_transform = transform
            self._stale = True
    
    def get_transform(self) -> _Transform2D:
        return self._req_transform
    
    def set_label(self, label: str | None = None):
        assert isinstance(label, (str, type(None))), label
        
        if label is not None and label != self._req_label:
            self._req_label = label
    
    def get_label(self) -> str | None:
        return self._req_label
    
    def set_coords(self, *ps: Dimension):
        assert all( isinstance(p, (Dimension, type(None))) for p in ps ), ps
        
        if ps is not None and ps != self._req_coords:
            self._req_coords[:] = ps
            self._stale = True
    
    def get_coords(self) -> list[float]:
        return self.coords()
    
    def _set_state(
            self,
            state: Literal['normal', 'hidden', 'disabled']
    ):
        assert state in ('normal', 'hidden', 'disabled')
        
        self.itemconfigure(state=state)
        
        if self._antialias_enabled:
            tk.Canvas.itemconfigure(self._canvas, self._id_aa, state=state)
        elif hasattr(self, '_id_aa'):
            tk.Canvas.itemconfigure(self._canvas, self._id_aa, state='hidden')
    
    def set_state(
            self,
            state: Literal['normal', 'hidden', 'disabled'] = None
    ):
        assert state in ('normal', 'hidden', 'disabled', None), state
        
        if state is not None and state != self._req_state:
            self._req_state = state
            self._stale = True
    
    def get_state(self) -> str:
        return self.cget('state')
    
    def set_zorder(self, zorder: IntFloat | None = None):
        assert isinstance(zorder, (IntFloat, type(None))), zorder
        if (self._x_side or self._y_side) and zorder is not None and not (
                1 <= zorder <= 100):
            raise ValueError(
                'The `zorder` for a user-defined artist must satisfy '
                f'1.0 <= `zorder` <= 100.0 but got `zorder` = {zorder}.'
            )
        
        if zorder is not None and zorder != self._req_zorder:
            self._req_zorder = float(zorder)
            self._stale = True
    
    def _get_zorder(self, oid: int) -> float:
        for tag in self._canvas.gettags(oid):
            if tag.startswith('zorder='):
                return float(tag[7:])
        raise ZorderNotFoundError('Zorder has not been initialized yet.')
    
    def get_zorder(self) -> float:
        return self._get_zorder(self._id)
    
    def _update_zorder(self):  # update the zorder tag
        def _delete_zorder(oid: int):
            try:
                old_zorder = self._get_zorder(oid)
            except ZorderNotFoundError:
                pass
            else:
                tk.Canvas.dtag(self._canvas, oid, f'zorder={old_zorder}')
        #> end of _delete_zorder()
        
        zorder = self._req_zorder if self._req_zorder is not None \
            else self._default_style["zorder"]
        new_tag = f'zorder={zorder}'
        
        _delete_zorder(self._id)
        tk.Canvas.addtag_withtag(self._canvas, new_tag, self._id)
        self._canvas._zorder_tags[self] = new_tag
        
        if hasattr(self, '_id_aa'):
            _delete_zorder(self._id_aa)
            tk.Canvas.addtag_withtag(self._canvas, new_tag, self._id_aa)
    
    def _bind(self, sequence: str, callback: Callable[[tk.Event], Any]):
        assert isinstance(sequence, str), sequence
        assert isinstance(callback, (Callable, type(None))), callback
        
        self._canvas.tag_bind(self._id, sequence, callback)
        if hasattr(self, '_id_aa'):
            self._canvas.tag_bind(self._id_aa, sequence, callback)
    
    def bind_enter(self, callback: Callable[[tk.Event], Any]):
        self._bind('<Enter>', callback)
    
    def bind_leave(self, callback: Callable[[tk.Event], Any]):
        self._bind('<Leave>', callback)
    
    def bind_motion(self, callback: Callable[[tk.Event], Any]):
        self._bind(MMOTION, callback)
    
    def bind_leftpress(self, callback: Callable[[tk.Event], Any]):
        self._bind(MLEFTPRESS, callback)
    
    def bind_leftmotion(self, callback: Callable[[tk.Event], Any]):
        self._bind(MLEFTMOTION, callback)
    
    def bind_leftrelease(self, callback: Callable[[tk.Event], Any]):
        self._bind(MLEFTRELEASE, callback)
    
    def bind_rightpress(self, callback: Callable[[tk.Event], Any]):
        self._bind(MRIGHTPRESS, callback)
    
    def bind_rightmotion(self, callback: Callable[[tk.Event], Any]):
        self._bind(MRIGHTMOTION, callback)
    
    def bind_rightrelease(self, callback: Callable[[tk.Event], Any]):
        self._bind(MRIGHTRELEASE, callback)
    
    def _unbind(self, sequence: str):
        assert isinstance(sequence, str), sequence
        
        self._canvas.tag_unbind(self._id, sequence)
        if hasattr(self, '_id_aa'):
            self._canvas.tag_unbind(self._id_aa, sequence)
    
    def unbind_leftpress(self):
        self._unbind(MLEFTPRESS)
    
    def unbind_leftmotion(self):
        self._unbind(MLEFTMOTION)
    
    def unbind_leftrelease(self):
        self._unbind(MLEFTRELEASE)
    
    def unbind_rightpress(self):
        self._unbind(MRIGHTPRESS)
    
    def unbind_rightmotion(self):
        self._unbind(MRIGHTMOTION)
    
    def unbind_rightrelease(self):
        self._unbind(MRIGHTRELEASE)
    
    def set_style(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_style(self):
        raise NotImplementedError
    
    def _get_legend_config(self):
        raise NotImplementedError


class _Text(_BaseArtist):
    _name = 'text'
    
    def __init__(
            self,
            canvas: tk.Canvas,
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
        
        super().__init__(canvas=canvas, antialias=False, **kwargs)
        
        self._req_bounds: dict[str, Any] = {}
        
        self._padx: tuple[float, float] = (0., 0.)
        self._pady: tuple[float, float] = (0., 0.)
        self._font: Font = Font() if font is None else font
        self._id = canvas.create_text(
            0, 0, anchor='se',
            text=text, font=self._font, state='hidden', tags=self._tags
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
        state = self._req_state
        
        if not self._stale:
            self._set_state(state)
            return
        
        self._set_state('hidden')
        
        # Update style
        root_defaults = self._root_default_style
        defaults = self._default_style
        cf = self._req_style.copy()
        cf.update({
            k: defaults.get(k, root_defaults[k])
            for k, v in cf.items() if v is None
        })
        self._font.configure(
            family=cf["family"],
            size=cf["size"],
            weight=cf["weight"],
            slant=cf["slant"],
            underline=cf["underline"],
            overstrike=cf["overstrike"]
        )
        self.itemconfigure(text=cf["text"], fill=cf["color"], angle=cf["angle"])
        
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
            self._set_state('normal')
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
            
            x, y, anchor = _get_sticky_xy(
                (x1, y1, x2, y2), sticky=sticky, padx=padx, pady=pady
            )
            
            # `anchor` must be 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw', or
            # 'center'
            if anchor != 'center':
                # Roll the anchor. e.g. 0 deg => 1 step, 45 deg => 2 step, ...
                angle = float(self.cget('angle'))
                assert 0.0 <= angle < 360.0, angle
                shift = int((angle + 22.5) // 45)  # step with 45 deg
                mapping = dict(zip(_ANCHORS, _ANCHORS[shift:] + _ANCHORS[:shift]))
                anchor = ''.join( mapping[a] for a in anchor )  # rolling
            x, y = self._req_transform(x, y, round_=True)
            if x.size == 0:
                x = y = -1
                anchor = 'se'
        
        self.coords(x, y)  # update position
        self.itemconfigure(anchor=anchor)  # update anchor
        
        # Update zorder and state
        self._set_state(state)
        self._update_zorder()
        self._padx = padx
        self._pady = pady
        self._stale = False
    
    def bbox(self, *, padding: bool = True) -> tuple[int, int, int, int] | None:
        assert isinstance(padding, bool), padding
        
        if self.cget('text') and (bbox := super().bbox()):
            x1, y1, x2, y2 = bbox
            if padding:
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
                pad = (pad, pad)
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
            "text": self.cget('text'),
            "color": self.cget('fill'),
            "angle": self.cget('angle'),
            **self._font.actual()
        }


class _Line(_BaseArtist):
    _name = 'line'
    
    def __init__(
            self,
            canvas: tk.Canvas,
            color: str | None = None,
            width: Dimension | None = None,
            dash: str | tuple[Int] | None = None,
            smooth: bool | None = None,
            datalabel: bool = False,
            **kwargs
    ):
        super().__init__(canvas=canvas, **kwargs)
        
        self._req_xy: NDArray[Float] = np.array([[]], dtype=float)
        
        self._xlims: NDArray[Float] = np.array([0., 1.], float)
        self._ylims: NDArray[Float] = np.array([0., 1.], float)
        self._datalabel: bool = datalabel
        self._datalabels: list[_DataLabel] = []
        self._datalabel_pos: tuple[int, int] | None = None
        self._datalabel_bind_ids: tuple[str, str] | None = None
        
        self._id = self._canvas.create_line(
            0, 0, 0, 0,
            fill='', width='0p', state='hidden', tags=self._tags
        )
        self._id_aa = self._canvas.create_line(
            0, 0, 0, 0,
            fill='', width='0p', state='hidden', tags=self._tags
        )
        self.set_style(color=color, width=width, dash=dash, smooth=smooth)
        
        if datalabel:
            self.bind_enter(self._datalabel_on_enter)
    
    def delete(self):
        super().delete()
        
        for dl in self._datalabels:
            dl.delete()
        self._datalabels.clear()
    
    def draw(self):
        state = self._req_state
        
        if not self._stale:
            self._set_state(state)
            return
        
        self._set_state('hidden')
        
        # Update coordinates
        if self._req_coords:
            xys = self._req_coords
        else:
            xy = self._req_transform(*self._req_xy, round_=True)
            xy = _drop_consecutive_duplicates(*xy)
            xy = _drop_linearly_redundant_points(xy)
            if self._antialias_enabled:
                xy = _cutoff_z_patterns(xy)
            xys = xy.ravel(order='F')  # x0, y0, x1, y1, x2, y2, ...
            if xy.size < 4:
                xys = (-1e4, -1e4, -1e4, -1e4)
        self.coords(*xys)
        
        # Update style
        root_defaults = self._root_default_style
        defaults = self._default_style
        cf = self._req_style.copy()
        cf.update({
            k: defaults.get(k, root_defaults[k])
            for k, v in cf.items() if v is None
        })
        cf.update(width=self._to_px(cf["width"]), fill=cf.pop('color'))
        if self._hover:
            cf.update(activewidth=cf["width"] + 2.)
        self.itemconfigure(**cf)
        
        if self._antialias_enabled:
            self._antialias(xys, **cf)
        
        # Update zorder and state
        self._update_zorder()
        self._set_state(state)
        
        # Update datalabels
        for dl in self._datalabels:
            dl.draw()
        
        self._stale = False
    
    def _antialias(
            self,
            xys: ArrayLike,
            fill: str,
            width: Dimension,
            dash: str | tuple[Int],
            smooth: bool,
            activewidth: Dimension | None = None
    ):
        width += 1.
        
        # Mix the foreground and background colors
        w = 0.3  # the weight for the foreground color
        canvas = self._canvas
        if self._antialias_bg:
            bg = self._antialias_bg()
        else:
            bg = canvas.cget('bg')
        bg = canvas.winfo_rgb(bg)
        fg = canvas.winfo_rgb(fill)
        rgb = [  # RGB (0~65535) => (0~255)
            int((f*w + b*(1. - w)) / 65535. * 255.) for f, b in zip(fg, bg)
        ]
        fill = '#{:02x}{:02x}{:02x}'.format(*rgb)
        
        # Update the transparent line
        id_og = self._id
        id_aa = self._id_aa
        canvas.itemconfigure(
            id_aa,
            fill=fill,
            width=width,
            dash=dash,
            smooth=smooth,
            activewidth=activewidth
        )
        canvas.coords(id_aa, *xys)
        canvas.tag_lower(id_aa, id_og)
    
    def set_style(
            self,
            color: str | None = None,
            width: Dimension | None = None,
            dash: str | tuple[Int] | None = None,
            smooth: bool | None = None
    ):
        assert isinstance(color, (str, type(None))), color
        assert isinstance(width, (Dimension, type(None))), width
        assert isinstance(dash, (str, tuple, type(None))), dash
        assert isinstance(smooth, (bool, type(None))), smooth
        
        old = self._req_style
        new = {
            "color": color, "width": width, "dash": dash, "smooth": smooth
        }
        new.update({ k: old.get(k, None) for k, v in new.items() if v is None })
        if new != old:
            self._req_style = new
            self._stale = True
    
    def get_style(self) -> dict[str, Any]:
        return {
            "color": self.cget('fill'),
            "width": self.cget('width'),
            "dash": self.cget('dash'),
            "smooth": self.cget('smooth')
        }
    
    def _get_legend_config(self) -> dict[str, Any]:
        return {
            "tag": self._tag,
            "color": self.cget('fill'),
            "width": self.cget('width'),
            "dash": self.cget('dash')
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
        assert length >= 2, (x, y)
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
            self._datalabels.clear()
            self._stale = True
    
    def get_data(self) -> tuple[NDArray[Float], NDArray[Float]]:
        return tuple(self._req_xy)
    
    def _find_closest_point(self, event: tk.Event) -> NDArray[IntFloat]:
        x, y = (event.x, event.y)
        xs, ys = self._req_transform(  # data => canvas coordinate
            *self._req_xy, round_=True, clip=False
        )
        i = np.argmin((xs - x)**2. + (ys - y)**2.)  # find the nearest point
        to_datacoord = self._req_transform.get_inverse()
        
        return to_datacoord(xs[i], ys[i], clip=False)  # canvas => data coordinate
    
    def _datalabel_on_enter(self, event: tk.Event):
        if self._datalabel_bind_ids is not None:
            return
        
        # Create a temporary datalabel
        data_xy = self._find_closest_point(event)
        
        dl = _DataLabel(self, tag='datalabel')
        dl._set_point(*data_xy)
        dl.draw()
        self._datalabels.append(dl)
        
        # Bind motion and leftpress callbacks
        self._datalabel_bind_ids = (
            self._canvas.bind(MMOTION, self._datalabel_on_motion, add=True),
            self._canvas.bind(MLEFTPRESS, self._datalabel_on_leftpress, add=True)
        )
    
    def _datalabel_on_motion(self, event: tk.Event):
        x, y = (event.x, event.y)
        dl = self._datalabels[-1]
        valid_items = {  # valid items which are higher than this line
            self._id_aa, dl._point._id, dl._arrow._id, dl._box._id, dl._text._id
        }
        displaylist = self._canvas.find_overlapping(
            x, y, x, y
        )[-(len(valid_items)+1):]
        try:  # find this line
            i = displaylist.index(self._id)
        except ValueError:  # not found => mouse pointer left this line
            self._datalabel_on_leave(event)
        else:
            if set(displaylist[i+1:]).difference(valid_items):  # found other
                # items higher than this line => mouse pointer left this line
                self._datalabel_on_leave(event)
            else:  # mouse pointer is moving on this line => update the datalabel
                data_xy = self._find_closest_point(event)
                dl._set_point(*data_xy)
                dl.draw()
    
    def _datalabel_on_leave(self, event: tk.Event):
        # Remove the temporary datalabel
        dl = self._datalabels[-1]
        dl.delete()
        
        # Unbind the motion and left press callbacks
        motion_id, leftpress_id = self._datalabel_bind_ids
        unbind(self._canvas, MMOTION, motion_id)
        unbind(self._canvas, MLEFTPRESS, leftpress_id)
        
        self._datalabel_bind_ids = None
    
    def _datalabel_on_leftpress(self, event: tk.Event):
        # Settle the datalabel: temporary => settled
        dl = self._datalabels[-1]
        dl._settle()
        dl.draw()
        
        # Unbind the motion and left press callbacks
        motion_id, leftpress_id = self._datalabel_bind_ids
        unbind(self._canvas, MMOTION, motion_id)
        unbind(self._canvas, MLEFTPRESS, leftpress_id)
        
        self._datalabel_bind_ids = None


class _BasePoly(_BaseArtist):
    def draw(self):
        state = self._req_state
        
        if not self._stale:
            self._set_state(state)
            return
        
        self._set_state('hidden')
        
        # Update coordinates
        xys = self._req_coords
        self.coords(*xys)
        
        # Update style
        root_defaults = self._root_default_style
        defaults = self._default_style
        cf = self._req_style.copy()
        cf.update({
            k: defaults.get(k, root_defaults[k])
            for k, v in cf.items() if v is None
        })
        cf.update(
            width=self._to_px(cf["width"]),
            fill=cf.pop('facecolor'),
            outline=cf.pop('edgecolor')
        )
        if self._hover:
            cf.update(activewidth=cf["width"] + 1.)
        self.itemconfigure(**cf)
        
        # Update zorder and state
        self._set_state(state)
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


class _Rectangle(_BasePoly):
    _name = 'rectangle'
    
    def __init__(
            self,
            canvas: tk.Canvas,
            facecolor: str | None = None,
            edgecolor: str | None = None,
            width: Dimension | None = None,
            **kwargs
    ):
        super().__init__(canvas=canvas, antialias=False, **kwargs)
        self._id = self._canvas.create_rectangle(
            0, 0, 0, 0,
            fill='', outline='', width='0p', state='hidden', tags=self._tags
        )
        self.set_style(facecolor=facecolor, edgecolor=edgecolor, width=width)


class _Oval(_BasePoly):
    _name = 'oval'
    
    def __init__(
            self,
            canvas: tk.Canvas,
            facecolor: str | None = None,
            edgecolor: str | None = None,
            width: Dimension | None = None,
            **kwargs
    ):
        super().__init__(canvas=canvas, antialias=False, **kwargs)
        self._id = self._canvas.create_oval(
            0, 0, 0, 0,
            fill='', outline='', width='0p', state='hidden', tags=self._tags
        )
        self.set_style(facecolor=facecolor, edgecolor=edgecolor, width=width)


class _Polygon(_BasePoly):
    _name = 'polygon'
    
    def __init__(
            self,
            canvas: tk.Canvas,
            facecolor: str | None = None,
            edgecolor: str | None = None,
            width: Dimension | None = None,
            smooth: bool | None = None,
            **kwargs
    ):
        super().__init__(canvas=canvas, antialias=False, **kwargs)
        self._id = self._canvas.create_polygon(
            0, 0, 0, 0,
            fill='', outline='', width='0p', state='hidden', tags=self._tags
        )
        self.set_style(
            facecolor=facecolor, edgecolor=edgecolor, width=width, smooth=smooth
        )
    
    def set_style(
            self,
            *args,
            smooth: bool | None = None,
            **kwargs
    ):
        assert isinstance(smooth, (bool, type(None))), smooth
        
        super().set_style(*args, **kwargs)
        if smooth is not None and self._req_style["smooth"] != smooth:
            self._req_style["smooth"] = smooth
            self._stale = True
    
    def get_style(self) -> dict[str, Any]:
        style = super().get_style()
        style["smooth"] = self.cget('smooth')
        
        return style


# =============================================================================
# ---- Figure Components
# =============================================================================
class _BaseComponent(_BaseElement):
    def __init__(self, plot: _Plot, canvas: tk.Canvas | None = None, **kwargs):
        assert isinstance(plot, _Plot), plot
        assert isinstance(plot._figure, Figure), plot._figure
        canvas = plot._tkwidget if canvas is None else canvas
        super().__init__(canvas, **kwargs)
        self._plot: _Plot = plot
        self._plot_tk: tk.Canvas = plot._tkwidget
        self._figure: Figure = plot._figure
    
    @property
    def _default_style(self) -> dict[str, Any]:
        return self._figure._default_style
    
    def update_theme(self):
        raise NotImplementedError
    
    def bbox(self):
        raise NotImplementedError


class _Axis(_BaseComponent):
    def __init__(
            self,
            plot: _Plot,
            side: Literal['r', 'b', 'l', 't'],
            *args, 
            **kwargs
    ):
        super().__init__(plot, *args, **kwargs)
        self._side: Literal['r', 'b', 'l', 't'] = side
        self._label: _Text = _Text(
            self._canvas, text='', tag=f'{self._tag}.label.text'
        )
    
    def update_theme(self):
        self._label._stale = True
    
    def draw(self):
        self._label.draw()
    
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


class _Ticks(_BaseComponent):
    def __init__(
            self,
            plot: _Plot,
            side: Literal['r', 'b', 'l', 't'],
            *args, 
            **kwargs
    ):
        super().__init__(plot, *args, **kwargs)
        
        self._req_ticks_enabled: bool = False
        self._req_labels_enabled: bool = False
        self._req_scientific: Int | None = None
        self._req_xys: tuple[
            Dimension, Dimension, Dimension, Dimension
        ] | None = None
        self._req_transform: _BaseTransform1D = _FirstOrderPolynomial()
        self._req_limits: list[IntFloat | None] = [None, None]
        self._req_margins: list[Dimension | None] = [None, None]
        self._req_scale: Literal['linear', 'log'] = 'linear'
        
        self._dummy_xys: tuple[
            Dimension, Dimension, Dimension, Dimension
        ] | None = None
        self._dummy_transform: _BaseTransform1D = _FirstOrderPolynomial()
        self._dummy_limits: list[IntFloat] = [_PMIN, _MAX]
        self._dummy_vertical: bool = False
        self._dummy_n_chars: Int = max( len(str(l)) for l in [_PMIN, _MAX] )
        self._dummy_size: IntFloat = 0
        self._dummy_label: _Text = _Text(
            self._canvas, text='', tag=f'{self._tag}.labels.text'
        )
        self._dummy_tick: _Line = _Line(
            self._canvas, tag=f'{self._tag}.ticks.line'
        )
        self._side: Literal['r', 'b', 'l', 't'] = side
        self._growing_side: int = {"r": 0, "b": 1, "l": 2, "t": 3}[side]
        self._stale: bool = True
        self._fitted_labels: dict[str, Any] = {}
        self._limits: list[IntFloat] = [_PMIN, _MAX]
        self._margins: list[IntFloat] = [0., 0.]
        self._growing_p: IntFloat = 0.
        self._ticks_cdata: NDArray[Float] = np.array([], dtype=float)
        self._labels: list[_Text] = []
        self._ticks: list[_Line] = []
    
    def update_theme(self):
        self._stale = True
        
        for label in self._labels:
            label._stale = True
        for tick in self._ticks:
            tick._stale = True
    
    def draw(
            self
    ) -> tuple[dict[str, Any], _BaseTransform1D] | None:
        self._dummy_label._set_state('hidden')
        texts, positions = self._generate_label_values()
        
        if not self._stale:
            return
        
        canvas = self._canvas
        if self._req_labels_enabled:
            tag = f'{self._tag}.labels.text'
            font = self._dummy_label._font
            style = self._dummy_label._req_style.copy()
            style.pop('text', None)
            bounds = self._dummy_label._req_bounds.copy()
            bounds.pop('xys', None)
            
            number_increase = len(texts) - len(self._labels)
            if number_increase < 0:  # delete labels
                for i in range(-number_increase):
                    self._labels.pop().delete()
            elif number_increase > 0:  # create labels
                self._labels.extend(
                    _Text(canvas, text='', font=font, tag=tag)
                    for i in range(number_increase)
                )
            
            for label, text, xys in zip(self._labels, texts, positions):
                label.set_style(text=text, **style)
                label.set_bounds(xys=xys, **bounds)
                label.draw()
        else:
            for label in self._labels:
                label.delete()
            self._labels.clear()
        
        if self._req_ticks_enabled:
            tag = f'{self._tag}.ticks.line'
            style = self._dummy_tick._req_style
            p = self._growing_p
            if self._side in ('t', 'b'):
                positions = [ (x1, p-2, x2, p+2) for x1, y1, x2, y2 in positions ]
            else:
                positions = [ (p-2, y1, p+2, y2) for x1, y1, x2, y2 in positions ]
            
            number_increase = len(positions) - len(self._ticks)
            if number_increase < 0:  # delete lines
                for i in range(-number_increase):
                    self._ticks.pop().delete()
            elif number_increase > 0:  # create lines
                self._ticks.extend(
                    _Line(canvas, tag=tag) for i in range(number_increase)
                )
            
            for tick, xys in zip(self._ticks, positions):
                tick.set_style(**style)
                tick.set_coords(*xys)
                tick.draw()
        else:
            for tick in self._ticks:
                tick.delete()
            self._ticks.clear()
        
        self._stale = False
    
    def _draw_dummy(self):
        texts, positions = self._generate_label_values(dummy=True)
        
        # Find possibly largest label
        n_chars = [ max( len(t) for t in tx.split('\n', 1) ) for tx in texts ]
        i = np.argmax(n_chars)
        text, xys = texts[i], positions[i]
        
        # Update the text item and draw
        label = self._dummy_label
        label.set_style(text=text)
        label.set_bounds(xys=xys)
        label.set_state('normal')
        label.draw()
        
        # Update current size of the text item
        tx1, ty1, tx2, ty2 = label.bbox()
        w, h = (tx2 - tx1), (ty2 - ty1)
        angle = float(label.cget('angle'))
        vertical = (angle - 90) % 180 == 0  # text direction
        self._dummy_n_chars = n_chars[i]
        self._dummy_size = max(w if self._side in 'tb' else h, 1)
        self._dummy_vertical = vertical
    
    def bbox(
            self,
            *,
            dummy: bool = False
    ) -> tuple[int, int, int, int] | None:
        assert isinstance(dummy, bool), dummy
        
        if dummy:
            if self._req_labels_enabled:
                return self._dummy_label.bbox()
            return None
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
    
    def set_ticks(
            self,
            enable: bool = True,
            color: str | None = None,
            width: Dimension | None = None,
            smooth: bool | None = None
    ):
        assert isinstance(enable, bool), enable
        
        self._dummy_tick.set_style(color=color, width=width, smooth=smooth)
        self._req_ticks_enabled = enable
        self._stale = True
    
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
        assert isinstance(enable, bool), enable
        assert isinstance(scientific, (Int, type(None))), scientific
        assert scientific is None or scientific > 0, scientific
        
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
        
        self._req_labels_enabled = enable
        if scientific is not None:
            self._req_scientific = scientific
        
        self._stale = True
    
    def get_labels(self) -> list[_Text]:
        return self._labels
    
    def _set_limits(
            self,
            min_: IntFloat | None = None,
            max_: IntFloat | None = None,
            margins: ArrayLike | Dimension | None = [None, None]
    ):
        assert isinstance(min_, (IntFloat, type(None))), min_
        assert isinstance(max_, (IntFloat, type(None))), max_
        assert len(margins) == 2, margins
        assert isinstance(margins[0], (Dimension, type(None))), margins
        assert isinstance(margins[1], (Dimension, type(None))), margins
        
        self._req_limits[:] = (min_, max_)
        self._req_margins[:] = margins
    
    def set_limits(
            self,
            min_: IntFloat | None = None,
            max_: IntFloat | None = None,
            margins: ArrayLike | Dimension | None = [None, None]
    ):
        assert isinstance(min_, (IntFloat, type(None))), min_
        assert isinstance(max_, (IntFloat, type(None))), max_
        if isinstance(margins, Dimension):
            margins = [margins, margins]
        margins = list(margins)
        assert len(margins) == 2, margins
        assert isinstance(margins[0], (Dimension, type(None))), margins
        assert isinstance(margins[1], (Dimension, type(None))), margins
        
        old_limits = self._req_limits.copy()
        if min_ is not None:
            self._req_limits[0] = min_
        if max_ is not None:
            self._req_limits[1] = max_
        if all( l is not None for l in self._req_limits ) \
                and self._req_limits[0] > self._req_limits[1]:
            msg = 'the value of `min_` must be smaller than or equal to ' \
                f'the value of `max_` but got {self._req_limits[0]} and ' \
                f'{self._req_limits[1]}.'
            self._req_limits[:] = old_limits
            raise ValueError(msg)
        
        for i, margin in enumerate(margins):
            if margin is not None:
                self._req_margins[i] = margin
    
    def get_limits(self) -> tuple[list[IntFloat], list[Dimension]]:
        return self._limits, self._margins
    
    def set_scale(self, scale: Literal['linear', 'log'] | None = None):
        assert scale in ('linear', 'log', None), scale
        
        if scale is not None and scale != self._req_scale:
            self._req_scale = scale
            self._stale = True
    
    def get_scale(self) -> Literal['linear', 'log']:
        return self._scale
    
    def _set_bounds_and_transform(
            self,
            xys: tuple[
                Dimension | None,
                Dimension | None,
                Dimension | None,
                Dimension | None
            ],
            dlimits: ArrayLike,
            climits: ArrayLike,
            growing_p: Dimension | None = None,
            dummy: bool = False
    ):
        assert isinstance(xys, tuple) and len(xys) == 4, xys
        assert all( isinstance(p, (Dimension, type(None))) for p in xys ), xys
        assert sum( p is None for p in xys ) == 1, xys
        assert isinstance(growing_p, (Dimension, type(None))), growing_p
        assert isinstance(dummy, bool), dummy
        
        (dmin, dmax), (cmin, cmax) = sorted(dlimits), climits
        assert cmin <= cmax, climits
        
        # Add margins
        default_marg1, default_marg2 = self._default_style[f"{self._tag}.margins"]
        req_marg1, req_marg2 = self._req_margins
        mrg1 = self._to_px(default_marg1 if req_marg1 is None else req_marg1)
        mrg2 = self._to_px(default_marg2 if req_marg2 is None else req_marg2)
        assert mrg1 >= 0. and mrg2 >= 0., (mrg1, mrg2)
        cmin, cmax = (cmin + mrg1), (cmax - mrg2)
        if cmin > cmax:
            cmin = cmax = round((cmin + cmax) / 2.)
        new_climits = [cmin, cmax]
        
        # Fetch the min and max values set by the user
        req_dmin, req_dmax = self._req_limits
        if req_dmin is not None:
            dmin = req_dmin
        if req_dmax is not None:
            dmax = req_dmax
        assert dmin <= dmax, (dmin, dmax)
        
        if linear_scale := (self._req_scale == 'linear'):
            tf_cls = _FirstOrderPolynomial
            dmin, dmax = max(dmin, _MIN), min(dmax, _MAX)
        else:  # log scale
            tf_cls = _Logarithm
            dmin, dmax = max(dmin, _PMIN), min(dmax, _MAX)
        dmax = max(dmax, dmin)
        raw_dmin, raw_dmax = (dmin, dmax)
        
        sci = 1 if not linear_scale \
            else self._req_scientific if self._req_scientific is not None \
            else self._default_style[f"{self._tag}.labels.scientific"]
        
        if dummy:
            max_n_labels = 2
        else:
            # Calculate the max number of labels fitting in the space
            if self._side in 'tb':  # top or bottom
                fixed_size = self._dummy_vertical
            else:  # left or right
                fixed_size = not self._dummy_vertical
            n_chars = self._dummy_n_chars
            size = self._dummy_size
            if fixed_size:
                size *= 1.2
            else:
                size *= (max(n_chars, sci) + 2) / n_chars
            max_n_labels = max(int((cmax - cmin) // size), 0)
        
        # Find appropriate min and max values and the actual number of labels
        if linear_scale:
            if max_n_labels <= 1:
                self._fitted_labels = {
                    "start": dmin, "stop": dmax, "num": max_n_labels
                }
            else:
                ## Use `Decimal` to represent numbers exactly
                dmin, dmax = Decimal(str(dmin)), Decimal(str(dmax))
                
                ## Find the nearest a*10^b, where a is an integer
                step = (dmax - dmin) / (max_n_labels - 1)  # excluding limits
                log_e, log_s = divmod(float(step.log10()), 1)
                log_e, log_s = Decimal(str(log_e)), Decimal(str(log_s))
                if (s := round(10**log_s, 0)) > 5:
                    s = 1
                    log_e += 1
                elif s > 2:
                    s = 5
                assert int(s) in (1, 2, 5), s
                step = s * 10**log_e
                
                dmin = (dmin / step).quantize(
                    Decimal('1.'), rounding=ROUND_FLOOR
                ) * step
                dmax = (dmax / step).quantize(
                    Decimal('1.'), rounding=ROUND_CEILING
                ) * step
                n = (dmax - dmin) / step + 1
                dmin, dmax, n = float(dmin), float(dmax), int(n)
                
                if dmin < _MIN or dmax > _MAX:
                    dmin, dmax = max(dmin, _MIN), min(dmax, _MAX)
                    self._fitted_labels = {
                        "start": raw_dmin, "stop": raw_dmax, "num": 2
                    }
                else:
                    self._fitted_labels = {
                        "start": dmin, "stop": dmax, "num": n
                    }
        else:  # log scale
            if max_n_labels <= 1:
                self._fitted_labels = {
                    "case": 0,
                    "start": dmin, "stop": dmax, "num": max_n_labels
                }
            else:
                ## Find the nearest a*10^b, where a is an integer
                e1, log_s1 = divmod(np.log10(dmin), 1)
                e1, log_s1 = Decimal(str(e1)), Decimal(str(log_s1))
                s1 = (10**log_s1).quantize(
                    Decimal('1.'), rounding=ROUND_FLOOR
                )
                
                ## Find the nearest a*10^b, where a is an integer
                e2, log_s2 = divmod(np.log10(dmax), 1)
                e2, log_s2 = Decimal(str(e2)), Decimal(str(log_s2))
                s2 = (10**log_s2).quantize(
                    Decimal('1.'), rounding=ROUND_CEILING
                )
                
                # Find `n`, the largest integer smaller than `max_n_labels`
                ## a*10^b
                case = 1
                n = int((e2 - e1) * 9 + 1 - (s1 - 1) + (s2 - 1))
                if n > max_n_labels:
                    case = 2
                    ## Only 1*10^b
                    if s2 > 1:
                        e2 += 1
                    s1 = s2 = 1
                    n = int(e2 - e1 + 1)
                    if n > max_n_labels:
                        case = 3
                        n = 2
                dmin, dmax = float(s1 * 10**e1), float(s2 * 10**e2)
                
                if dmin < _PMIN or dmax > _MAX:
                    dmin, dmax = max(dmin, _MIN), min(dmax, _MAX)
                    self._fitted_labels = {
                        "case": 0,
                        "start": raw_dmin, "stop": raw_dmax, "num": 2
                    }
                else:
                    self._fitted_labels = {
                        "case": case,
                        "s1": int(s1), "s2": int(s2),
                        "e1": float(e1), "e2": float(e2)
                    }
        
        # Set the exact data limits if they are requested
        if req_dmin is not None:
            dmin = req_dmin
        if req_dmax is not None:
            dmax = req_dmax
        
        new_dlimits = [dmin, dmax] if dlimits[0] < dlimits[1] else [dmax, dmin]
        tf = tf_cls.from_points(new_dlimits, new_climits)
        
        if dummy:
            self._dummy_xys = xys
            self._dummy_transform = tf
            self._dummy_limits = dlimits
            return
        
        if xys != self._req_xys:
            self._req_xys = xys
            self._stale = True
        if tf != self._req_transform:
            self._req_transform = tf
            self._stale = True
        self._limits = [dmin, dmax]
        self._margins = [mrg1, mrg2]
        self._growing_p = growing_p
    
    def _get_transform(self) -> _BaseTransform1D:
        return self._req_transform
    
    def _generate_label_values(
            self,
            dummy: bool = False
    ) -> tuple[list[str], NDArray[Float]]:
        assert isinstance(dummy, bool), dummy
        
        if dummy:
            x1, y1, x2, y2 = self._dummy_xys
            transform = self._dummy_transform
        else:
            x1, y1, x2, y2 = self._req_xys
            transform = self._req_transform
        linear_scale = self._req_scale == 'linear'
        sci = 1 if not linear_scale \
            else self._req_scientific if self._req_scientific is not None \
            else self._default_style[f"{self._tag}.labels.scientific"]
        
        # Make the labels' values
        if dummy:
            data = np.asarray(self._dummy_limits, dtype=float)
        elif linear_scale:
            data = np.linspace(**self._fitted_labels, endpoint=True, dtype=float)
        else:  # log scale
            fitted = self._fitted_labels
            case = fitted["case"]
            if case == 0:
                data = np.linspace(
                    fitted["start"], fitted["stop"], fitted["num"], dtype=float
                )
            elif case == 1:
                data = (
                    (
                        10**np.arange(fitted["e1"], fitted["e2"]+1, dtype=float)
                    )[:, None] * 
                    np.arange(1., 9+1, dtype=float)[None, :]
                ).ravel()
                start = int(fitted["s1"] - 1)
                stop = int(data.size - (9 - fitted["s2"]))
                data = data[start:stop]
            elif case == 2:
                data = 10**np.arange(fitted["e1"], fitted["e2"]+1, dtype=float)
            else:
                data = 10**np.asarray([fitted["e1"], fitted["e2"]], dtype=float)
        assert np.isfinite(data).all(), data
        
        ## Filter out the out-of-range values
        if not dummy:
            req_dmin, req_dmax = self._req_limits
            if req_dmin is not None:
                data = data[data >= req_dmin]
            if req_dmax is not None:
                data = data[data <= req_dmax]
        
        # Formatting
        texts = [
            t.replace('e', '\ne') if 'e' in (t := '{0:.{1}g}'.format(d, sci))
                else t + '\n'
            for d in data
        ]
        if not dummy and not linear_scale:
            ## Omit the exponential part if base != 1
            texts = [
                b + '\n' if (b := t.split('\n', 1)[0]) != '1' else t
                for t in texts
            ]
        if self._side != 'b':
            ## b\ne^a => e^a\nb (put the exponent above the base)
            texts = [ '\n'.join(t.split('\n', 1)[::-1]) for t in texts ]
        
        # Transform the data coordinates into the canvas coordinates
        if dummy:
            cdata = np.zeros_like(data)
        else:
            cdata = transform(data, round_=True)
            self._ticks_cdata = cdata
        
        if self._side in ('t', 'b'):
            positions = [ (x, y1, x, y2) for x in cdata ]
        else:  # left or right
            positions = [ (x1, y, x2, y) for y in cdata ]
        
        return texts, positions


class _Frame(_BaseComponent):
    def __init__(self, plot: _Plot, *args, **kwargs):
        super().__init__(plot, *args, **kwargs)
        
        self._req_facecolor: str | None = None
        self._req_grid_enabled: dict[str, bool | None] = dict.fromkeys('rblt')
        self._req_grid_cdata: dict[str, NDArray[Float]] = {}
        
        self._stale_grid: bool = True
        
        self._dummy_grid: dict[str, _Line] = {
            "r": _Line(self._canvas, tag=f'{self._tag}.grid.line'),
            "b": _Line(self._canvas, tag=f'{self._tag}.grid.line'),
            "l": _Line(self._canvas, tag=f'{self._tag}.grid.line'),
            "t": _Line(self._canvas, tag=f'{self._tag}.grid.line')
        }
        self._cover: _Polygon = _Polygon(
            self._canvas, tag=f'{self._tag}.cover.polygon'
        )
        self._edge: _Rectangle = _Rectangle(
            self._canvas, tag=f'{self._tag}.edge.rectangle'
        )
        self._grid: dict[str, list[_Line]] = {"r": [], "b": [], "l": [], "t": []}
    
    def update_theme(self):
        self._stale_grid = True
        
        self._cover._stale = True
        self._edge._stale = True
        for lines in self._grid.values():
            for line in lines:
                line._stale = True
    
    def draw(self):
        canvas = self._canvas
        
        # Update frame's facecolor
        default_color = self._default_style["frame"]["facecolor"]
        bg = default_color if self._req_facecolor is None else self._req_facecolor
        canvas.configure(background=bg)
        
        # Draw frame edge
        self._edge.draw()
        
        # Draw background cover
        w, h = self._plot._size
        x1, y1, x2, y2 = self._edge.get_coords()
        self._cover.set_coords(  # covers whole canvas except the frame
            -1, -1,   w,  -1,   w,   h,   -1,  h,   -1, y1,
            x1, y1,   x1, y2,   x2, y2,   x2, y1,   -1, y1
        )
        self._cover.draw()
        
        if not self._stale_grid:
            return
        
        # Draw grid
        default_enabled = self._default_style[f'{self._tag}.grid.enabled']
        tag = f'{self._tag}.grid.line'
        for side, lines in self._grid.items():
            if (enabled := self._req_grid_enabled[side]) is None:
                enabled = side in default_enabled
            if not enabled:
                for line in lines:
                    line.delete()
                lines.clear()
                continue
            
            style = self._dummy_grid[side]._req_style
            positions = self._req_grid_cdata[side]
            if side in ('t', 'b'):
                positions = [ (p, y1+1, p, y2-1) for p in positions ]
            else:
                positions = [ (x1+1, p, x2-1, p) for p in positions ]
            
            number_increase = len(positions) - len(lines)
            if number_increase < 0:  # delete lines
                for i in range(-number_increase):
                    lines.pop().delete()
            elif number_increase > 0:  # create lines
                lines.extend(
                    _Line(canvas, tag=tag) for i in range(number_increase)
                )
            
            for line, xys in zip(lines, positions):
                line.set_style(**style)
                line.set_coords(*xys)
                line.draw()
        
        self._stale_grid = False
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._edge.bbox()
    
    def set_coords(self, *args, **kwargs):
        self._edge.set_coords(*args, **kwargs)
        self._stale_grid |= self._edge._stale
    
    def set_facecolor(self, color: str | None = None):
        assert isinstance(color, (str, type(None))), color
        
        self._req_facecolor = color
    
    def get_facecolor(self) -> str:
        return self._canvas.cget('background')
    
    def set_edgecolor(self, color: str | None = None):
        assert isinstance(color, (str, type(None))), color
        
        self._edge.set_style(edgecolor=color)
    
    def get_edgecolor(self) -> str:
        return self._edge.get_style()["edgecolor"]
    
    def set_grid(
            self,
            side: Literal['r', 'b', 'l', 't'],
            enable: bool = True,
            color: str | None = None,
            width: Dimension | None = None,
            smooth: bool | None = None
    ):
        assert side in ('r', 'b', 'l', 't'), side
        assert isinstance(enable, bool), enable
        
        if enable != self._req_grid_enabled[side]:
            self._req_grid_enabled[side] = enable
        
        self._dummy_grid[side].set_style(color=color, width=width, smooth=smooth)
        self._stale_grid = True
    
    def get_grid(self) -> dict[str, list[_Line]]:
        return self._grid
    
    def _set_grid_cdata(
            self,
            r: NDArray[Float],
            b: NDArray[Float],
            l: NDArray[Float],
            t: NDArray[Float]
    ):
        cdata = {"r": r, "b": b, "l": l, "t": t}
        if not self._req_grid_cdata or not all(
                np.array_equal(cdata[side], self._req_grid_cdata[side])
                for side in 'rblt'
        ):
            self._req_grid_cdata = cdata
            self._stale_grid = True


class _Legend(_BaseComponent):
    def __init__(self, plot: _Plot, *args, **kwargs):
        tag = kwargs.pop('tag')
        canvas = ScrolledCanvas(
            plot._tkwidget,
            *args,
            container_type='tk',
            propagate_geometry=False,
            scroll_orient='both',
            bind_mousewheel_add=False,
            **kwargs
        )
        canvas._zorder_tags: dict[_BaseArtist, str] = {}
        canvas._figure = plot._figure
        super().__init__(plot, canvas=canvas, tag=tag)
        
        self._req_enabled: bool = False
        self._req_facecolor: str | None = None
        self._req_edge: dict[str, Any] = dict.fromkeys(['edgecolor', 'edgewidth'])
        self._req_bounds: dict[str, Any] = dict.fromkeys(['xys', 'width', 'padx'])
        self._req_ipadding: dict[str, tuple[Dimension, Dimension] | None] \
            = dict.fromkeys(['ipadx', 'ipady'])
        self._req_symbols: list[dict[str, Any]] = []
        self._req_labels: list[str] = []
        
        self._padx: tuple[Dimension, Dimension] = (0., 0.)
        self._dummy_label: _Text = _Text(canvas, text='', tag=f'{self._tag}.text')
        self._symbols: list[_Line] = []
        self._labels: list[_Text] = []
        
        self._id: int = self._plot_tk.create_window(
            0, 0, anchor='nw', window=canvas.container, state='hidden'
        )
    
    def update_theme(self):
        for artist in self._canvas._zorder_tags:
            artist._stale = True
    
    def draw(self):
        defaults = self._default_style
        cf = self._req_bounds.copy()
        x1, y1, x2, y2 = self._req_bounds["xys"]
        cf.update({
            k: defaults[f"{self._tag}.{k}"]
            for k, v in cf.items() if v is None
        })
        px1, px2 = padx = self._to_px(cf["padx"])
        width = self._to_px(cf["width"]) if self._req_enabled else 0
        x1 = x2 - px2 - width
        if x1 > x2:
            x1 = x2 = round((x1 + x2) / 2.)
        
        self._plot_tk.coords(self._id, x1, y1)
        self._plot_tk.itemconfigure(self._id, width=width, height=y2-y1+1)
        
        self._plot_tk.itemconfigure(self._id, state='hidden')
        
        canvas = self._canvas
        if (bg := self._req_facecolor) is None:
            bg = defaults[f"{self._tag}.facecolor"]
        canvas.configure(background=bg)
        
        cf = self._req_edge.copy()
        cf.update({
            k: defaults[f"{self._tag}.{k}"]
            for k, v in cf.items() if v is None
        })
        canvas.container.configure(
            background=cf["edgecolor"], padx=cf["edgewidth"], pady=cf["edgewidth"]
        )
        
        labels, symbols = self._labels, self._symbols
        for symbol in symbols:
            symbol.delete()
        symbols.clear()
        for label in labels:
            label.delete()
        labels.clear()
        
        if self._req_enabled and self._req_labels:
            cf = self._req_ipadding.copy()
            cf.update({
                k: defaults[f"{self._tag}.{k}"]
                for k, v in cf.items() if v is None
            })
            ipx1, ipy1 = self._to_px((cf["ipadx"][0], cf["ipady"][0]))
            
            sym_width = self._to_px(defaults[f"{self._tag}.symbols.width"])
            sym_width = self._to_px(defaults[f"{self._tag}.symbols.width"])
            
            sym_x1 = ipx1
            sym_x2 = sym_x1 + sym_width
            lab_x1 = sym_x2 + 1
            
            tag = f'{self._tag}.labels.text'
            font = self._dummy_label._font
            style = self._dummy_label._req_style.copy()
            style.pop('text', None)
            bounds = self._dummy_label._req_bounds.copy()
            bounds.pop('xys', None)
            
            req_symbols, req_labels = self._req_symbols, self._req_labels
            for i, (text, sym_kw) in enumerate(zip(req_labels, req_symbols)):
                if i == 0:
                    xys = (lab_x1, ipy1, None, None)
                else:
                    xys = (lab_x1, y2 + 1, None, None)
                label = _Text(canvas, text=text, font=font, tag=tag)
                label.set_style(**style)
                label.set_bounds(xys=xys, **bounds)
                label.draw()
                labels.append(label)
                
                x1, y1, x2, y2 = label.bbox(padding=False)
                y = round((y1 + y2) / 2.)
                xys2 = (sym_x1, y, sym_x2, y)
                symbol = _Line(canvas, **sym_kw)
                symbol.set_coords(*xys2)
                symbol.draw()
                symbols.append(symbol)
        
        state = 'normal' if self._req_enabled else 'hidden'
        self._plot_tk.itemconfigure(self._id, state=state)
        self._padx = padx
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        if not self._req_enabled:
            return None
        
        bbox = self._plot_tk.bbox(self._id)
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        p1, p2 = self._padx
        return (x1-p1, y1, x2+p2, y2)
    
    def set_facecolor(self, color: str | None = None):
        assert isinstance(color, (str, type(None))), color
        
        if color is not None:
            self._req_facecolor = color
    
    def get_facecolor(self) -> str:
        return self._canvas.get('background')
    
    def set_edge(
            self, color: str | None = None, width: Dimension | None = None
    ):
        assert isinstance(color, (str, type(None))), color
        assert isinstance(width, (Dimension, type(None))), width
        
        if color is not None:
            self._req_edge["edgecolor"] = color
        if width is not None:
            self._req_edge["edgewidth"] = width
    
    def get_edge(self) -> dict[str, Any]:
        return {
            "color": self._canvas.container["background"],
            "width": self._canvas.container["padding"]
        }
    
    def set_enabled(self, enable: bool = True):
        self._req_enabled = enable
    
    def _set_bounds(
            self,
            xys: tuple[Dimension, Dimension, Dimension, Dimension]
    ):
        assert isinstance(xys, (tuple, type(None))), xys
        assert all( isinstance(p, (Dimension, type(None))) for p in xys ), xys
        
        self._req_bounds["xys"] = xys
    
    def set_size(
            self,
            width: Dimension | None = None,
            padx: tuple[Dimension, Dimension] | None = None
    ):
        assert isinstance(width, (Dimension, type(None))), width
        assert isinstance(padx, (tuple, type(None))), padx
        if padx is not None:
            assert len(padx) == 2, padx
            assert all( isinstance(p, Dimension) for p in padx ), padx
        
        if width is not None:
            self._req_bounds["width"] = width
        if padx is not None:
            self._req_bounds["padx"] = padx
    
    def get_size(self) -> dict[str, Any]:
        return {"width": self._canvas.container.winfo_width(), "padx": self._padx}
    
    def set_ipadding(
            self,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None
    ):
        assert isinstance(padx, (Dimension, tuple, type(None))), padx
        assert isinstance(pady, (Dimension, tuple, type(None))), pady
        
        padding = []
        for pad in [padx, pady]:
            if isinstance(pad, Dimension):
                pad = (pad, pad)
            elif isinstance(pad, tuple):
                assert len(pad) == 2, [padx, pady]
                assert all( isinstance(p, Dimension) for p in pad ), [padx, pady]
            padding.append(pad)
        padx, pady = padding
        
        old = self._req_ipadding
        new = {"ipadx": padding[0], "ipady": padding[1]}
        new.update({ k: old.get(k, None) for k, v in new.items() if v is None })
        
        self._req_ipadding = new
    
    def set_labels(
            self,
            color: str | None = None,
            angle: IntFloat | None = None,
            family: str | None = None,
            size: Int | None = None,
            weight: str | None = None,
            slant: str | None = None,
            underline: bool | None = None,
            overstrike: bool | None = None,
            padx: Dimension | tuple[Dimension, Dimension] | None = None,
            pady: Dimension | tuple[Dimension, Dimension] | None = None
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
    
    def get_labels(self) -> list[_Text]:
        return self._labels
    
    def _set_contents(
            self, lines: list[_Line], labels: list[str]
    ):
        assert isinstance(lines, list), lines
        assert isinstance(labels, list), labels
        assert len(lines) == len(labels), (len(lines), len(labels))
        assert all( isinstance(line, _Line) for line in lines ), lines
        
        symbols_kw = [ line._get_legend_config() for line in lines ]
        
        self._req_symbols = symbols_kw
        self._req_labels = labels


class _DataLabel(_BaseComponent):
    def __init__(self, line: _Line, **kwargs):
        assert isinstance(line, _Line), line
        
        plot: _Plot = line._canvas._wrapper
        super().__init__(plot, **kwargs)
        assert self._canvas is line._canvas, (self._canvas, line._canvas)
        
        self._req_offset: tuple[Dimension, Dimension] | None = None  # text offset
        self._req_scientific: Int | None = None
        self._req_xy: NDArray[Float] | None = None  # data point
        
        self._settled: bool = False  # temporary or settled
        self._line: _Line = line  # mother line
        self._line_width: str | None = None
        self._box_edgewidth: str | None = None
        self._point_edgewidth: str | None = None
        self._arrow_edgewidth: str | None = None
        self._motion_start: tuple[Int, Int, Int, Int, Int, Int] \
            = (0, 0, 0, 0, 0, 0)
        self._drag_text_xy: tuple[Int, Int] | None = None  # latest dragged position
        
        self._arrow: _Polygon = _Polygon(
            self._canvas, movable=True, tag=f'{self._tag}.arrow.polygon'
        )
        self._point: _Oval = _Oval(
            self._canvas, movable=True, tag=f'{self._tag}.point.oval'
        )
        self._box: _Rectangle = _Rectangle(
            self._canvas, movable=True, tag=f'{self._tag}.box.rectangle'
        )
        self._text: _Text = _Text(
            self._canvas, text='', movable=True, tag=f'{self._tag}.text'
        )
        
        for artist in [self._arrow, self._point, self._box, self._text]:
            artist.bind_enter(self._on_enter)
            artist.bind_leave(self._on_leave)
            artist.bind_leftpress(self._on_leftpress)
            artist.bind_leftmotion(self._on_leftmotion)
            artist.bind_rightpress(self._on_rightpress)
    
    def update_theme(self):
        for artist in (self._arrow, self._point, self._box, self._text):
            artist._stale = True
        self._line._stale = True  # force update `self`
    
    def draw(self):
        data_xy = self._req_xy
        sci = self._req_scientific if self._req_scientific is not None \
            else self._default_style[f"{self._tag}.scientific"]
        offset = self._to_px(
            self._req_offset if self._req_offset is not None \
            else self._default_style[f"{self._tag}.offset"]
        )
        
        # Draw text
        bg = self._line.cget('fill')
        fg = contrast_color(bg)
        label = self._line.get_label()
        point = '({0:.{2}g}, {1:.{2}g})'.format(*data_xy, sci)
        x0, y0 = self._line._req_transform(*data_xy, round_=True, clip=False)
        if self._drag_text_xy is None:
            x1, y1 = (x0 + offset[0]), (y0 + offset[1])
        else:
            x1, y1 = self._drag_text_xy
        self._text.set_style(text='\n'.join([label, point]), color=fg)
        self._text.set_bounds((x1, y1, x1, y1))
        self._text.draw()
        if not self._settled:  # magnify the font size
            req_style = self._text._req_style
            original_text_size = req_style["size"]
            try:
                req_style["size"] = self._text._font.actual('size') + 1
                self._text._stale = True
                self._text.draw()
            finally:
                req_style["size"] = original_text_size
                self._text._stale = True
        
        # Draw box
        box_x1, box_y1, box_x2, box_y2 = self._text.bbox()
        self._box.set_style(facecolor=bg)
        self._box.set_coords(box_x1, box_y1, box_x2, box_y2)
        self._box.draw()
        
        # Draw point
        if not (w := float(self._line.cget('activewidth'))):
            w = float(self._line.cget('width'))
        point_r = int(np.ceil(w / 2)) + 2
        point_x1, point_y1 = (x0 - point_r), (y0 - point_r)
        point_x2, point_y2 = (x0 + point_r), (y0 + point_r)
        self._point.set_style(facecolor=bg)
        self._point.set_coords(point_x1, point_y1, point_x2, point_y2)
        self._point.draw()
        
        # Draw arrow (pratically, it's a triangle)
        s = min(box_x2-box_x1+1, box_y2-box_y1+1) / 2.  # shortest side / 2
        tan_dx, tan_dy = (x1 - x0), (y1 - y0)  # tangent vector
        dist = np.sqrt(tan_dx**2 + tan_dy**2)
        tan_dx, tan_dy = (tan_dx / dist), (tan_dy / dist)  # unit tangent vector
        r = s if dist == s else min(max(dist/abs(dist-s)*s/2.*0.7, 1.), s)
        ppd_dx, ppd_dy = int(tan_dy * r), int(-tan_dx * r)  # perpendicular vector
        x2, y2 = (x1 - ppd_dx), (y1 - ppd_dy)
        x1, y1 = (x1 + ppd_dx), (y1 + ppd_dy)
        self._arrow.set_style(facecolor=bg)
        self._arrow.set_coords(x0, y0, x1, y1, x2, y2)
        self._arrow.draw()
    
    def bbox(self) -> tuple[int, int, int, int] | None:
        pass
    
    def delete(self):
        self._on_leave()
        for artist in (self._arrow, self._point, self._box, self._text):
            artist.delete()
        self._line._datalabels.remove(self)
    
    def set_text(
            self,
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
            scientific: Int | None = None,
            offset: Dimension | tuple[Dimension, Dimension] | None = None
    ):
        assert isinstance(offset, (Dimension, tuple, type(None))), offset
        assert isinstance(scientific, (Int, type(None))), scientific
        assert scientific is None or scientific > 0, scientific
        
        if isinstance(offset, Dimension):
            offset = (offset, offset)
        elif isinstance(offset, tuple):
            assert len(offset) == 2, offset
            assert all( isinstance(d, Dimension) for d in offset ), offset
        
        self._text.set_style(
            color=color,
            angle=angle,
            family=family,
            size=size,
            weight=weight,
            slant=slant,
            underline=underline,
            overstrike=overstrike
        )
        self._text.set_bounds(padx=padx, pady=pady)
        
        if scientific is not None:
            self._req_scientific = scientific
        if offset is not None:
            self._req_offset = offset
            self._drag_text_xy = None
    
    def get_text(self) -> _Text:
        return self._text
    
    def _set_point(self, x: IntFloat, y: IntFloat):
        assert isinstance(x, IntFloat), x
        assert isinstance(y, IntFloat), y
        
        self._req_xy = np.asarray([x, y])  # data point
        self._drag_text_xy = None
    
    def _settle(self):
        self._settled = True
        for artist in (self._arrow, self._point, self._box):
            artist._hover = True
    
    def _on_enter(self, event: tk.Event | None = None):
        def _set_active_width(artist: _BaseArtist):
            normal_w = artist.cget('width')
            active_w = artist.cget('activewidth') if artist._hover else normal_w
            artist.itemconfigure(width=active_w)
            return normal_w
        #> end of _set_active_width()
        
        # Restore the normal widths if they haven't been restored
        if self._line_width is not None:
            self._on_leave(event)
        
        self._box_edgewidth = _set_active_width(self._box)
        self._point_edgewidth = _set_active_width(self._point)
        self._arrow_edgewidth = _set_active_width(self._arrow)
        self._line_width = _set_active_width(self._line)
    
    def _on_leave(self, event: tk.Event | None = None):
        # Restore the normal widths
        self._box.itemconfigure(width=self._box_edgewidth)
        self._point.itemconfigure(width=self._point_edgewidth)
        self._arrow.itemconfigure(width=self._arrow_edgewidth)
        self._line.itemconfigure(width=self._line_width)
        self._box_edgewidth = self._point_edgewidth = self._arrow_edgewidth \
            = self._line_width = None
    
    def _on_leftpress(self, event: tk.Event):
        # Save the start position
        x1, y1, x2, y2 = self._text._req_bounds["xys"]
        self._motion_start = (event.x, event.y, x1, y1)
        
        # Lift artists to the top
        self._arrow.lift(self._tag)
        self._point.lift(self._tag)
        self._box.lift(self._tag)
        self._text.lift(self._tag)
    
    def _on_leftmotion(self, event: tk.Event):
        mouse_x0, mouse_y0, text_x0, text_y0 = self._motion_start
        dx, dy = (event.x - mouse_x0, event.y - mouse_y0)
        self._drag_text_xy = (text_x0 + dx), (text_y0 + dy)  # update position
        self.draw()
        self._on_enter()  # update box edgewidth if left
    
    def _on_rightpress(self, event: tk.Event):
        self.delete()


# =============================================================================
# ---- Figure Subwidget wrappers
# =============================================================================
def _with_draw_events(draw_func: Callable):
    @wraps(draw_func)
    def _wrapper(self, *args, **kwargs):
        tkwidget = self._tkwidget
        tkwidget.event_generate(DRAWSTARTED)
        try:
            draw_func(self, *args, **kwargs)
        finally:
            tkwidget.event_generate(DRAWENDED)
    
    return _wrapper


class _BaseWidgetWrapper:
    def __init__(self, tkwidget: tk.Misc):
        assert isinstance(tkwidget, tk.Misc), tkwidget
        assert isinstance(tkwidget.master, Figure), tkwidget.master
        tkwidget._figure = figure = tkwidget.master
        tkwidget._wrapper = self
        
        self._resize = defer(100)(self._resize)
        self._tkwidget: tk.Misc = tkwidget
        self._figure: Figure = figure
        self._resizing: bool = False
        self._draw_idle_id: str = 'after#'
        self._size: tuple[int, int] = (
            tkwidget.winfo_reqwidth(), tkwidget.winfo_reqheight()
        )
        self._req_facecolor: str | None = None
        
        if isinstance(tkwidget, tk.Canvas):
            tkwidget._zorder_tags: dict[_BaseArtist, str] = {}
            tkwidget.bind('<Configure>', self._on_configure, add=True)
    
    @property
    def _default_style(self) -> dict[str, Any]:
        return self._figure._default_style
    
    def _to_px(self, dimension: Dimension) -> float | tuple[float, ...]:
        return _to_px(self._tkwidget._root(), dimension)
    
    def _on_configure(self, event: tk.Event):
        def _resize():
            self._resize(event)
            self._resizing = False
        #> end of _resize()
        
        if not self._resizing:
            self._resizing = True
            self._tkwidget.itemconfigure('resizable=True', state='hidden')
            self._tkwidget.itemconfigure('movable=True', state='hidden')
        self._tkwidget.after_cancel(self._draw_idle_id)
        self._draw_idle_id = self._tkwidget.after_idle(_resize)
    
    def _resize(self, event: tk.Event):
        self._size = (event.width, event.height)
    
    def update_theme(self):
        if isinstance(self._tkwidget, tk.Canvas):
            for artist in self._tkwidget._zorder_tags:
                artist._stale = True
    
    def draw(self):
        raise NotImplementedError
    
    def draw_idle(self):
        self._tkwidget.after_cancel(self._draw_idle_id)
        self._draw_idle_id = self._tkwidget.after_idle(self.draw)
    
    def set_facecolor(self, color: str | None = None):
        assert isinstance(color, (str, type(None))), color
        
        if color is not None:
            self._req_facecolor = color


class _Suptitle(_BaseWidgetWrapper):
    _tag: str = 'suptitle'
    
    def __init__(self, master: Figure, *args, **kwargs):
        assert isinstance(master, Figure), master
        figure = master
        canvas = tk.Canvas(figure, *args, **kwargs)
        super().__init__(canvas)
        self._text: _Text = _Text(canvas, text='', tag=f'{self._tag}.text')
    
    def _resize(self, event: tk.Event):
        super()._resize(event)
        self.set_bounds()
        self._text.draw()
    
    @_with_draw_events
    def draw(self):
        # Update facecolor
        default_color = self._default_style["frame.cover.polygon"]["facecolor"]
        bg = default_color if self._req_facecolor is None else self._req_facecolor
        self._tkwidget.configure(background=bg)
        
        # Draw text
        self._text.draw()
        xys = self._text.bbox()
        if xys is not None:
            x1, y1, x2, y2 = xys
            self._tkwidget.configure(width=x2-x1+1, height=y2-y1+1)
            self._tkwidget.update_idletasks()  # triggers `self._on_configure`
    
    def get_facecolor(self) -> str:
        return self._tkwidget.cget('background')
    
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


class _Plot(_BaseWidgetWrapper):
    _tag: str = 'plot'
    
    def __init__(self, master: Figure, *args, **kwargs):
        assert isinstance(master, Figure), master
        figure = master
        canvas = tk.Canvas(figure, *args, **kwargs)
        super().__init__(canvas)
        del self._req_facecolor
        
        artists = {"lines": []}
        self._colors: dict[str, Cycle] = {"lines": self._create_color_cycle()}
        self._tartists: dict[str, list[_BaseArtist]] = deepcopy(artists)
        self._bartists: dict[str, list[_BaseArtist]] = deepcopy(artists)
        self._lartists: dict[str, list[_BaseArtist]] = deepcopy(artists)
        self._rartists: dict[str, list[_BaseArtist]] = deepcopy(artists)
        self._transforms: dict[str, _Transform2D] = {}
        self._dbounds: dict[str, NDArray[Float]] = {}
        self._cbounds: dict[str, NDArray[Int]] = {}
        
        self._title: _Text = _Text(canvas, text='', tag='title.text')
        self._taxis: _Axis = _Axis(self, side='t', tag='taxis')
        self._baxis: _Axis = _Axis(self, side='b', tag='baxis')
        self._laxis: _Axis = _Axis(self, side='l', tag='laxis')
        self._raxis: _Axis = _Axis(self, side='r', tag='raxis')
        self._tticks: _Ticks = _Ticks(self, side='t', tag='tticks')
        self._bticks: _Ticks = _Ticks(self, side='b', tag='bticks')
        self._lticks: _Ticks = _Ticks(self, side='l', tag='lticks')
        self._rticks: _Ticks = _Ticks(self, side='r', tag='rticks')
        self._frame: _Frame = _Frame(self, tag='frame')
        self._legend: _Legend = _Legend(self, tag='legend')
        
        self._veil: _Rectangle = _Rectangle(
            canvas, state='hidden', tag='veil.rectangle'
        )
        self._rubberband: _Rectangle = _Rectangle(
            canvas, state='hidden', tag='toolbar.rubberband.rectangle'
        )
        self._rubberband._bounds: tuple[Int, Int, Int, Int]
        
        canvas.bind('<Enter>', lambda e: figure._set_hovered_plot(self), add=True)
        canvas.bind(
            '<Leave>', lambda e: figure._unset_hovered_plot(self), add=True
        )
        
        self.set_btickslabels(True)
        self.set_ltickslabels(True)
        self.set_bticksticks(True)
        self.set_lticksticks(True)
    
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
    
    def update_theme(self):
        super().update_theme()
        self._frame.update_theme()
        self._legend.update_theme()
        for side in 'rblt':
            self._get_axis(side).update_theme()
            self._get_ticks(side).update_theme()
    
    @_with_draw_events
    def draw(self):
        if not self._figure._initialized:
            self._figure._initialize()
        
        canvas = self._tkwidget
        
        # Get data limits
        _dlimits = [
            np.asarray([
                getattr(a, lims) for a in sum(artists.values(), [])
                if a._req_state != 'hidden'
            ])
            for artists, lims in [
                (self._rartists, '_ylims'),
                (self._bartists, '_xlims'),
                (self._lartists, '_ylims'),
                (self._tartists, '_xlims')
            ]
        ]
        dbounds = np.array([(_DMIN, _DMAX)]*4, dtype=float)
        for i, lims in enumerate(_dlimits):
            if lims.size:
                dbounds[i] = [lims[:, 0].min(), lims[:, 1].max()]
        dbounds[0] = dbounds[0][::-1]  # flip y
        dbounds[2] = dbounds[2][::-1]  # flip y
        del _dlimits
        
        # Draw items and get empty space for frame
        w, h = self._size
        cx1, cy1, cx2, cy2 = (0, 0, w-1, h-1)
        
        ## Draw title
        self._title.set_bounds((cx1, cy1, cx2, cy2))
        self._title.draw()
        if bbox := self._title.bbox():
            cy1 = bbox[3] + 1
            if cy1 > cy2:
                cy1 = cy2 = round((cy1 + cy2) / 2.)
        cxys_axes = [cx1, cy1, cx2, cy2]
        
        ## Draw axes (top and bottom)
        ## These axes will be drawn again later after the actual frame dimensions
        ## are determined
        self._baxis.set_bounds((cx1, None, cx2, cy2))
        self._baxis.draw()
        self._taxis.set_bounds((cx1, cy1, cx2, None))
        self._taxis.draw()
        if bbox := self._baxis.bbox(): cy2 = bbox[1] - 1
        if bbox := self._taxis.bbox(): cy1 = bbox[3] + 1
        if cy1 > cy2: cy1 = cy2 = round((cy1 + cy2) / 2.)
        cxys_outer = [cx1, cy1, cx2, cy2]  # empty space for the ticks and frame
        
        ## Draw dummy ticks (top and bottom)
        cbounds = [[cy1, cy2], [cx1, cx2]] * 2  # shape: (4, 2)
        self._bticks._set_bounds_and_transform(
            (cx1, None, cx2, cy2), dbounds[1], cbounds[1], dummy=True
        )
        self._bticks._draw_dummy()
        self._tticks._set_bounds_and_transform(
            (cx1, cy1, cx2, None), dbounds[3], cbounds[3], dummy=True
        )
        self._tticks._draw_dummy()
        if bbox := self._bticks.bbox(dummy=True): cy2 = bbox[1] - 1
        if bbox := self._tticks.bbox(dummy=True): cy1 = bbox[3] + 1
        if cy1 > cy2: cy1 = cy2 = round((cy1 + cy2) / 2.)
        
        ## Draw legend
        ## This will be drawn again later after the user-defined artists are drawn
        self._legend._set_bounds((None, cy1, cx2, cy2))
        self._legend.draw()
        ## Update the bounds of the empty space for the frame (left and right)
        if bbox := self._legend.bbox():
            cx2 = bbox[0] - 1
        if cx1 > cx2: cx1 = cx2 = round((cx1 + cx2) / 2.)
        cxys_axes[2] = cx2
        
        ## Draw axes (top and bottom)
        ## These axes will be drawn again later after the actual frame dimensions
        ## are determined
        self._raxis.set_bounds((None, *cxys_axes[1:]))
        self._raxis.draw()
        self._laxis.set_bounds((*cxys_axes[:2], None, cxys_axes[3]))
        self._laxis.draw()
        if bbox := self._raxis.bbox(): cx2 = bbox[0] - 1
        if bbox := self._laxis.bbox(): cx1 = bbox[2] + 1
        if cx1 > cx2: cx1 = cx2 = round((cx1 + cx2) / 2.)
        if cy1 > cy2: cy1 = cy2 = round((cy1 + cy2) / 2.)
        cxys_outer[0], cxys_outer[2] = (cx1, cx2)
        
        ## Draw dummy ticks (left and right)
        self._rticks._set_bounds_and_transform(
            (None, cy1, cx2, cy2), dbounds[0], cbounds[0], dummy=True
        )
        self._rticks._draw_dummy()
        self._lticks._set_bounds_and_transform(
            (cx1, cy1, None, cy2), dbounds[2], cbounds[2], dummy=True
        )
        self._lticks._draw_dummy()
        if bbox := self._rticks.bbox(dummy=True): cx2 = bbox[0] - 1
        if bbox := self._lticks.bbox(dummy=True): cx1 = bbox[2] + 1
        if cx1 > cx2: cx1 = cx2 = round((cx1 + cx2) / 2.)
        cxys_inner = [cx1, cy1, cx2, cy2]  # empty space for the frame
        
        ## Draw ticks and update the data limits and canvas limits
        cbounds = np.asarray([[cy1, cy2], [cx1, cx2]] * 2)  # shape: (4, 2)
        tfs = {}
        tick_cdata = {}
        for i, (side, dat, cnv) in enumerate(zip('rblt', dbounds, cbounds)):
            growing_p = cxys_inner[i-2]
            xys = cxys_inner.copy()
            xys[i] = None  # growing bound
            xys[i-2] = cxys_outer[i-2]  # outer bound
            ticks = self._get_ticks(side)
            ticks._set_bounds_and_transform(tuple(xys), dat, cnv, growing_p)
            ticks.draw()
            tfs[side] = ticks._get_transform().copy()
            tick_cdata[side] = ticks._ticks_cdata
        
        # Draw frame
        self._frame.set_coords(*cxys_inner)
        self._frame._set_grid_cdata(**tick_cdata)
        self._frame.draw()
        
        # Draw the axes again after the actual frame dimensions are determined
        for i, side in enumerate('rblt'):
            xys = cxys_inner.copy()
            xys[i] = None
            xys[i-2] = cxys_axes[i-2]  # outer bound
            axis = self._get_axis(side)
            axis.set_bounds(tuple(xys))
            axis.draw()
        
        # Draw user defined artists
        cx1, cx2, cy1, cy2 = cx1+1, cx2-1, cy1+1, cy2-1
        if cx1 > cx2: cx1 = cx2 = round((cx1 + cx2) / 2.)
        if cy1 > cy2: cy1 = cy2 = round((cy1 + cy2) / 2.)
        cbounds = np.asarray([[cy1, cy2], [cx1, cx2]] * 2)  # shape: (4, 2)
        cbounds = dict(zip('rblt', cbounds))
        dbounds = dict(zip('rblt', dbounds))
        transforms = {}
        for x_side in 'bt':
            for y_side in 'rl':
                transforms[x_side + y_side] = _Transform2D(
                    tfs[x_side], tfs[y_side],
                    inp_xbounds=dbounds[x_side], inp_ybounds=dbounds[y_side],
                    out_xbounds=cbounds[x_side], out_ybounds=cbounds[y_side]
                )
        for artist in self.artists:
            sides = artist._x_side + artist._y_side
            artist.set_transform(transforms[sides])
            artist.draw()
        self._transforms.update(tfs)
        self._dbounds.update(dbounds)
        self._cbounds.update(cbounds)
        
        ## Draw legend again after the user-defined artists are drawn
        lines, labels = [], []
        for side in 'bt':
            for line in getattr(self, f'_{side}artists')["lines"]:
                if (label := line.get_label()) is not None:
                    labels.append(label)
                    lines.append(line)
        self._legend._set_contents(lines, labels)
        self._legend.draw()
        
        # Draw other artists
        self._veil.set_coords(0, 0, w-1, h-1)
        self._veil.draw()
        self._rubberband.draw()
        
        # Raise artists in order
        for tag in sorted(
                set(self._tkwidget._zorder_tags.values()),
                key=lambda t: float(t.split('zorder=', 1)[1])
        ):
            canvas.tag_raise(tag)
    
    def delete_all(self, draw: bool = False):
        for artist in self.artists:
            artist.delete()
        
        if draw:
            self.draw()
    
    def _create_color_cycle(self) -> Cycle:
        return Cycle(self._default_style["colors"])
    
    def set_facecolor(self, color: str | None = None):
        assert isinstance(color, (str, type(None))), color
        
        self._frame._cover.set_style(facecolor=color)
    
    def get_facecolor(self) -> str:
        return self._frame._cover.get_style()["facecolor"]
    
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
    
    def _get_axis(self, side: Literal['r', 'b', 'l', 't']) -> _Axis:
        assert side in ('r', 'b', 'l', 't'), side
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
            side: Literal['r', 'b', 'l', 't'],
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
    
    def _get_ticks(self, side: Literal['r', 'b', 'l', 't']) -> _Ticks:
        assert side in ('r', 'b', 'l', 't'), side
        return getattr(self, f'_{side}ticks')
    
    def get_tticks(self) -> _Ticks:
        return self._get_ticks('t')
    
    def get_bticks(self) -> _Ticks:
        return self._get_ticks('b')
    
    def get_lticks(self) -> _Ticks:
        return self._get_ticks('l')
    
    def get_rticks(self) -> _Ticks:
        return self._get_ticks('r')
    
    def _set_ticksticks(
            self,
            side: Literal['r', 'b', 'l', 't'],
            *args,
            **kwargs
    ):
        ticks = self._get_ticks(side)
        ticks.set_ticks(*args, **kwargs)
    
    def set_tticksticks(self, *args, **kwargs):
        return self._set_ticksticks('t', *args, **kwargs)
    
    def set_bticksticks(self, *args, **kwargs):
        return self._set_ticksticks('b', *args, **kwargs)
    
    def set_lticksticks(self, *args, **kwargs):
        return self._set_ticksticks('l', *args, **kwargs)
    
    def set_rticksticks(self, *args, **kwargs):
        return self._set_ticksticks('r', *args, **kwargs)
    
    def _set_tickslabels(
            self,
            side: Literal['r', 'b', 'l', 't'],
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
    
    def set_tscale(self, *args, **kwargs):
        self._tticks.set_scale(*args, **kwargs)
    
    def set_bscale(self, *args, **kwargs):
        self._bticks.set_scale(*args, **kwargs)
    
    def set_lscale(self, *args, **kwargs):
        self._lticks.set_scale(*args, **kwargs)
    
    def set_rscale(self, *args, **kwargs):
        self._rticks.set_scale(*args, **kwargs)
    
    def get_tscale(self) -> Literal['linear', 'log']:
        return self._tticks.get_scale()
    
    def get_bscale(self) -> Literal['linear', 'log']:
        return self._bticks.get_scale()
    
    def get_lscale(self) -> Literal['linear', 'log']:
        return self._lticks.get_scale()
    
    def get_rscale(self) -> Literal['linear', 'log']:
        return self._rticks.get_scale()
    
    def get_frame(self) -> _Frame:
        return self._frame
    
    def set_grid(self, *args, **kwargs):
        self._frame.set_grid(*args, **kwargs)
    
    def get_grid(self) -> dict[str, list[_Line]]:
        return self._frame.get_grid()
    
    def set_legend(self, enable: bool = True):
        self._legend.set_enabled(enable)
    
    def get_legend(self) -> _Legend:
        return self._legend
    
    def set_legend_size(self, *args, **kwargs):
        self._legend.set_size(*args, **kwargs)
    
    def get_legend_size(self) -> dict[str, Any]:
        return self._legend.get_size()
    
    def set_legend_facecolor(self, *args, **kwargs):
        self._legend.set_facecolor(*args, **kwargs)
    
    def get_legend_facecolor(self) -> str:
        return self._legend.get_facecolor()
    
    def set_legend_edge(self, *args, **kwargs):
        self._legend.set_edge(*args, **kwargs)
    
    def get_legend_edge(self) -> dict[str, Any]:
        return self._legend.get_edge()
    
    def set_legend_labels(self, *args, **kwargs):
        self._legend.set_labels(*args, **kwargs)
    
    def get_legend_labels(self) -> list[_Text]:
        return self._legend.get_labels()
    
    def line(
            self,
            b: ArrayLike | None = None,
            l: ArrayLike | None = None,
            t: ArrayLike | None = None,
            r: ArrayLike | None = None,
            color: str | None = None,
            width: Dimension | None = None,
            dash: str | tuple[Int] | None = None,
            smooth: bool | None = None,
            state: Literal['normal', 'hidden', 'disabled'] = 'normal',
            zorder: IntFloat | None = None,
            label: str | None = None,
            datalabel: bool = True,
            antialias: bool = True
    ) -> _Line:
        assert isinstance(label, (str, type(None))), label
        
        if not ((b is None) ^ (t is None)):
            raise ValueError('Either `b` or `t` must be a array-like value.')
        if not ((l is None) ^ (r is None)):
            raise ValueError('Either `l` or `r` must be a array-like value.')
        
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
            self._tkwidget,
            color=next(color_cycle),
            width=width,
            dash=dash,
            smooth=smooth,
            state=state,
            label=label,
            datalabel=True,
            antialias=antialias,
            antialias_bg=lambda: self._frame.get_facecolor(),
            hover=True,
            movable=True,
            resizable=True,
            x_side=x_side,
            y_side=y_side,
            tag='line'
        )
        line.set_data(x.ravel(), y.ravel())
        line.set_zorder(zorder)
        x_lines.append(line)
        y_lines.append(line)
        
        return line


class _Toolbar(_BaseWidgetWrapper):#TODO: tooptips
    _tag: str = 'toolbar'
    _cursors: dict[str, str] = {"pan": 'fleur', "zoom": 'crosshair'}
    
    def __init__(
            self,
            master: Figure,
            var_coord: tk.Variable,
            **kwargs
    ):
        assert isinstance(master, Figure), master
        figure = master
        frame = tk.Frame(figure, **kwargs)
        super().__init__(frame)
        
        self._keypress: dict[str, bool] = dict.fromkeys('axy', False)
        self._org_cursors: dict[_Plot, str] = {}
        self._bindings: dict[_Plot, list[str]] = {}
        self._active_plot: _Plot | None = None
        self._motion_start: tuple[Int, Int] | None = None
        self._anchor_start: tuple[Int, Int] | None = None
        self._pan_offsets: tuple[Int, Int] | None = None
        self._zoom_box: tuple[Int, Int, Int, Int] | None = None
        
        self._home_bt = ttk.Button(
            frame, text='Home', command=self._home_view, takefocus=False
        )
        self._home_bt.pack(side='left')
        
        self._prev_bt = ttk.Button(
            frame,
            text='Prev',
            command=self._prev_view,
            takefocus=False,
            state='disabled'
        )
        self._prev_bt.pack(side='left', padx=('6p', '0p'))
        
        self._next_bt = ttk.Button(
            frame,
            text='Next',
            command=self._next_view,
            takefocus=False,
            state='disabled'
        )
        self._next_bt.pack(side='left', padx=('3p', '0p'))
        
        self._var_mode: vrb.StringVar = vrb.StringVar(  # pan, zoom, or none
            frame, value='none'
        )
        self._var_mode.trace_add('write', self._on_mode_changed, weak=True)
        self._pan_bt = ttk.Checkbutton(
            frame,
            text='Pan',
            variable=self._var_mode,
            onvalue='pan',
            offvalue='none',
            bootstyle='toolbutton',
            takefocus=False
        )
        self._pan_bt.pack(side='left', padx=('6p', '0p'))
        
        self._zoom_bt = ttk.Checkbutton(
            frame,
            text='Zoom',
            variable=self._var_mode,
            onvalue='zoom',
            offvalue='none',
            bootstyle='toolbutton',
            takefocus=False
        )
        self._zoom_bt.pack(side='left', padx=('3p', '0p'))
        
        self._xyz_lb = tk.Label(frame, textvariable=var_coord)
        self._xyz_lb.pack(side='left', padx=('6p', '0p'))
        
        figure.bind('<KeyPress>', self._on_key, add=True)
        figure.bind('<KeyRelease>', self._on_key, add=True)
    
    def update_theme(self):
        super().update_theme()
        text_color = self._default_style["text"]["color"]
        self._xyz_lb.configure(fg=text_color)
    
    def draw(self):
        # Update facecolor
        default_color = self._default_style["frame.cover.polygon"]["facecolor"]
        bg = default_color if self._req_facecolor is None else self._req_facecolor
        self._tkwidget.configure(background=bg)
        self._xyz_lb.configure(background=bg)
    
    def get_facecolor(self) -> str:
        return self._tkwidget.cget('background')
    
    def _on_key(self, event: tk.Event):
        event_type = getattr(event.type, 'name', event.type)
        assert event_type in ('KeyPress', 'KeyRelease'), event
        
        # Update the keypress state
        if (key := event.keysym.lower()) in self._keypress:
            self._keypress[key] = event_type == 'KeyPress'
    
    def _on_mode_changed(self, *args):
        fig = self._figure
        org_cursors = self._org_cursors
        bindings = self._bindings
        mode = self._var_mode
        prev_mode, new_mode = mode.previous_value, mode.get()
        mode.value_changed(update=True)
        
        if prev_mode == new_mode:
            return
        
        if prev_mode != 'none':  # stop pan or zoom mode
            for plot in fig._plots.flat:
                canvas = plot._tkwidget
                canvas.configure(cursor=org_cursors[plot])
                
                press_id, motion_id, release_id = bindings[plot]
                unbind(canvas, MLEFTPRESS, press_id)
                unbind(canvas, MLEFTMOTION, motion_id)
                unbind(canvas, MLEFTRELEASE, release_id)
                
                plot._veil.set_state('hidden')
                plot._veil.draw()
            
            org_cursors.clear()
            bindings.clear()
        
        assert not org_cursors, org_cursors
        assert not bindings, bindings
        
        if new_mode == 'none':
            return
        
        # Start pan or zoom mode
        if new_mode == 'pan':
            press_cb, motion_cb, release_cb = (
                self._pan_on_leftpress,
                self._pan_on_leftmotion,
                self._pan_on_leftrelease
            )
        else:  # zoom mode
            press_cb, motion_cb, release_cb = (
                self._zoom_on_leftpress,
                self._zoom_on_leftmotion,
                self._zoom_on_leftrelease
            )
        new_cursor = self._cursors[new_mode]
        
        for plot in fig._plots.flat:
            canvas = plot._tkwidget
            org_cursors[plot] = canvas["cursor"]
            canvas.configure(cursor=new_cursor)
            
            bindings[plot] = [
                canvas.bind(MLEFTPRESS, press_cb, add=True),
                canvas.bind(MLEFTMOTION, motion_cb, add=True),
                canvas.bind(MLEFTRELEASE, release_cb, add=True)
            ]
            
            plot._veil.set_state('normal')
            plot._veil.draw()
    
    def _update_history(self):
        # Append current limits
        fig = self._figure
        history = fig._history
        
        # Drop the views after current step
        history.drop_future()
        
        # Create a ViewSet with current views
        viewset = _ViewSet(
            _PlotView(
                plot,
                {s: plot._get_ticks(s).get_limits() for s in 'rblt'}
            )
            for plot in fig._plots.flat
        )
        
        # Update the view history
        if not history or viewset != history[-1]:
            history.add(viewset)
    
    def _home_view(self):
        fig = self._figure
        history = fig._history
        autoscale = self._keypress["a"]
        
        # Initialize history if not exists
        if not history:
            self._update_history()
        
        # Updaet the view
        for plot in fig._plots.flat:
            if autoscale:
                limits, margins = ((None, None), (None, None))
                for side in 'rblt':
                    plot._get_ticks(side)._set_limits(*limits, margins)
            else:
                # Restore the first pair of limits
                for pview in history[0]:
                    if isinstance(pview, _PlotView) and pview.plot is plot:
                        break
                else:
                    raise ValueError(
                        f'`_Plot` ({plot}) was not found in the `_ViewSetHistory` '
                        f'({history._stack}).'
                    )
                
                for side, (limits, margins) in pview.view.items():
                    plot._get_ticks(side)._set_limits(*limits, margins)
        fig.draw()
        
        # Update history
        self._update_history()
    
    def _prev_view(self):
        fig = self._figure
        history = fig._history
        
        # Initialize history if not exists
        if not history:
            self._update_history()
        
        # Get the info of the previous view
        viewset = history.back()
        
        # Update the view
        for pview in viewset:
            plot, view = pview
            for side, (limits, margins) in view.items():
                plot._get_ticks(side)._set_limits(*limits, margins)
        fig.draw()
    
    def _next_view(self):
        fig = self._figure
        history = fig._history
        
        # Initialize history if not exists
        if not history:
            self._update_history()
        
        # Get the info of the next view
        viewset = history.forward()
        
        # Update the view
        for pview in viewset:
            plot, view = pview
            for side, (limits, margins) in view.items():
                plot._get_ticks(side)._set_limits(*limits, margins)
        fig.draw()
    
    def _pan_on_leftpress(self, event: tk.Event):
        plot = self._figure._hovered_plot
        canvas = plot._tkwidget
        if not (movable_oids := canvas.find_withtag('movable=True')):
            return
        
        # Get the anchor point (the lowest item in the stacking list)
        x1, y1, _, _ = canvas.bbox(movable_oids[0])
        
        # Initialize the states
        self._update_history()
        self._active_plot = plot
        self._motion_start = (event.x, event.y)
        self._anchor_start = (x1, y1)
    
    def _pan_on_leftmotion(self, event: tk.Event):
        if not (plot := self._active_plot):
            return
        
        # Determine current pan mode
        along_x = self._keypress["x"]
        along_y = self._keypress["y"]
        if not along_x and not along_y:
            along_x = along_y = True
        
        # Calculate the amounts of offsets
        start_x, start_y = self._motion_start
        anchor_x, anchor_y = self._anchor_start
        dx = event.x - start_x if along_x else 0
        dy = event.y - start_y if along_y else 0
        x1, y1 = (anchor_x + dx, anchor_y + dy)
        
        # Update movable artists
        plot._tkwidget.moveto('movable=True', x1, y1)
        
        # Update the states
        self._pan_offsets = (dx, dy)
    
    def _pan_on_leftrelease(self, event: tk.Event):
        if not (plot := self._active_plot) or not (offsets := self._pan_offsets):
            self._clear_pan_zoom_states()
            return
        
        # Calculate the amounts of mouse offsets
        dx, dy = offsets
        cbounds = plot._cbounds
        cx12 = np.sort(cbounds["b"])
        cy12 = np.sort(cbounds["l"])
        cx1, cx2 = cx12 - dx
        cy1, cy2 = cy12 - dy
        
        # Update view and clear states
        self._pan_zoom_on_leftrelease((cx1, cy1, cx2, cy2))
        self._clear_pan_zoom_states()
    
    def _zoom_on_leftpress(self, event: tk.Event):
        self._update_history()
        plot = self._figure._hovered_plot
        
        # Calculate rubberband coordinates
        cbounds = plot._cbounds
        xlimits = np.sort(cbounds["b"])
        ylimits = np.sort(cbounds["l"])
        x, y = np.clip(event.x, *xlimits), np.clip(event.y, *ylimits)
        
        # Update rubberband
        rubberband = plot._rubberband
        rubberband.set_coords(x, y, x, y)
        rubberband.set_state('normal')
        rubberband.draw()
        
        # Initilize the states
        self._active_plot = plot
        self._motion_start = (x, y)
    
    def _zoom_on_leftmotion(self, event: tk.Event):
        plot = self._active_plot
        x1, y1 = self._motion_start
        
        # Determine current pan mode
        along_x = self._keypress["x"]
        along_y = self._keypress["y"]
        if not along_x and not along_y:
            along_x = along_y = True
        
        # Calculate rubberband coordinates
        cbounds = plot._cbounds
        xlimits = np.sort(cbounds["b"])
        ylimits = np.sort(cbounds["l"])
        if along_x:
            x = np.clip(event.x, *xlimits)
            x1, x2 = np.sort([x1, x])
        else:
            x1, x2 = xlimits
        if along_y:
            y = np.clip(event.y, *ylimits)
            y1, y2 = np.sort([y1, y])
        else:
            y1, y2 = ylimits
        
        # Update rubberband
        rubberband = plot._rubberband
        rubberband.set_coords(x1, y1, x2, y2)
        rubberband.draw()
        
        # Update the states
        self._zoom_box = (x1, y1, x2, y2)
    
    def _zoom_on_leftrelease(self, event: tk.Event):
        if not (plot := self._active_plot):
            self._clear_pan_zoom_states()
            return
        
        # Hide rubberband
        plot = self._active_plot
        rubberband = plot._rubberband
        rubberband.set_state('hidden')
        rubberband.draw()
        
        if not (box := self._zoom_box):
            return
        
        # Update view and clear states
        self._pan_zoom_on_leftrelease(box)
        self._clear_pan_zoom_states()
    
    def _pan_zoom_on_leftrelease(self, cbounds: tuple[Int, Int, Int, Int]):
        plot = self._active_plot
        cx1, cy1, cx2, cy2 = cbounds
        
        for side, tf in plot._transforms.items():
            # Calculate canvas limits from cbounds and margins
            ticks = plot._get_ticks(side)
            _, [mrg1, mrg2] = ticks.get_limits()
            c1, c2 = [cx1, cx2] if side in 'bt' else [cy1, cy2]
            c12 = np.array([c1+mrg1, c2-mrg2])
            if c12[0] > c12[1]:
                c12 = np.array([c1, c2])
            
            # Transform into data limits
            with np.errstate(all='raise'):
                try:
                    itf = tf.get_inverse()
                    d12 = itf(c12)
                    d12.sort()
                except FloatingPointError:
                    continue
            
            # Update data limits
            if np.isfinite(d12).all():
                ticks.set_limits(*d12)
        plot.draw()
        
        # Update history
        self._update_history()
    
    def _clear_pan_zoom_states(self):
        self._active_plot = None
        self._motion_start = None
        self._anchor_start = None
        self._pan_offsets = None
        self._zoom_box = None


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
        
        self._initialized: bool = False
        self._req_size: tuple[Int | None, Int | None] = (None, None)
        self._req_facecolor: str | None = None
        self._plots: NDArray[_Plot] = np.array([], dtype=object)
        self._hovered_plot: _Plot | None = None
        self._draw_idle_id: str = 'after#'
        self._history: _ViewSetHistory = _ViewSetHistory(self)
        self._var_coord: vrb.StringVar = vrb.StringVar(self, value='()')
        self._default_style: dict[str, Any]
        
        self.grid_propagate(False)  # allow `self` to be resized
        self.update_theme()
        self.set_size(width=width, height=height)
        
        self._suptitle: _Suptitle
        if suptitle:
            self.set_suptitle(text=suptitle)
        
        self._toolbar: _Toolbar
        if toolbar:
            self.set_toolbar(True)
        
        self.bind('<Destroy>', self._on_destroy, add=True)
        self.bind('<<ThemeChanged>>', self._on_theme_changed, add=True)
        self.focus_set()
        
        self.after_idle(self._initialize)
    
    def _to_px(self, dimension: Dimension) -> float | tuple[float, ...]:
        return _to_px(self._root(), dimension)
    
    def _initialize(self):
        if hasattr(self, '_suptitle'):
            self._suptitle.draw()
        self._initialized = True
    
    def _set_hovered_plot(self, plot: _Plot | None = None):
        assert isinstance(plot, (_Plot, type(None))), plot
        self._hovered_plot = plot
    
    def _unset_hovered_plot(self, plot: _Plot):
        assert isinstance(plot, _Plot), plot
        if self._hovered_plot == plot:
            self._hovered_plot = None
    
    def _on_destroy(self, event: tk.Event | None = None):
        _cleanup_tk_attributes(self)
    
    def _on_theme_changed(self, event: tk.Event):
        self.update_theme()
        if self._initialized:
            self.draw_idle()
    
    def update_theme(self):
        self._default_style = deepcopy(STYLES[self._root().style.theme.type])
        
        # Update undock button
        id_ = str(id(self))
        if not (udbt_style := self.undock_button["style"]).startswith(id_):
            udbt_style = f'{id(self)}.{udbt_style}'
        default = self._default_style["frame.cover.polygon"]["facecolor"]
        udbt_bg = default if self._req_facecolor is None else self._req_facecolor
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
        
        # Update suptitle
        if hasattr(self, '_suptitle'):
            self._suptitle.update_theme()
        
        # Update plots
        for plot in self._plots.flat:
            plot.update_theme()
        
        # Update toolbar
        if hasattr(self, '_toolbar'):
            self._toolbar.update_theme()
    
    def draw(self):
        # Initialization
        if not self._initialized:
            self._initialize()
        
        # Update facecolor
        default_color = self._default_style["frame.cover.polygon"]["facecolor"]
        bg = default_color if self._req_facecolor is None else self._req_facecolor
        self.configure(background=bg)
        
        # Draw suptitle
        if hasattr(self, '_suptitle'):
            self._suptitle.draw()
        
        # Draw plots
        for plot in self._plots.flat:
            plot.draw()
        
        # Draw toolbar
        if hasattr(self, '_toolbar'):
            self._toolbar.draw()
    
    def draw_idle(self):
        self.after_cancel(self._draw_idle_id)
        self._draw_idle_id = self.after_idle(self.draw)
    
    def set_size(
            self,
            width: Dimension | None = None,
            height: Dimension | None = None
    ):
        if width is not None:
            self._req_size[0] = width
        if height is not None:
            self._req_size[1] = height
        
        default_size = self._default_style["size"]
        new_size = tuple(
            round(self._to_px(request if request is not None else default))
            for request, default in zip(self._req_size, default_size)
        )
        
        old_size = self.get_size()
        if new_size != old_size:
            self.configure(width=new_size[0], height=new_size[1])
    
    def get_size(self) -> tuple[Int, Int]:
        return (self.winfo_reqwidth(), self.winfo_reqheight())
    
    def set_facecolor(self, color: str | None = None):
        assert isinstance(color, (str, type(None))), color
        
        if color is not None:
            self._req_facecolor = color
            
            if hasattr(self, '_suptitle'):
                self._suptitle.set_facecolor(color)
            
            for plot in self._plots.flat:
                plot.set_facecolor(color)
            
            if hasattr(self, '_toolbar'):
                self._toolbar.set_facecolor(color)
    
    def get_facecolor(self) -> str:
        return self.cget('background')
    
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
                self._suptitle._tkwidget.grid(row=0, column=0, sticky='we')
                if self._plots.size:
                    n_rows, n_cols = self._plots.shape
                    self._suptitle._tkwidget.grid(columnspan=n_cols)
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
            padx: Dimension | tuple[Dimension, Dimension] = ('0p', '0p'),
            pady: Dimension | tuple[Dimension, Dimension] = ('0p', '0p')
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
        for r, row in enumerate(self._plots):
            for c, plot in enumerate(row):
                plot._tkwidget.grid_forget()
                self.grid_columnconfigure(c, weight=0)
            self.grid_rowconfigure(r, weight=0)
        self._history.clear()
        
        # Update suptitle's position
        if hasattr(self, '_suptitle'):
            self._suptitle._tkwidget.grid(columnspan=n_cols)
        
        # Update toolbar's position
        if hasattr(self, '_toolbar'):
            self._toolbar._tkwidget.grid(row=n_rows+1, columnspan=n_cols)
        
        # Create plots
        self._plots: NDArray[_Plot] = np.array([
            [ _Plot(self) for c in range(n_cols) ] for r in range(n_rows)
        ])
        grid_kw = {"sticky": 'nesw', "padx": padx, "pady": pady}
        for r, row in enumerate(self._plots, 1):  # plots start from 2nd row
            for c, plot in enumerate(row):
                plot._tkwidget.grid(row=r, column=c, **grid_kw)
                plot._tkwidget.configure(width=0, height=0)  # this makes all space
                 # become extra space which will be distributed to each row and
                 # column with each weight specified in `grid_rowconfigure` and
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
            self._toolbar = _Toolbar(self, var_coord=self._var_coord)
            kw = {"column": 0, "sticky": 'we', "padx": 9}
            if self._plots.size:
                n_rows, n_cols = self._plots.shape
                self._toolbar._tkwidget.grid(row=n_rows+1, columnspan=n_cols, **kw)
            else:
                self._toolbar._tkwidget.grid(row=1, **kw)
                 # the toolbar will be `grid` again when `set_plots` is called
                self.grid_columnconfigure(0, weight=1)
        elif not enable and hasattr(self, '_toolbar'):
            self._toolbar._tkwidget.destroy()
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
    
    root = Window(title='Figure Demo', themename='morph')
    
    x = np.arange(0, 10, 1/48000, dtype=float)
    y = np.sin(2*np.pi*1*x)
    #x = np.array([3, 6, 6, 3, 3], dtype=float)
    #y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5], dtype=float)
    #y = 10**x
    
    fig = Figure(root, toolbar=True)
    fig.pack(fill='both', expand=True)
    
    suptitle = fig.set_suptitle('<Suptitle>')
    plts = fig.set_plots(1, 1)
    plts = np.atleast_2d(plts)
    for r in range(plts.shape[0]):
        for c in range(plts.shape[1]):
            plt = plts[r, c]
            for i in range(10):
                plt.line(x, y*i, label=f'line {i} in plot {(r, c)}')
            plt.set_title('<Title>')
            plt.set_tlabel('<top-label>')
            plt.set_blabel('<bottom-label>')
            plt.set_llabel('<left-label>')
            plt.set_rlabel('<right-label>')
            plt.set_ttickslabels(True)
            plt.set_rtickslabels(True)
            #plt.set_lscale('log')
            #plt.set_llimits(10, 150)
            plt.set_llimits(-1.7, 1.7)
            plt.set_legend(True)
    
    fig.after(3000, lambda: root.style.theme_use('cyborg'))
    
    root.mainloop()

