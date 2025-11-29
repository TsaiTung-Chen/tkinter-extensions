"""
Created on Sun Dec 11 19:18:31 2022
@author: tungchentsai
"""
from functools import wraps
from types import NoneType
from typing import TYPE_CHECKING, Any, Literal, overload
from collections.abc import Callable, Sequence
from PIL import Image, ImageColor
from PIL.ImageTk import PhotoImage
import tkinter as tk
import tkinter.font as tk_font

tk_get_default_root: Callable[[Any], tk.Tk] = (
    tk._get_default_root  # pyright: ignore [reportAttributeAccessIssue]
)

import numpy as np
import ttkbootstrap as tb
from ttkbootstrap import colorutils

from tkinter_extensions._constants import (
    DEFAULT_PPD, BUILTIN_WIDGETS, MODIFIERS, MODIFIER_MASKS,
    EventCommand,
    Int, IntFloat, _IntFloat, ScreenUnits, _ScreenUnits
)
# =============================================================================
# MARK: Typing
# =============================================================================
if TYPE_CHECKING:
    def mixin_base[T](cls: type[T]) -> type[T]: ...
else:
    def mixin_base(cls: type) -> type[object]:
        return object


class DropObject(type):
    def __new__(
        mcls, clsname: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ):
        bases = tuple( base for base in bases if base is not object )
        
        return super().__new__(mcls, clsname, bases, namespace)


# =============================================================================
# MARK: Others
# =============================================================================
@overload
def scale_size(widget: tk.Misc | None, size: IntFloat, /) -> int: ...
@overload
def scale_size(
    widget: tk.Misc | None,
    size1: IntFloat,
    size2: IntFloat,
    /,
    *sizes: IntFloat
) -> tuple[int, ...]: ...
@overload
def scale_size(
    widget: tk.Misc | None, sizes: Sequence[IntFloat], /
) -> tuple[int, ...]: ...
def scale_size(widget, *sizes):
    """
    Scale the size based on the scaling factor of tkinter. 
    This is used most frequently to adjust the assets for image-based widget
    layouts and font sizes.
    
    Parameters:
    
        widget (Widget):
            The widget object.
        
        size (Union[int, List, Tuple]):
            A single integer or an iterable of integers.
    
    Returns:
    
        Union[int, List]:
            An integer or list of integers representing the new size.
    """
    assert len(sizes) >= 1, sizes

    if widget is None:
        widget = tk_get_default_root('scale_size')
    ppd: float = widget.winfo_fpixels('1p')  # current pixels per point
    factor: float = ppd / DEFAULT_PPD # current UI scaling factor
    
    if len(sizes) == 1:
        sizes_ = sizes[0]
        if isinstance(sizes_, _IntFloat):
            return round(sizes_ * factor)
        elif isinstance(sizes_, Sequence):
            return tuple( round(s * factor) for s in sizes_ )
        raise TypeError(
            'The elements of `sizes` must be of type `int`, `tuple`, or `list` '
            f'but got {type(sizes_)}.'
        )
    
    return tuple( round(size * factor) for size in sizes )


@overload
def to_pixels(root: tk.Misc, dimension: ScreenUnits) -> int: ...
@overload
def to_pixels(root: tk.Misc, dimension: None) -> None: ...
@overload
def to_pixels(
    root: tk.Misc, dimension: tuple[ScreenUnits, ...]
) -> tuple[int, ...]: ...
@overload
def to_pixels(
    root: tk.Misc,
    dimension: tuple[ScreenUnits | None, ...]
) -> tuple[int | None, ...]: ...
def to_pixels(root, dimension):
    assert isinstance(root, tk.Misc), root
    assert isinstance(dimension, (_ScreenUnits, tuple, NoneType)), dimension
    
    _winfo_fpixels = root.winfo_fpixels
    _to_pixels: Callable[[ScreenUnits], int] = (
        lambda d: round(_winfo_fpixels(str(d)))
    )
    
    if dimension is None:
        return None
    elif isinstance(dimension, _ScreenUnits):
        return _to_pixels(dimension)
    return tuple( None if d is None else _to_pixels(d) for d in dimension )


def quit_if_all_closed(window: tk.Misc) -> Callable[[tk.Event | None], None]:
    def _quit_if_all_closed(event: tk.Event | None = None) -> None:
        root: tk.Tk = window._root()  # pyright: ignore [reportAttributeAccessIssue]
        if len(root.children) > 1:
            window.destroy()
        else:
            root.quit()
            root.destroy()
    #> end of _quit_if_all_closed()
    return _quit_if_all_closed


def get_center_position(widget: tk.Misc) -> tuple[int, int]:
    widget.update_idletasks()
    
    if (width := widget.winfo_width()) == 1:
        width = widget.winfo_reqwidth()
    if (height := widget.winfo_height()) == 1:
        height = widget.winfo_reqheight()
    x_root, y_root = widget.winfo_rootx(), widget.winfo_rooty()
    
    return (x_root + width//2, y_root + height//2)


def center_window(to_center: tk.Misc, center_of: tk.Misc) -> None:
    """
    Center the `to_center` window on the `center_of` window.
    """
    x_center, y_center = get_center_position(center_of)

    if (width := to_center.winfo_width()) == 1:
        width = to_center.winfo_reqwidth()
    if (height := to_center.winfo_height()) == 1:
        height = to_center.winfo_reqheight()
    x, y = (x_center - width//2, y_center - height//2)
    
    tk.Wm.wm_geometry(to_center, f'+{x}+{y}')  # pyright: ignore [reportCallIssue, reportArgumentType]
     # ignore type checking because `tk.Frame` can be undocked to become a
     # toplevel, who also has this command


def defer(ms: int = 1000) -> Callable[[Callable], Callable]:
    """
    Call the decorated function `ms` after the last call on the wrapper.
    Only the last call on the wrapper will exactly execute the decorated 
    function. This means that all the others will be cancelled
    """
    def _decorator[**P, R](func: Callable[P, R]) -> Callable[P, None]:  # decorator
        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
            nonlocal last_id
            root = tk_get_default_root(func.__name__)
            root.after_cancel(last_id)
            last_id = root.after(ms, lambda: func(*args, **kwargs))
        #> end of _wrapped()
        
        last_id: str = '-1'
        return _wrapper
    #> end of _wrapper()
    
    return _decorator


def unbind(widget: tk.Misc, sequence: str, funcid: str | None = None) -> None:
    """
    The built-in function does not unbind the expected specified function
    when `funcid` is provided. So we make this workaround function 
    to replace the use of the original `widget.unbind` method
    """
    if not funcid:
        return widget.unbind(sequence, funcid)
    
    cmds = widget.bind(sequence).split('\n')  # [cmd, '', cmd, '', ...]
    widget.unbind(sequence)  # remove bindings but does not `widget.deletecommand`
    
    # Rebind the other functions
    other_cmds = [ c for c in cmds[::2] if funcid not in c ]  # [cmd, cmd, ...]
    if other_cmds:
        widget.bind(sequence, '\n'.join([ c + '\n' for c in other_cmds ]))


def bind_recursively(
    widget: tk.Misc,
    seqs: str | Sequence[str],
    funcs: EventCommand | Sequence[EventCommand],
    add: bool | Literal['', '+'] | None = None,
    *,
    key: str,
    skip_toplevel: bool = False
) -> None:
    if (
        skip_toplevel
        and widget == widget.winfo_toplevel()
        and widget != widget._root()  # pyright: ignore [reportAttributeAccessIssue]
    ):
        return  # skip toplevel excluding root
    
    if isinstance(seqs, str):
        seqs = [seqs]
    if callable(funcs):
        funcs = [funcs]
    assert len(seqs) == len(funcs), (seqs, funcs)
    
    # Propagate
    for child in widget.winfo_children():
        if skip_toplevel and (child == child.winfo_toplevel()):
            continue  # skip toplevel and its descendants
        bind_recursively(
            child, seqs, funcs, add, key=key, skip_toplevel=skip_toplevel
        )
    
    #TODO: improve readability
    recursively_bound: dict[str, dict[str, str]] = getattr(
        widget, '_recursively_bound', {}
    )
    setattr(widget, '_recursively_bound', recursively_bound)
    for seq, func in zip(seqs, funcs):
        assert seq.startswith('<') and seq.endswith('>'), seq
        subbound: dict[str, str] = recursively_bound.setdefault(seq, {})
        if key in subbound:
            continue  # skip if the key already exists
        subbound[key] = widget.bind(seq, func, add)  # func id


def unbind_recursively(
    widget: tk.Misc,
    seqs: str | list[str] | None = None,
    *,
    key: str,
    skip_toplevel: bool = False
) -> None:
    if (
        skip_toplevel
        and widget == widget.winfo_toplevel()
        and widget != widget._root()  # pyright: ignore [reportAttributeAccessIssue]
    ):
        return  # skip toplevel except for root
    
    if isinstance(seqs, str):
        seqs = [seqs]
    
    # Propagate
    for child in widget.winfo_children():
        if skip_toplevel and (child == child.winfo_toplevel()):
            continue  # skip toplevel and its descendants
        unbind_recursively(
            child, seqs, key=key, skip_toplevel=skip_toplevel
        )
    
    #TODO: improve readability
    recursively_bound: dict[str, dict[str, str]] = getattr(
        widget, '_recursively_bound', {}
    )
    if not recursively_bound:
        return
    
    setattr(widget, '_recursively_bound', recursively_bound)
    for _seq, subbound in list(recursively_bound.items()):
        for _key, func_id in list(subbound.items()):
            if _key != key:
                continue
            if seqs is None:
                unbind(widget, _seq, func_id)
            elif _seq in seqs:
                unbind(widget, _seq, func_id)
            del subbound[_key]
        if not subbound:
            del recursively_bound[_seq]
    if not recursively_bound:
        delattr(widget, '_recursively_bound')


#TODO: change the order of `redirected` and `source` to be more intuitive
#TODO: rename: `redirect_layout_managements`
def redirect_layout_managers(
    redirected: tk.Misc,  #TODO: rename this with `target`
    source: tk.Misc,
    orig_prefix: str = 'content_'  #TODO: rename this with `prefix_innateness`
) -> tuple[str, ...]:
    """
    Redirect layout manager to the `source`'s layout manager
    """
    layout_methods = (
        'pack_configure', 'pack',
        'pack_forget',
        'pack_info',
        'pack_propagate',
        'pack_slaves',
        'place_configure', 'place',
        'place_forget',
        'place_info',
        'place_slaves',
        'grid_configure', 'grid',
        'grid_forget',
        'grid_remove',
        'grid_info',
        'grid_propagate',
        'grid_slaves',
        'grid_bbox',
        'grid_location',
        'grid_size',
        'grid_anchor'
    )
    
    for name in layout_methods:
        setattr(redirected, orig_prefix+name, getattr(redirected, name))
        setattr(redirected, name, getattr(source, name))
    
    return layout_methods


def get_modifiers(state: int, platform_specific: bool = True) -> set:
    modifiers = set()
    _modifiers = MODIFIERS if platform_specific else MODIFIER_MASKS
    for mod in _modifiers:
        if state & MODIFIER_MASKS[mod]:
            modifiers.add(mod)
    
    return modifiers


def create_font(
    new_name: str | None = None,
    base_font: str | tk_font.Font = 'TkDefaultFont',
    family: str | None = None,
    size: int | None = None,
    weight: Literal['normal', 'bold'] | None = None,
    slant: Literal['roman', 'italic'] | None = None,
    underline: bool | None = None,
    overstrike: bool | None = None
) -> tk_font.Font:
    assert isinstance(base_font, (str, tk_font.Font)), base_font
    
    if isinstance(base_font, str):
        base_font = tk_font.nametofont(base_font)
    assert isinstance(base_font, tk_font.Font), type(base_font)
    new_font = base_font.copy().actual()
    
    if family is not None:
        new_font["family"] = family
    if size is not None:
        _size, size = size, int(size)
        if isinstance(_size, str) and (
            _size.startswith('+') or _size.startswith('-')
        ):
            size += new_font["size"]
        new_font["size"] = size
    if weight is not None:
        new_font["weight"] = weight
    if slant is not None:
        new_font["slant"] = slant
    if underline is not None:
        new_font["underline"] = underline
    if overstrike is not None:
        new_font["overstrike"] = overstrike
    
    return tk_font.Font(name=new_name, **new_font)


def get_font(class_name: str, default: str = 'TkDefaultFont') -> tk_font.Font:
    style = tb.Style.get_instance()
    assert style is not None, 'Style instance is not available'
    font_name = style.lookup(class_name, 'font',  default=default)
    return tk_font.nametofont(font_name)


def create_font_style(
    class_name: str,
    prefix: str,
    bootstyle: str | tuple[str, ...] | None = '',
    orient: Literal['horizontal', 'vertical'] | None = None,
    apply: bool = True,
    **kwargs
) -> tuple[str, tk_font.Font]:
    subs = ['', *class_name.split('.')]
    for i, widget_name in enumerate(subs[::-1]):
        if widget_name in BUILTIN_WIDGETS:
            break
    else:
        raise ValueError(
            f'Cannot find the widget name in {BUILTIN_WIDGETS} for class_name '
            f'{class_name}.'
        )
    
    if widget_name != 'Treeview':
        assert widget_name.startswith('T')
        widget_name = widget_name[1:]
    minor = False if i == 0 else True
    
    dummy_widget = getattr(tb, widget_name)(bootstyle=bootstyle)
    if orient:
        dummy_widget.configure(orient=orient)
    style_name = f'{prefix}.{dummy_widget["style"]}'
    dummy_widget.destroy()
    
    style = tb.Style.get_instance()
    assert style is not None, 'Style instance is not available'
    base_font = get_font(class_name)
    
    if minor:
        minor_style_name = style_name + f'.{subs[-1]}'
        new_font = create_font(minor_style_name, base_font, **kwargs)
        if apply:
            style.configure(style_name)
            style.configure(minor_style_name, font=new_font)
    else:
        new_font = create_font(style_name, base_font, **kwargs)
        if apply:
            style.configure(style_name, font=new_font)
    
    return style_name, new_font


def is_color(color: str) -> bool:
    try:
        ImageColor.getrgb(color)
    except ValueError:
        return False
    return True


def contrast_color(color: str) -> str:
    return colorutils.contrast_color(ImageColor.getrgb(color), model='rgb')


@overload
def recolor_black(
    image: Image.Image,
    new_color: tuple[Int, Int, Int],   # RGB
    photoimage: Literal[False] = False,
    master: tk.Misc | None = None
) -> Image.Image: ...
@overload
def recolor_black(
    image: Image.Image,
    new_color: tuple[Int, Int, Int],   # RGB
    photoimage: Literal[True],
    master: tk.Misc | None = None
) -> PhotoImage: ...
def recolor_black(
    image,
    new_color,  # RGB
    photoimage = False,
    master = None
):
    data = np.array(image.convert('RGBA'), copy=True)
    is_rgb = slice(3)  # channel 0, 1, 2
    is_black = (data[..., is_rgb] == 0).all(axis=-1)  # 2D pixels (rgb all 0)
    data[is_black, is_rgb] = new_color  # replace black with `new_color`
    img_converted = Image.fromarray(data, mode='RGBA')
    
    if photoimage:
        return PhotoImage(img_converted, master=master)
    return img_converted


@overload
def create_image_pair(
    image: Image.Image,
    widget: tk.Misc,
    photoimage: Literal[False] = False,
    master: tk.Misc | None = None
) -> tuple[Image.Image, Image.Image]: ...
@overload
def create_image_pair(
    image: Image.Image,
    widget: tk.Misc,
    photoimage: Literal[True],
    master: tk.Misc | None = None
) ->  tuple[PhotoImage, PhotoImage]: ...
def create_image_pair(
    image,
    widget,
    photoimage = False,
    master = None
):
    _convert_bitdepth = lambda x: round((255./65535.) * x)
    
    style = tb.Style()
    fg = style.lookup(widget["style"], 'foreground')
    fg = tuple(map(_convert_bitdepth, widget.winfo_rgb(fg)))
    fg_pressed = style.lookup(widget["style"], 'foreground', ['pressed'])
    fg_pressed = tuple(map(_convert_bitdepth, widget.winfo_rgb(fg_pressed)))
    img_normal = recolor_black(image, fg, photoimage=photoimage, master=master)
    img_pressed = recolor_black(
        image, fg_pressed, photoimage=photoimage, master=master
    )
    
    return img_normal, img_pressed


def create_color_image(
    color: str,
    size: Int | tuple[Int, Int],
    padding: int | tuple | list = (0, 0),
    autoscale: bool = True,
    photoimage: bool = False,
    master: tk.Misc | None = None
) -> Image.Image | PhotoImage:
    """
    Create an image of size `size` and color `color` with external padding 
    `padding`. The padded margin has the same rgb color but with zero alpha 
    value
    """
    color_tuple = ImageColor.getrgb(color)
    assert len(color_tuple) in (3, 4), color_tuple
    assert isinstance(size, (int, tuple, list)), size
    assert isinstance(padding, (int, tuple, list)), padding
    
    if isinstance(size, int):
        size = (size, size)
    if isinstance(padding, int):
        padding = [padding, padding]
    if len(color_tuple) == 3:
        color_tuple += (255,)  # RGB => RGBA
    
    if autoscale:
        master = master or tk_get_default_root('create_color_image')
        assert len(size_ := scale_size(master, size)) == 2, size_
        size = size_
    
    sizex, sizey = size
    padx, pady = padding
    padded_size = tuple( s + 2*p for s, p in zip(size, padding) )
    
    # Alpha data
    alpha = np.zeros(padded_size, dtype='uint8')
    alpha[padx:padx + sizex, pady:pady + sizey] = color_tuple[-1]
    
    # RGB data
    data = [ np.full(padded_size, c, dtype='uint8') for c in color_tuple[:3] ]
    
    # RGBA data
    data.append(alpha)
    data = np.stack(data, axis=-1)
    
    # Convert to RGBA image (row => y, col => x)
    image = Image.fromarray(data.swapaxes(0, 1), mode='RGBA')
    
    if photoimage:
        return PhotoImage(image, master=master)
    return image


@overload
def modify_hsl(
    color: tuple[Int, Int, Int] | str,
    func: Callable[[Int, Int, Int], tuple[Int, Int, Int]],
    *,
    inmodel: Literal['hsl', 'rgb', 'hex'] = 'hex',
    outmodel: Literal['hex'] = 'hex',
) -> str: ...
@overload
def modify_hsl(
    color: tuple[Int, Int, Int] | str,
    func: Callable[[Int, Int, Int], tuple[Int, Int, Int]],
    *,
    inmodel: Literal['hsl', 'rgb', 'hex'] = 'hex',
    outmodel: Literal['rgb']
) -> tuple[Int, Int, Int] | tuple[Int, Int, Int, Int]: ...
@overload
def modify_hsl(
    color: tuple[Int, Int, Int] | str,
    func: Callable[[Int, Int, Int], tuple[Int, Int, Int]],
    *,
    inmodel: Literal['hsl', 'rgb', 'hex'] = 'hex',
    outmodel: Literal['hsl']
) -> tuple[Int, Int, Int]: ...
def modify_hsl(color, func, *, inmodel='hex', outmodel='hex') -> Any:
    h, s, l = colorutils.color_to_hsl(color, model=inmodel)
    h, s, l = func(h, s, l)
    
    if outmodel == 'hsl':
        return (h, s, l)
    elif outmodel == 'rgb':
        rgb = colorutils.color_to_rgb((h, s, l), model='hsl')
        assert rgb is not None, 'Invalid RGB color'
        return rgb
    elif outmodel == 'hex':
        return colorutils.color_to_hex((h, s, l), model='hsl')
    else:
        raise ValueError(
            f"Invalid output model {outmodel}. Must be one of 'hsl', 'rgb', or "
            "'hex.'"
        )

