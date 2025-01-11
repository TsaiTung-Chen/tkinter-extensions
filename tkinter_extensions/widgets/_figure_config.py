# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:15:49 2020

@author: tungchentsai
"""

from copy import deepcopy
# =============================================================================
# ---- Common style
# =============================================================================
common_style = {
    "size": ('960p', '540p'),
    
    "suptitle.text": {
        "zorder": 10.0,
        "family": None,
        "size": 20,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 0,
        "sticky": 'n',
        "padx": ('0p', '0p'),
        "pady": ('3p', '3p')
    },
    
    "title.text": {
        "zorder": 10.0,
        "family": None,
        "size": 17,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 0,
        "sticky": 'n',
        "padx": ('0p', '0p'),
        "pady": ('2p', '2p')
    },
    
    "taxis.title.text": {
        "zorder": 4.0,
        "family": None,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 0,
        "sticky": 'n',
        "padx": ('0p', '0p'),
        "pady": ('0p', '3p')
    },
    
    "baxis.title.text": {
        "zorder": 4.0,
        "family": None,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 0,
        "sticky": 's',
        "padx": ('0p', '0p'),
        "pady": ('3p', '0p')
    },
    
    "laxis.title.text": {
        "zorder": 4.0,
        "family": None,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 90,
        "sticky": 'w',
        "padx": ('0p', '3p'),
        "pady": ('0p', '0p')
    },
    
    "raxis.title.text": {
        "zorder": 4.0,
        "family": None,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": -90,
        "sticky": 'e',
        "padx": ('3p', '0p'),
        "pady": ('0p', '0p')
    },
    
    "tticks.label.text": {
        "zorder": 3.0,
        "family": None,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 0,
        "sticky": 'n',
        "padx": ('0p', '0p'),
        "pady": ('0p', '2p')
    },
    "tticks.label.scientific": 4,
    "tticks.label.max_ticks": 13,
    "tticks.line": {
        "zorder": 2.0,
        "width": '1p',
        "smooth": False
    },
    
    "bticks.label.text": {
        "zorder": 3.0,
        "family": None,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 0,
        "sticky": 's',
        "padx": ('0p', '0p'),
        "pady": ('2p', '0p')
    },
    "bticks.label.scientific": 4,
    "bticks.label.max_ticks": 13,
    "bticks.line": {
        "zorder": 2.0,
        "width": '1p',
        "smooth": False
    },
    
    "lticks.label.text": {
        "zorder": 3.0,
        "family": None,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 90,
        "sticky": 'w',
        "padx": ('0p', '2p'),
        "pady": ('0p', '0p')
    },
    "lticks.label.scientific": 4,
    "lticks.label.max_ticks": 13,
    "lticks.line": {
        "zorder": 2.0,
        "width": '1p',
        "smooth": False
    },
    
    "rticks.label.text": {
        "zorder": 3.0,
        "family": None,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": -90,
        "sticky": 'e',
        "padx": ('2p', '0p'),
        "pady": ('0p', '0p')
    },
    "rticks.label.scientific": 4,
    "rticks.label.max_ticks": 13,
    "rticks.line": {
        "zorder": 2.0,
        "width": '1p',
        "smooth": False
    },
    
    "frame.rect": {
        "zorder": 0.0
    },
    
    "grid.enabled": True,
    "grid.line": {
        "zorder": 1.0
    },
    
    "legend.enabled": True,
    "legend.rect": {
        "zorder": 0.0
    },
    "legend.text": {
        "zorder": 1.0,
        "family": None,
        "size": 12,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 0,
        "sticky": 'w',
        "padx": ('2p', '2p'),
        "pady": ('2p', '0p')
    },
    
    "text": {
        "zorder": 5.0,
        "family": None,
        "size": 12,
        "weight": 'normal',
        "slant": 'roman',
        "underline": False,
        "overstrike": False,
        "angle": 0,
        "sticky": 'center',
        "padx": ('0p', '0p'),
        "pady": ('0p', '0p')
    },
    
    "line": {
        "zorder": 5.0,
        "width": '2.0p',
        "smooth": False,
        "color": ''
    },
    
    "marker": {
        "zorder": 5.0,
        "size": '4.0p',
        "edgecolor": ''
    },
    
    "rect": {
        "zorder": 5.0,
        "edgecolor": ''
    },
    
    "toolbar": {
    }
}


# =============================================================================
# ---- Light style
# =============================================================================
light_style = deepcopy(common_style)
light_style.update({
    "colors": [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    "facecolor": '#D9E3F1',
    
    "suptitle.text": {
        **light_style["suptitle.text"],
        "color": 'black'
    },
    
    "title.text": {
        **light_style["title.text"],
        "color": 'black'
    },
    
    "taxis.title.text": {
        **light_style["taxis.title.text"],
        "color": 'black'
    },
    
    "baxis.title.text": {
        **light_style["baxis.title.text"],
        "color": 'black'
    },
    
    "laxis.title.text": {
        **light_style["laxis.title.text"],
        "color": 'black'
    },
    
    "raxis.title.text": {
        **light_style["raxis.title.text"],
        "color": 'black'
    },
    
    "tticks.label.text": {
        **light_style["tticks.label.text"],
        "color": '#1A1A1A'
    },
    "tticks.line": {
        **light_style["tticks.line"],
        "color": 'black'
    },
    
    "bticks.label.text": {
        **light_style["bticks.label.text"],
        "color": '#1A1A1A'
    },
    "bticks.line": {
        **light_style["bticks.line"],
        "color": 'black'
    },
    
    "lticks.label.text": {
        **light_style["lticks.label.text"],
        "color": '#1A1A1A'
    },
    "lticks.line": {
        **light_style["lticks.line"],
        "color": 'black'
    },
    
    "rticks.label.text": {
        **light_style["rticks.label.text"],
        "color": '#1A1A1A'
    },
    "rticks.line": {
        **light_style["rticks.line"],
        "color": 'black'
    },
    
    "frame.rect": {
        **light_style["frame.rect"],
        "facecolor": '#CFD9E8',
        "edgecolor": '#666666'
    },
    
    "grid.color": 'white',
    
    "legend.text": {
        **light_style["legend.text"],
        "color": 'black'
    },
    "legend.rect": {
        **light_style["legend.rect"],
        "facecolor": '#CFD9E8',
        "edgecolor": '#666666'
    },
    
    "text": {
        **light_style["text"],
        "color": 'black'
    },
    
    "rect": {
        **light_style["rect"],
        "facecolor": '',
        "edgecolor": 'black'
    }
})


# =============================================================================
# ---- Dark Style
# =============================================================================
dark_style = deepcopy(light_style)#TODO: update colors


# =============================================================================
# ---- Styles
# =============================================================================
STYLES = {"light": light_style, "dark": dark_style}

