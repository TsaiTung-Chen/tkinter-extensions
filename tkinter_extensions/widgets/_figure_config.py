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
    
    "taxis.label.text": {
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
    
    "baxis.label.text": {
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
    
    "laxis.label.text": {
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
    
    "raxis.label.text": {
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
    
    "tticks.labels.text": {
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
    "tticks.labels.scientific": 4,
    "tticks.labels.max_ticks": 13,
    "tticks.line": {
        "zorder": 2.0,
        "width": '1p',
        "smooth": False
    },
    
    "bticks.labels.text": {
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
    "bticks.labels.scientific": 4,
    "bticks.labels.max_ticks": 13,
    "bticks.line": {
        "zorder": 2.0,
        "width": '1p',
        "smooth": False
    },
    
    "lticks.labels.text": {
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
    "lticks.labels.scientific": 4,
    "lticks.labels.max_ticks": 13,
    "lticks.line": {
        "zorder": 2.0,
        "width": '1p',
        "smooth": False
    },
    
    "rticks.labels.text": {
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
    "rticks.labels.scientific": 4,
    "rticks.labels.max_ticks": 13,
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
        "size": '2.0p',
        "edgecolor": ''
    },
    
    "rect": {
        "zorder": 5.0,
        "width": '2.0p',
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
    
    "taxis.label.text": {
        **light_style["taxis.label.text"],
        "color": 'black'
    },
    
    "baxis.label.text": {
        **light_style["baxis.label.text"],
        "color": 'black'
    },
    
    "laxis.label.text": {
        **light_style["laxis.label.text"],
        "color": 'black'
    },
    
    "raxis.label.text": {
        **light_style["raxis.label.text"],
        "color": 'black'
    },
    
    "tticks.labels.text": {
        **light_style["tticks.labels.text"],
        "color": '#1A1A1A'
    },
    "tticks.line": {
        **light_style["tticks.line"],
        "color": 'black'
    },
    
    "bticks.labels.text": {
        **light_style["bticks.labels.text"],
        "color": '#1A1A1A'
    },
    "bticks.line": {
        **light_style["bticks.line"],
        "color": 'black'
    },
    
    "lticks.labels.text": {
        **light_style["lticks.labels.text"],
        "color": '#1A1A1A'
    },
    "lticks.line": {
        **light_style["lticks.line"],
        "color": 'black'
    },
    
    "rticks.labels.text": {
        **light_style["rticks.labels.text"],
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

