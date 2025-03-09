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
    "colors": [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    
    "suptitle.text": {
        "zorder": 10.0,
        "size": 20,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'n',
        "pady": ('3p', '3p')
    },
    
    "title.text": {
        "zorder": 10.0,
        "size": 17,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'n',
        "pady": ('2p', '2p')
    },
    
    "taxis.label.text": {
        "zorder": 4.0,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'n',
        "pady": ('0p', '3p')
    },
    
    "baxis.label.text": {
        "zorder": 4.0,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 's',
        "pady": ('3p', '0p')
    },
    
    "laxis.label.text": {
        "zorder": 4.0,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 90,
        "sticky": 'w',
        "padx": ('0p', '3p')
    },
    
    "raxis.label.text": {
        "zorder": 4.0,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "angle": -90,
        "sticky": 'e',
        "padx": ('3p', '0p')
    },
    
    "tticks.labels.text": {
        "zorder": 3.0,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'n',
        "pady": ('0p', '6p')
    },
    "tticks.labels.scientific": 4,
    "tticks.labels.max_ticks": 13,
    "tticks.margins": ('9p', '9p'),
    "tticks.ticks.line": {
        "zorder": 2.0,
        "width": '1p'
    },
    
    "bticks.labels.text": {
        "zorder": 3.0,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 's',
        "pady": ('6p', '0p')
    },
    "bticks.labels.scientific": 4,
    "bticks.labels.max_ticks": 13,
    "bticks.margins": ('9p', '9p'),
    "bticks.ticks.line": {
        "zorder": 2.0,
        "width": '1p'
    },
    
    "lticks.labels.text": {
        "zorder": 3.0,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 90,
        "sticky": 'w',
        "padx": ('0p', '6p')
    },
    "lticks.labels.scientific": 4,
    "lticks.labels.max_ticks": 13,
    "lticks.margins": ('9p', '9p'),
    "lticks.ticks.line": {
        "zorder": 2.0,
        "width": '1p'
    },
    
    "rticks.labels.text": {
        "zorder": 3.0,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "angle": -90,
        "sticky": 'e',
        "padx": ('6p', '0p')
    },
    "rticks.labels.scientific": 4,
    "rticks.labels.max_ticks": 13,
    "rticks.margins": ('9p', '9p'),
    "rticks.ticks.line": {
        "zorder": 2.0,
        "width": '1p'
    },
    
    "frame.rect": {
        "zorder": 0.0,
        "width": '1p'
    },
    
    "frame.grid.enabled": ('b', 'l'),
    "frame.grid.line": {
        "zorder": 1.0,
        "width": '1p'
    },
    
    "legend.enabled": True,
    "legend.edgewidth": '1p',
    "legend.width": '150p',
    "legend.padx": ('6p', '0p'),
    "legend.ipadx": ('9p', '6p'),
    "legend.ipady": ('0p', '0p'),
    "legend.symbols.width": '15p',
    "legend.labels.text": {
        "zorder": 1.0,
        "size": 12,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'nw',
        "padx": ('3p', '0p'),
        "pady": ('3p', '0p')
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
        "width": '2p',
        "smooth": False,
        "dash": ''
    },
    
    "rect": {
        "zorder": 5.0,
        "width": '2p',
        "facecolor": ''
    }
}


# =============================================================================
# ---- Light style
# =============================================================================
light_style = deepcopy(common_style)
light_style.update({
    "facecolor": '#D9E3F1',
    
    "taxis.label.text": {
        **common_style["taxis.label.text"],
        "color": 'black'
    },
    
    "baxis.label.text": {
        **common_style["baxis.label.text"],
        "color": 'black'
    },
    
    "laxis.label.text": {
        **common_style["laxis.label.text"],
        "color": 'black'
    },
    
    "raxis.label.text": {
        **common_style["raxis.label.text"],
        "color": 'black'
    },
    
    "tticks.labels.text": {
        **common_style["tticks.labels.text"],
        "color": '#1A1A1A'
    },
    
    "bticks.labels.text": {
        **common_style["bticks.labels.text"],
        "color": '#1A1A1A'
    },
    
    "lticks.labels.text": {
        **common_style["lticks.labels.text"],
        "color": '#1A1A1A'
    },
    
    "rticks.labels.text": {
        **common_style["rticks.labels.text"],
        "color": '#1A1A1A'
    },
    
    "frame.rect": {
        **common_style["frame.rect"],
        "facecolor": '#CFD9E8',
        "edgecolor": '#666666'
    },
    "frame.grid.line": {
        **common_style["frame.grid.line"],
        "color": 'white'
    },
    
    "legend.facecolor": '#D9E3F1',
    "legend.edgecolor": '#D9E3F1',
    
    "text": {
        **common_style["text"],
        "color": 'black'
    },
    
    "line": {
        **common_style["line"],
        "color": 'black'
    },
    
    "rect": {
        **common_style["rect"],
        "edgecolor": 'black'
    }
})


# =============================================================================
# ---- Dark Style
# =============================================================================
dark_style = deepcopy(common_style)
dark_style.update({
    "facecolor": 'black',
    
    "taxis.label.text": {
        **common_style["taxis.label.text"],
        "color": '#F2F2F2'
    },
    
    "baxis.label.text": {
        **common_style["baxis.label.text"],
        "color": '#F2F2F2'
    },
    
    "laxis.label.text": {
        **common_style["laxis.label.text"],
        "color": '#F2F2F2'
    },
    
    "raxis.label.text": {
        **common_style["raxis.label.text"],
        "color": '#F2F2F2'
    },
    
    "tticks.labels.text": {
        **common_style["tticks.labels.text"],
        "color": '#F2F2F2'
    },
    
    "bticks.labels.text": {
        **common_style["bticks.labels.text"],
        "color": '#F2F2F2'
    },
    
    "lticks.labels.text": {
        **common_style["lticks.labels.text"],
        "color": '#F2F2F2'
    },
    
    "rticks.labels.text": {
        **common_style["rticks.labels.text"],
        "color": '#F2F2F2'
    },
    
    "frame.rect": {
        **common_style["frame.rect"],
        "facecolor": 'black',
        "edgecolor": '#808080'
    },
    "frame.grid.line": {
        **common_style["frame.grid.line"],
        "color": '#404040'
    },
    
    "legend.facecolor": 'black',
    "legend.edgecolor": 'black',
    
    "text": {
        **common_style["text"],
        "color": 'white'
    },
    
    "line": {
        **common_style["line"],
        "color": 'white'
    },
    
    "rect": {
        **common_style["rect"],
        "edgecolor": 'white'
    }
})


# =============================================================================
# ---- Styles
# =============================================================================
STYLES = {"light": light_style, "dark": dark_style}

