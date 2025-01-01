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
    "size": (960, 540),
    "suptitle": {
        "text": {
            "zorder": 10.0,
            "family": None,
            "size": 20,
            "weight": 'normal',
            "slant": 'roman',
            "underline": False,
            "overstrike": False,
            "angle": 0,
            "sticky": 'n',   # 'n', 'ne', 'e', 'w', or 'nw'
            "padx": (0, 0),
            "pady": (3, 3)
        },
    },
    "plot": {
    },
    "title": {
        "text": {
            "zorder": 10.0,
            "family": None,
            "size": 17,
            "weight": 'normal',
            "slant": 'roman',
            "underline": False,
            "overstrike": False,
            "angle": 0,
            "sticky": 'n',   # 'n', 'ne', 'e', 'w', or 'nw'
            "padx": (0, 0),
            "pady": (2, 2)
        },
    },
    "frame": {
        "zorder": 0.0,
        "grid": {
            "zorder": 1.0,
            "enabled": True
        }
    },
    "taxis": {
        "title": {
            "text": {
                "zorder": 3.0,
                "family": None,
                "size": 13,
                "weight": 'normal',
                "slant": 'roman',
                "underline": False,
                "overstrike": False,
                "angle": 0,
                "sticky": 'n',
                "padx": (0, 0),
                "pady": (3, 3)
            },
        },
        "tick": {
            "text": {
                "zorder": 2.0,
                "family": None,
                "size": 11,
                "weight": 'normal',
                "slant": 'roman',
                "underline": False,
                "overstrike": False,
                "angle": 0,
                "sticky": 'n',
                "padx": (0, 0),
                "pady": (2, 2)
            },
            "scientific": (4, -4),
        },
        "linewidth": 1.0
    },
    "baxis": {
        "title": {
            "text": {
                "zorder": 3.0,
                "family": None,
                "size": 13,
                "weight": 'normal',
                "slant": 'roman',
                "underline": False,
                "overstrike": False,
                "angle": 0,
                "sticky": 's',
                "padx": (0, 0),
                "pady": (3, 3)
            },
        },
        "tick": {
            "text": {
                "zorder": 2.0,
                "family": None,
                "size": 11,
                "weight": 'normal',
                "slant": 'roman',
                "underline": False,
                "overstrike": False,
                "angle": 0,
                "sticky": 's',
                "padx": (0, 0),
                "pady": (2, 2)
            },
            "scientific": (4, -4),
        },
        "linewidth": 1.0
    },
    "laxis": {
        "title": {
            "text": {
                "zorder": 3.0,
                "family": None,
                "size": 13,
                "weight": 'normal',
                "slant": 'roman',
                "underline": False,
                "overstrike": False,
                "angle": 90,
                "sticky": 'w',
                "padx": (3, 3),
                "pady": (0, 0)
            },
        },
        "tick": {
            "text": {
                "zorder": 2.0,
                "family": None,
                "size": 11,
                "weight": 'normal',
                "slant": 'roman',
                "underline": False,
                "overstrike": False,
                "angle": 0,
                "sticky": 'w',
                "padx": (2, 2),
                "pady": (0, 0)
            },
            "scientific": (4, -4),
        },
        "linewidth": 1.0
    },
    "raxis": {
        "title": {
            "text": {
                "zorder": 3.0,
                "family": None,
                "size": 13,
                "weight": 'normal',
                "slant": 'roman',
                "underline": False,
                "overstrike": False,
                "angle": -90,
                "sticky": 'w',
                "padx": (3, 3),
                "pady": (0, 0)
            },
        },
        "tick": {
            "text": {
                "zorder": 2.0,
                "family": None,
                "size": 11,
                "weight": 'normal',
                "slant": 'roman',
                "underline": False,
                "overstrike": False,
                "angle": 0,
                "sticky": 'w',
                "padx": (2, 2),
                "pady": (0, 0)
            },
            "scientific": (4, -4),
        },
        "linewidth": 1.0
    },
    "legend": {
        "enabled": True,
        "text": {
            "zorder": 5.0,
            "family": None,
            "size": 12,
            "weight": 'normal',
            "slant": 'roman',
            "underline": False,
            "overstrike": False,
            "angle": 0,
        }
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
        "padx": (0, 0),
        "pady": (0, 0)
    },
    "line": {
        "zorder": 4.0,
        "width": 2.0,
        "smooth": False
    },
    "marker": {
        "zorder": 4.0,
        "size": 4.0
    },
    "rect": {
        "zorder": 4.0
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
    ]
})

light_style.update({
    "facecolor": '#D9E3F1'
})

light_style["suptitle"]["text"].update({
    "color": 'black'
})

light_style["title"]["text"].update({
    "color": 'black'
})

light_style["frame"].update({
    "facecolor": '#CFD9E8',
    "edgecolor": '#666666'
})
light_style["frame"]["grid"].update({
    "color": 'white'
})

light_style["taxis"]["title"]["text"].update({
    "color": 'black'
})
light_style["taxis"]["tick"].update({
    "color": 'black'
})
light_style["taxis"]["tick"]["text"].update({
    "color": '#1A1A1A'
})

light_style["baxis"]["title"]["text"].update({
    "color": 'black'
})
light_style["baxis"]["tick"].update({
    "color": 'black'
})
light_style["baxis"]["tick"]["text"].update({
    "color": '#1A1A1A'
})

light_style["laxis"]["title"]["text"].update({
    "color": 'black'
})
light_style["laxis"]["tick"].update({
    "color": 'black'
})
light_style["laxis"]["tick"]["text"].update({
    "color": '#1A1A1A'
})

light_style["raxis"]["title"]["text"].update({
    "color": 'black'
})
light_style["raxis"]["tick"].update({
    "color": 'black'
})
light_style["raxis"]["tick"]["text"].update({
    "color": '#1A1A1A'
})

light_style["legend"]["text"].update({
    "color": 'black'
})
light_style["legend"].update({
    "facecolor": '#CFD9E8',
    "edgecolor": '#666666'
})

light_style["text"].update({
    "color": 'black'
})

light_style["rect"].update({
    "facecolor": '',
    "edgecolor": 'black'
})

light_style["toolbar"].update({
    "facecolor": '#D9E3F1'
})


# =============================================================================
# ---- Dark Style
# =============================================================================
dark_style = deepcopy(light_style)#TODO: update colors


# =============================================================================
# ---- Styles
# =============================================================================
STYLES = {"light": light_style, "dark": dark_style}

