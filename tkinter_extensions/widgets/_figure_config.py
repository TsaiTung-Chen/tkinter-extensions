# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:15:49 2020

@author: tungchentsai
"""

from copy import deepcopy
# =============================================================================
# ---- Common style
# =============================================================================
"""
Zorder:
    0.     frame: grid
    1~100. user defined
    101.   frame: cover, edge
    102.   ticks: ticks, labels
    103.   axis: label
    104.   legend
    105.   title
    106.   suptitle
    107.   datalabel: arrow, point, box, text
    108.   veil: rectangle
    109.   toolbar: rubberband
"""

common_style = {
    "size": ('960p', '540p'),
    "colors": [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    
    #TODO: margin
    
    "frame": {
    },
    "frame.cover.polygon": {
        "zorder": 101.0,
        "width": '1p',
        "edgecolor": ''
    },
    "frame.edge.rectangle": {
        "zorder": 101.1,
        "width": '1p'
    },
    "frame.grid.enabled": ('b', 'l'),
    "frame.grid.line": {
        "zorder": 0.0,
        "width": '1p'
    },
    
    "suptitle.text": {
        "zorder": 106.0,
        "size": 20,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'n',
        "pady": ('3p', '3p')
    },
    
    "title.text": {
        "zorder": 105.0,
        "size": 17,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'n',
        "pady": ('2p', '2p')
    },
    
    "taxis.label.text": {
        "zorder": 103.0,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'n',
        "pady": ('0p', '3p')
    },
    
    "baxis.label.text": {
        "zorder": 103.0,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 's',
        "pady": ('3p', '0p')
    },
    
    "laxis.label.text": {
        "zorder": 103.0,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 90,
        "sticky": 'w',
        "padx": ('0p', '3p')
    },
    
    "raxis.label.text": {
        "zorder": 103.0,
        "size": 13,
        "weight": 'normal',
        "slant": 'roman',
        "angle": -90,
        "sticky": 'e',
        "padx": ('3p', '0p')
    },
    
    "tticks.labels.scientific": 4,
    "tticks.labels.max_ticks": 13,
    "tticks.margins": ('9p', '9p'),
    "tticks.ticks.line": {
        "zorder": 102.0,
        "width": '1p'
    },
    "tticks.labels.text": {
        "zorder": 102.1,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'n',
        "pady": ('0p', '6p')
    },
    
    "bticks.labels.scientific": 4,
    "bticks.labels.max_ticks": 13,
    "bticks.margins": ('9p', '9p'),
    "bticks.ticks.line": {
        "zorder": 102.0,
        "width": '1p'
    },
    "bticks.labels.text": {
        "zorder": 102.1,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 's',
        "pady": ('6p', '0p')
    },
    
    "lticks.labels.scientific": 4,
    "lticks.labels.max_ticks": 13,
    "lticks.margins": ('9p', '9p'),
    "lticks.ticks.line": {
        "zorder": 102.0,
        "width": '1p'
    },
    "lticks.labels.text": {
        "zorder": 102.1,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 90,
        "sticky": 'w',
        "padx": ('0p', '6p')
    },
    
    "rticks.labels.scientific": 4,
    "rticks.labels.max_ticks": 13,
    "rticks.margins": ('9p', '9p'),
    "rticks.ticks.line": {
        "zorder": 102.0,
        "width": '1p'
    },
    "rticks.labels.text": {
        "zorder": 102.1,
        "size": 11,
        "weight": 'normal',
        "slant": 'roman',
        "angle": -90,
        "sticky": 'e',
        "padx": ('6p', '0p')
    },
    
    "legend.enabled": True,
    "legend.edgewidth": '1p',
    "legend.width": '150p',
    "legend.padx": ('6p', '0p'),
    "legend.ipadx": ('9p', '6p'),
    "legend.ipady": ('0p', '0p'),
    "legend.symbols.width": '15p',
    "legend.labels.text": {
        "zorder": 104.0,
        "size": 12,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'nw',
        "padx": ('3p', '0p'),
        "pady": ('3p', '0p')
    },
    
    "datalabel.offset": ('0p', '-36p'),
    "datalabel.scientific": 4,
    "datalabel.arrow.polygon": {
        "zorder": 107.0,
        "width": '1p'
    },
    "datalabel.point.oval": {
        "zorder": 107.1,
        "width": '1p'
    },
    "datalabel.box.rectangle": {
        "zorder": 107.2,
        "width": '1p'
    },
    "datalabel.text": {
        "zorder": 107.3,
        "size": 12,
        "weight": 'normal',
        "slant": 'roman',
        "angle": 0,
        "sticky": 'center',
        "padx": ('6p', '6p'),
        "pady": ('6p', '6p')
    },
    
    "veil.rectangle": {
        "zorder": 108.0,
        "width": '0p'
    },
    
    "toolbar.rubberband.rectangle": {
        "zorder": 109.0,
        "width": '1p'
    },
    
    "text": {
        "zorder": 1.0,
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
        "zorder": 1.0,
        "width": '2p',
        "smooth": False,
        "dash": ''
    },
    
    "rectangle": {
        "zorder": 1.0,
        "width": '2p',
        "facecolor": ''
    },
    
    "oval": {
        "zorder": 1.0,
        "width": '2p',
        "facecolor": ''
    },
    
    "polygon": {
        "zorder": 1.0,
        "width": '2p',
        "smooth": False,
        "facecolor": ''
    }
}


# =============================================================================
# ---- Light style
# =============================================================================
light_facecolor = '#D9E3F1'
light_frame_facecolor = '#CFD9E8'
light_frame_edgecolor = '#666666'

light_style = deepcopy(common_style)
light_style.update({
    "frame": {
        **common_style["frame"],
        "facecolor": light_frame_facecolor
    },
    "frame.cover.polygon": {
        **common_style["frame.cover.polygon"],
        "facecolor": light_facecolor
    },
    "frame.edge.rectangle": {
        **common_style["frame.edge.rectangle"],
        "edgecolor": light_frame_edgecolor
    },
    "frame.grid.line": {
        **common_style["frame.grid.line"],
        "color": 'white'
    },
    
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
    
    "legend.facecolor": light_facecolor,
    "legend.edgecolor": light_facecolor,
    
    "datalabel.point.oval": {
        **common_style["datalabel.point.oval"],
        "edgecolor": '#1A1A1A'
    },
    "datalabel.arrow.polygon": {
        **common_style["datalabel.arrow.polygon"],
        "edgecolor": '#1A1A1A'
    },
    "datalabel.box.recgangle": {
        **common_style["datalabel.box.rectangle"],
        "edgecolor": '#1A1A1A'
    },
    
    "toolbar.rubberband.rectangle": {
        **common_style["toolbar.rubberband.rectangle"],
        "edgecolor": 'black'
    },
    
    "text": {
        **common_style["text"],
        "color": 'black'
    },
    
    "line": {
        **common_style["line"],
        "color": 'black'
    },
    
    "rectangle": {
        **common_style["rectangle"],
        "edgecolor": 'black'
    },
    
    "oval": {
        **common_style["oval"],
        "edgecolor": 'black'
    },
    
    "polygon": {
        **common_style["polygon"],
        "edgecolor": 'black'
    }
})


# =============================================================================
# ---- Dark Style
# =============================================================================
dark_facecolor = 'black'
dark_frame_facecolor = 'black'
dark_frame_edgecolor = '#808080'

dark_style = deepcopy(common_style)
dark_style.update({
    "frame": {
        **common_style["frame"],
        "facecolor": dark_frame_facecolor
    },
    "frame.cover.polygon": {
        **common_style["frame.cover.polygon"],
        "facecolor": dark_facecolor
    },
    "frame.edge.rectangle": {
        **common_style["frame.edge.rectangle"],
        "edgecolor": dark_frame_edgecolor
    },
    "frame.grid.line": {
        **common_style["frame.grid.line"],
        "color": '#404040'
    },
    
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
    
    "legend.facecolor": dark_facecolor,
    "legend.edgecolor": dark_facecolor,
    
    "datalabel.point.oval": {
        **common_style["datalabel.point.oval"],
        "edgecolor": '#F2F2F2'
    },
    "datalabel.arrow.polygon": {
        **common_style["datalabel.arrow.polygon"],
        "edgecolor": '#F2F2F2'
    },
    "datalabel.box.recgangle": {
        **common_style["datalabel.box.rectangle"],
        "edgecolor": '#F2F2F2'
    },
    
    "toolbar.rubberband.rectangle": {
        **common_style["toolbar.rubberband.rectangle"],
        "edgecolor": 'white'
    },
    
    "text": {
        **common_style["text"],
        "color": 'white'
    },
    
    "line": {
        **common_style["line"],
        "color": 'white'
    },
    
    "rectangle": {
        **common_style["rectangle"],
        "edgecolor": 'white'
    },
    
    "oval": {
        **common_style["oval"],
        "edgecolor": 'white'
    },
    
    "polygon": {
        **common_style["polygon"],
        "edgecolor": 'white'
    }
})


# =============================================================================
# ---- Styles
# =============================================================================
STYLES = {"light": light_style, "dark": dark_style}

