# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:15:49 2020

@author: tungchentsai

USAGE:
    import matplotlib as mpl
    from rc_style import rc_style_light
    with plt.rc_context(rc_style_light):
        # do something
	# or decorate functions using `@use_rc_style(rc_style_light)`
    # or `mpl.rcParams.update(rc_style_light)`

DEFAULT:
Z-order	 Artist
==================================================================
0        Images (AxesImage, FigureImage, BboxImage)
1        Patch, PatchCollection
2        Line2D, LineCollection (including minor ticks, grid lines)
2.01     Major ticks
3        Text (including axes labels and titles)
5        Legend

TUTORIALS:
https://matplotlib.org/stable/tutorials/introductory/customizing.html
"""

# =============================================================================
# %% Styles ( default: `matplotlib.rcdefaults()` )
# =============================================================================
common_style = {
    "interactive": False,
    "savefig.dpi": 120,
    
    "font.family": 'serif',   # default: 'sans-serif'
    "font.size": 15.0,   # default: 10.0
    
    "mathtext.fontset": 'stix',   # default: 'dejavusans'
    
    "figure.figsize": (9, 6),
    "figure.dpi": 54,   # default: 120
    "figure.constrained_layout.use": True,
#    "figure.autolayout": True,   # this enables fig.tight_layout()
    "figure.titlesize": 'x-large',   # default: 'large'
    
    "axes.formatter.limits": (-3, 3),
    "axes.formatter.use_mathtext": True,
    "axes.grid": True,
    "axes.grid.which": 'both',   # default: 'major'
    "axes.axisbelow": True,
    "axes.titlesize": 'large',
    "axes.linewidth": 1.,
    
    "lines.linewidth": 2,
    "lines.markersize": 4,
    "lines.markeredgewidth": 1.2,
    
    "pcolor.shading": 'nearest',   # not exists for matplotlib version < 3.3
    
    "legend.loc": 'best',
    "legend.fontsize": 'small',
    
#    "xtick.minor.visible": True,
    
#    "ytick.minor.visible": True,
}

light_style = common_style.copy()
light_style.update({
    "text.color":'black',
    
    "savefig.facecolor": 'white',
    
    "figure.facecolor": '#D9E3F1',
    "figure.edgecolor": '#D9E3F1',
    
    "axes.facecolor": '#CFD9E8',
    "axes.edgecolor": '#666666',
    "axes.labelcolor": '#1A1A1A',
    
    "patch.edgecolor": '#444444',
    
    "grid.color": 'white',
    
    "legend.facecolor": '#CFD9E8',
    "legend.edgecolor": '#666666',
    "legend.labelcolor": 'black',
    
    "xtick.color": '#1A1A1A',
    
    "ytick.color": '#1A1A1A',
})

dark_style = common_style.copy()
dark_style.update({
    "text.color": 'white',
    
    "savefig.facecolor": 'black',
    
    "figure.facecolor": 'black',
    "figure.edgecolor": 'black',
    
    "axes.facecolor": 'black',
    "axes.edgecolor": '#808080',
    "axes.labelcolor": '#F2F2F2',
    
    "patch.edgecolor": '#D3D3D3',
    
    "grid.color": '#404040',
    
    "legend.facecolor": 'black',
    "legend.edgecolor": '#808080',
    "legend.labelcolor": 'white',
    
    "xtick.color": '#F2F2F2',
    
    "ytick.color": '#F2F2F2',
})


RC = {"light": light_style, "dark": dark_style}

