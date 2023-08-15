# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:10:52 2023

@author: Jeff_Tsai
"""

import ttkbootstrap as ttk

from ..utils import redirect_layout_managers
# =============================================================================
# ---- Widgets
# =============================================================================
class CollapsedFrame(ttk.Frame):
    def __init__(self,
                 master=None,
                 text='',
                 labelwidget=None,
                 orient='vertical',
                 style=None,
                 bootstyle=None,
                 **kw):
        assert orient in ('vertical', 'horizontal'), orient
        assert (not style) or ('TButton' in style) or ('TFrame' in style), style
        assert text or labelwidget, (text, labelwidget)
        
        self._orient = orient
        bootstyle_button = bootstyle
        if bootstyle:
            bootstyle = ttk.Bootstyle.ttkstyle_widget_color(bootstyle)
        
        if style and ('TButton' in style):
            style_button = style
            style = None
        else:
            style_button = None
        
        self._labelwidget = labelwidget = labelwidget or ttk.Button(
            master,
            text=text,
            takefocus=False,
            style=style_button,
            bootstyle=bootstyle_button
        )
        
        self._container = container = ttk.Labelframe(master,
                                                     labelwidget=labelwidget,
                                                     style=style,
                                                     bootstyle=bootstyle,
                                                     **kw)
        
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        super().__init__(container, relief='flat', borderwidth=0)
        self.grid(sticky='nesw')
        
        labelwidget.configure(command=self.toggle)
        redirect_layout_managers(self, container, orig_prefix='content_')
    
    @property
    def container(self):
        return self._container
    
    @property
    def labelwidget(self):
        return self._labelwidget
    
    def toggle(self):
        if self.content_grid_info():  # hide
            if self._orient == 'vertical':
                width = self.container.winfo_width()
                height = self.labelwidget.winfo_height()
            else:  # horizontal
                width = self.labelwidget.winfo_width()
                height = self.container.winfo_height()
            self.container.grid_propagate(0)
            self.content_grid_remove()
            self.container.configure(width=width, height=height)
        else:  # show
            self.container.grid_propagate(1)
            self.content_grid()


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    root = ttk.Window(themename='cyborg', size=(200, 100))
    
    collapsed = CollapsedFrame(root,
                               text='Click me to collapse',
                               orient='vertical',
                               bootstyle='success-outline')
    collapsed.grid(padx=6, pady=6)
    ttk.Label(collapsed, text='Label 1').pack()
    ttk.Label(collapsed, text='Label 2').pack()
    
    root.mainloop()

