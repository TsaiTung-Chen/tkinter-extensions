#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:58:14 2024

@author: tungchentsai
"""

import ttkbootstrap as ttk
import tkinter_extensions as te


root = ttk.Window(themename='cyborg')
root.withdraw()

win1 = ttk.Toplevel(title='ScrolledText')
win1.lift()
st = te.widgets.ScrolledText(win1, autohide=True, wrap='none', readonly=True)
st.insert('end', ttk.tk.__doc__)
st.pack(fill='both', expand=True)

win2 = ttk.Toplevel(title='ScrolledFrame')
win2.lift()
sf = te.widgets.ScrolledFrame(win2, autohide=False, scroll_orient='vertical')
sf.pack(fill='both', expand=True)
for i in range(20):
    text = str(i) + ': ' + '_'.join(str(i) for i in range(30))
    ttk.Button(sf, text=text).pack(anchor='e')

win3 = ttk.Toplevel(title='ScrolledCanvas')
win3.lift()
sc = te.widgets.ScrolledWidget(win3, widget=ttk.Canvas, vbootstyle='round-light')
sc.pack(fill='both', expand=True)
sc.create_polygon((10, 100), (600, 300), (900, 600), (300, 600), (300, 600),
                  outline='red', stipple='gray25', smooth=1)
x1, y1, x2, y2 = sc.bbox('all')
sc.configure(bg='gray', width=x2, height=y2)

for win in [win1, win2, win3]:
    win.place_window_center()
    win.protocol('WM_DELETE_WINDOW', te.utils.quit_if_all_closed(win))

root.mainloop()

