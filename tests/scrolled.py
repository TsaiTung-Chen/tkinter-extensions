#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:40:38 2025

@author: tungchentsai
"""

import ttkbootstrap as ttk

from tkinter_extensions.utils import quit_if_all_closed
from tkinter_extensions.widgets import (
    Window, ScrolledText, ScrolledFrame, ScrolledCanvas
)


root = Window(themename='cyborg')
root.withdraw()


win1 = ttk.Toplevel(title='ScrolledText')
win1.lift()

st = ScrolledText(win1, autohide=True, wrap='none', readonly=True)
st.insert('end', ttk.tk.__doc__)
st.pack(fill='both', expand=True)


win2 = ttk.Toplevel(title='ScrolledFrame')
win2.lift()

sf = ScrolledFrame(win2, autohide=False, scroll_orient='vertical')
sf.pack(fill='both', expand=True)
for i in range(20):
    text = str(i) + ': ' + '_'.join(str(i) for i in range(30))
    ttk.Button(sf, text=text).pack(anchor='e')


win3 = ttk.Toplevel(title='ScrolledCanvas')
win3.lift()

sc = ScrolledCanvas(
    win3, scroll_orient='vertical', autohide=False, vbootstyle='round-light'
)
sc.pack(fill='both', expand=True)
sc.create_polygon(
    (10, 5), (600, 300), (900, 600), (300, 600), (300, 600),
    outline='red', stipple='gray25'
)
sc.configure(bg='gray')


for win in [win1, win2, win3]:
    win.place_window_center()
    win.protocol('WM_DELETE_WINDOW', quit_if_all_closed(win))


root.mainloop()