#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 00:32:44 2024

@author: tungchentsai
"""

from time import sleep
from multiprocessing import Process, Event

import ttkbootstrap as ttk
# =============================================================================
# ---- Functions
# =============================================================================
def main():
    event = Event()
    
    print('Startup...')
    process = Process(target=show_splashscreen, args=(event,), daemon=True)
    process.start()
    
    print('Initializing...')
    root = ttk.Window(title='Main Program', size=[500, 500])
    root.wm_withdraw()
    ttk.Label(root, text='Main program').pack(fill='y', expand=True)
    sleep(3)  # mimic some time-consuming startup process
    
    print('Remove the splash screen.')
    root.after(1000, event.set)
    
    print('Start mainloop.')
    root.place_window_center()
    root.wm_deiconify()
    root.mainloop()


def show_splashscreen(event: Event):
    root = ttk.Window(size=[250, 250], overrideredirect=True)
    root.wm_withdraw()
    root.attributes('-topmost', 1)
    ttk.Label(root, text='Initializing...').pack(side='bottom', pady=15)
    root.place_window_center()
    root.wm_deiconify()
    root.after_idle(remove_splashscreen_once_event_set, root, event)
    root.mainloop()


def remove_splashscreen_once_event_set(root, event: Event) -> bool:
    if event.is_set():
        root.destroy()
        return True
    else:
        root.after(500, remove_splashscreen_once_event_set, root, event)
        return False


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    main()

