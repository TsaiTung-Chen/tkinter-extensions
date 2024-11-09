#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 12:04:44 2024

@author: tungchentsai
"""
import tkinter as tk

import sounddevice as sd
import ttkbootstrap as ttk
import tkinter_extensions as te
# =============================================================================
# ---- Functions
# =============================================================================
def query_devices(
        device: int | str | None = None,
        kind: str | None = None,
        always_list: bool = False
) -> dict | sd.DeviceList:
    """
    Query the available audio devices.

    Parameters
    ----------
    device : int | str | None, optional
        Numeric device ID or device name substring(s).
        If specified, information about only the given `device` is returned in
        a single dictionary. The default is `None`.
    kind : str | None, optional
        If `device` is not specified and `kind` is `'input'` or `'output'`, a
        single dictionary is returned with information about the default input
        or output device, respectively. If `device` is not specified and `kind`
        is `'all-input'` or `'all-output'`, a `DeviceList` is returned with 
        information about all input or output devices, respectively. The
        default is `None`.
    always_list : bool, optional
        If yes, always return a `DeviceList`. The default is `False`.

    Returns
    -------
    dict | DeviceList
        A dictionary with information about the given `device` or a `DeviceList`
        containing one dictionary for each available device.

    """
    assert kind in ('input', 'output', 'all-input', 'all-output', None), kind
    
    if kind in ('all-input', 'all-output'):
        all_in_or_out = always_list = True
        kind = kind[4:]
    else:
        all_in_or_out = False
    
    if device is None and all_in_or_out:
        devices_info = sd.query_devices()
        devices_info = sd.DeviceList(
            filter(lambda info: info[f"max_{kind}_channels"] > 0, devices_info))
    else:
        devices_info = sd.query_devices(device=device, kind=kind)
    
    if isinstance(devices_info, sd.DeviceList):
        for info in devices_info:
            _add_hostapi_name(info)
        return devices_info
    
    assert isinstance(devices_info, dict), type(devices_info)
    
    _add_hostapi_name(devices_info)
    
    if always_list:
        return sd.DeviceList((devices_info,))
    return devices_info


def _add_hostapi_name(device_info: dict) -> dict:
    assert isinstance(device_info, dict), type(device_info)
    
    # Find hostapi name
    hostapi = sd.query_hostapis(device_info["hostapi"])
    device_info["hostapi_name"] = hostapi["name"]
    
    # Reorder the keys and values
    keys = list(device_info)
    idx = keys.index('hostapi')
    keys.pop(idx)
    keys.insert(idx, 'hostapi_name')
    keys.append('hostapi')
    device_info_backup = device_info.copy()
    device_info.clear()
    for key in keys:
        device_info[key] = device_info_backup[key]
    
    return device_info


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    devs_info = query_devices()
    idevs_info = query_devices(kind='all-input')
    odevs_info = query_devices(kind='all-output')
    odev_info = query_devices(kind='output')
    
    ichannels, ochannels = 2, 2
    imapping, omapping = [1], [1]
    
    root = ttk.Window('SoundDevice Test', themename='cyborg')
    root.wm_withdraw()
    
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill='both', expand=True)
    
    # All devices available
    info_lf = ttk.Labelframe(frame, text=' Devices ', padding=6)
    info_lf.pack(fill='both', expand=True)
    ttk.Label(info_lf, text=repr(devs_info)).pack()
    
    # Set output device
    odev_lf = ttk.Labelframe(frame, text='Device Out', padding=6)
    odev_lf.pack(fill='both', expand=True, pady=[6, 0])
    odev = tk.IntVar(root, value=odev_info["index"])
    odev_cb = ttk.OptionMenu(
        odev_lf,
        odev,
        None,
        *[ info["index"] for info in odevs_info ],
        bootstyle='outline'
    )
    odev_cb.pack(anchor='w')
    
    def _update_mapping(kind: str):
        assert kind in ('input', 'output'), kind
        
        if kind == 'input':
            global imapping
            imapping = [ int(w["text"]) for w in odev_dnd.dnd_widgets ]
        else:
            global omapping
            omapping = [ int(w["text"]) for w in odev_dnd.dnd_widgets ]
    
    odev_dnd = te.widgets.OrderlyContainer(odev_lf)
    odev_dnd.pack(fill='both', expand=True, pady=[6, 0])
    odev_dnd.dnd_put(
        [[ ttk.Button(odev_dnd, text=i) for i in range(1, ochannels+1) ]],
        ipadding=1
    )
    odev_dnd.set_dnd_commit_callback(lambda *_: _update_mapping('output'))
    
    root.wm_deiconify()
    root.place_window_center()
    root.mainloop()

