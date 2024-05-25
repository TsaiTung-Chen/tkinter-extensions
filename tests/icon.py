from base64 import b64encode
import ttkbootstrap as ttk

ICON_PATH = r"icon.png"

root = ttk.Window()
with open(ICON_PATH, 'rb') as file:
    data = b64encode(file.read())
img = ttk.PhotoImage(data=data)
root.wm_iconphoto(True, img)
root.mainloop()

