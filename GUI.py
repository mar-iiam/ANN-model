from doctest import master
import main as mn


from tkinter import filedialog

from PIL import ImageTk
import PIL.Image
import sqlite3
from tkinter import Tk, Button, Canvas , Label ,Frame ,Entry
from PIL import Image, ImageFont

mainWin = Tk()
myLabel = Label(mainWin, text="Object detection Model")
myLabel.pack()


def click():
    x = Label(mainWin, text="Object detection Model")
    x.pack()


button1 = Button(mainWin, text="buttion1", command=click,fg="red")
button1.pack()

# to locate  label.grid(row,col)

frame = Frame(mainWin, bg='#45aaf2')

lbl_pic_path = Label(frame, text='Image Path:', padx=25, pady=25,font=('verdana',16), bg='#45aaf2')

lbl_show_pic = Label(frame, bg='#45aaf2')
entry_pic_path =Entry(frame, font=('verdana',16))
btn_browse = Button(frame, text='Select Image',bg='grey', fg='#ffffff',
                       font=('verdana',16))

# mn.predict_output(entry_pic_path)
def selectPic():
    global img
    filename = filedialog.askopenfilename(initialdir="/images", title="Select Image",
                           filetypes=(("png images","*.png"),("jpg images","*.jpg")))
    img = Image.open(filename)
    img = img.resize((200,200), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    lbl_show_pic['image'] = img
    entry_pic_path.insert(0, filename)
    xw=mn.predict_output(filename)
    x = Label(mainWin, text=xw)
    x.pack()
btn_browse['command'] = selectPic

frame.pack()

lbl_pic_path.grid(row=0, column=0)
entry_pic_path.grid(row=0, column=1, padx=(0,20))
lbl_show_pic.grid(row=1, column=0, columnspan="2")
btn_browse.grid(row=2, column=0, columnspan="2", padx=10, pady=10)

mainWin.mainloop()