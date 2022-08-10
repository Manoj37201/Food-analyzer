import sys
import os
import tkinter
import tkinter.messagebox
window=tkinter.Tk()
window.title("Choose what to classify")
window.geometry("230x200")

title = tkinter.Label(text="Choose what to analyze", background = "cyan", fg="Red", font=("times", 15, "bold" ))
title.grid(column=0, row=0, padx=10, pady=10)

def testfruit():
    os.system('python testing_fruit.py')

def testdesi():
    os.system('python testing_indian.py')

B=tkinter.Button(window,text="Fruits or veggies",command= testfruit)
B.grid(column=0, row=2, padx=10, pady=10)

c=tkinter.Button(window,text="Desi food",command= testdesi)
c.grid(column=0, row=4, padx=10, pady=10)

window.mainloop()