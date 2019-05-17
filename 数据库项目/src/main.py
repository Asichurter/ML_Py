# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:09:27 2019

@author: 10904
"""

from frame import LoginFrame
import tkinter as tk
import tkinter.ttk as ttk

def main():
    root = tk.Tk()
    #im = tk.PhotoImage(file=r'C:/Users/10904/Desktop/数据库项目/image/background.png')
    #background = tk.Canvas(root)
    #background.create_image(0,0, image=im)
    #background.pack()
    root.config(bg = '#F0FFF0')
    root.geometry('600x600')
    root.resizable(width=False, height=False)
    root.title('用户登录')
    login = LoginFrame(root)
    root.mainloop()

if __name__ == '__main__':
    main()