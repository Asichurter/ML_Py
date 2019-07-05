# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:09:27 2019

@author: 唐郅杰
"""

from frame import LoginFrame
import tkinter as tk
import tkinter.ttk as ttk

def main():
    root = tk.Tk()
    root.config(bg = '#F0FFF0')
    root.geometry('600x600')
    root.resizable(width=False, height=False)
    root.title('用户登录')
    login = LoginFrame(root)
    root.mainloop()

if __name__ == '__main__':
    main()