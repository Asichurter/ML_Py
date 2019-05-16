# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:02:47 2019

@author: 10904
"""

import tkinter as tk
import tkinter.messagebox as msgbox
import tkinter.font as tkf
from validate import get_validate_img
import MySQLdb as db

class LoginFrame():
    def __init__(self, root,valPath= r'C:/Users/10904/Desktop/数据库项目/image/valCache/val.png', color='#F0FFF0',
                 db='databasesys'):
        self.ValPath = valPath
        self.Root = root
        self.DB = db
        first_font = tkf.Font(family='Fixdsys', size=25, weight=tkf.BOLD)
        second_font = tkf.Font(family='Fixdsys', size=20, weight=tkf.BOLD)
        
        self.ValidateMsg = get_validate_img(valPath)
        self.ValidatePhoto = tk.PhotoImage(file=valPath, name='validatephoto')
        self.WarningPhoto = tk.PhotoImage(file=r'C:/Users/10904/Desktop/数据库项目/image/warning.png')
        
        self.Top = tk.Frame(root,bg='#F0FFF0')
        self.TopLabel = tk.Label(self.Top, text='****** 欢迎登陆 ******', 
                                 font=first_font, pady=10,bg='#F0FFF0')
        
        self.UsrFrame = tk.Frame(self.Top,bg='#F0FFF0')
        self.UsrLabel = tk.Label(self.UsrFrame, text='用户名 ', font=second_font,bg='#F0FFF0')
        self.UsrName = tk.StringVar(self.UsrFrame, name='username')
        self.UsrInput = tk.Entry(self.UsrFrame, textvariable=self.UsrName, 
                                 bd=4, width=30)
        
        self.PwFrame = tk.Frame(self.Top,bg='#F0FFF0')
        self.PwLabel = tk.Label(self.PwFrame, text='密码 ', font=second_font,bg='#F0FFF0')
        self.UsrPw = tk.StringVar(self.PwFrame, name='userpw')
        self.PwInput = tk.Entry(self.PwFrame, textvariable=self.UsrPw, show='*',
                                bd=4, width=30)
        
        self.FuncFrame = tk.Frame(self.Top,bg='#F0FFF0')
        self.Submit = tk.Button(self.FuncFrame, text='登陆', 
                                command=self.submit_login,
                                font=second_font, bd=2)
        
        self.ValFrame = tk.Frame(self.Top,bg='#F0FFF0')
        self.ValLabel = tk.Label(self.ValFrame, text='验证码 ', font=second_font,bg='#F0FFF0')
        self.ValVar = tk.StringVar(self.ValFrame, name='valvalue')
        self.ValInput = tk.Entry(self.ValFrame, textvariable=self.ValVar, bd=2, width=10)
        self.ValShow = tk.Label(self.ValFrame, image=self.ValidatePhoto)
        
        self.Top.pack(side=tk.TOP, pady=30)
        #self.TopLabel.pack(anchor=tk.N)
        self.TopLabel.grid(row=0,column=0, pady=30)
        
        #self.UsrFrame.pack(side=tk.TOP, pady=30)
        self.UsrFrame.grid(row=1,column=0, pady=20)
        #self.UsrLabel.pack(side=tk.LEFT, padx=10, pady=30)
        self.UsrLabel.grid(row=0,column=0,padx=10)
        #self.UsrInput.pack(side=tk.RIGHT, padx=10, pady=30)
        self.UsrInput.grid(row=0,column=1,padx=10)
        
        #self.PwFrame.pack(side=tk.TOP)
        self.PwFrame.grid(row=2,column=0, pady=20)
        #self.PwLabel.pack(side=tk.LEFT, padx=25, pady=0)
        self.PwLabel.grid(row=0,column=1,padx=10)
        #self.PwInput.pack(side=tk.RIGHT, padx=25, pady=0)
        self.PwInput.grid(row=0,column=2,padx=10)
        
        #self.ValFrame.pack(side=tk.TOP, pady=30)
        self.ValFrame.grid(row=3,column=0, pady=30)
        #self.ValLabel.pack(side=tk.LEFT, padx=40)
        self.ValLabel.grid(row=0,column=0,padx=10)
        #self.ValInput.pack(side=tk.LEFT, padx=10)
        self.ValInput.grid(row=0,column=1,padx=10)
        #self.ValShow.pack(side=tk.RIGHT, padx=10)
        self.ValShow.grid(row=0,column=2,padx=10)
        
        #self.FuncFrame.pack(side=tk.TOP, pady=10)
        self.FuncFrame.grid(row=4, column=0, pady=30)
        #self.Submit.pack(side=tk.TOP, padx=10, pady=10)
        self.Submit.grid(row=0,column=0,padx=10)
        
    def recreate_validate(self):
        self.ValidateMsg = get_validate_img()
        self.ValidatePhoto = tk.PhotoImage(self.ValPath)
        self.ValShow.update()
        #self.Root.update()
        
    def submit_login(self, host='localhost', port=3306):
        val = self.ValInput.get()
        if val!=self.ValidateMsg:
            msgbox.showerror(title='登陆失败', message='验证码错误!')
            self.ValInput.delete(0,tk.END)
            #self.recreate_validate()
            return False
        usr = self.UsrInput.get()
        pw = self.PwInput.get()
        con = db.connect(host=host, port=port, user='root',
                        passwd='', db=self.DB, charset='utf8')
        cur = con.cursor()
        sql = 'select PassWd from login where UserName=\'%s\''%usr
        cur.execute(sql)
        if cur.rowcount == 0:
            msg = '用户不存在！'
            msgbox.showerror(title='登陆失败', message=msg)
            self.UsrInput.delete(0,tk.END)
            self.PwInput.delete(0,tk.END)
            return False
        else:
            rs = cur.fetchone()
            if rs[0]!=pw:
                msg = '密码错误！'
                msgbox.showerror(title='登陆失败', message=msg)
                self.PwInput.delete(0,tk.END)
                return False
        self.Root.destroy()
        new = tk.Tk()
        new.mainloop()
    
    
    
    
    
    
    
    
    
    
        
        