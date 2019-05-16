# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:02:47 2019

@author: 10904
"""

import tkinter as tk
import tkinter.messagebox as msgbox
import tkinter.font as tkf
from validate import get_validate_img
import MySQLdb as mysql

class LoginFrame():
    def __init__(self, root,valPath= r'D:/ML_Py/数据库项目/image/valCache/val.png', color='#F0FFF0',
                 db='databasesys'):
        self.Con = mysql.connect(host='localhost', port=3306, user='root',
                        passwd='123456', db=db, charset='utf8')
        self.Cur = self.Con.cursor()
        self.Fonts = []
        self.ValPath = valPath
        self.Root = root
        self.DB = db
        first_font = tkf.Font(family='Fixdsys', size=25, weight=tkf.BOLD)
        second_font = tkf.Font(family='Fixdsys', size=20, weight=tkf.BOLD)
        self.Fonts.append(first_font)
        self.Fonts.append(second_font)
        
        self.ValidateMsg = get_validate_img(valPath)
        self.ValidatePhoto = tk.PhotoImage(file=valPath, name='validatephoto')
        self.WarningPhoto = tk.PhotoImage(file=r'D:/ML_Py/数据库项目/image/warning.png')
        
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
        
        self.ValFrame = tk.Frame(self.Top,bg='#F0FFF0')
        self.ValLabel = tk.Label(self.ValFrame, text='验证码 ', font=second_font,bg='#F0FFF0')
        self.ValVar = tk.StringVar(self.ValFrame, name='valvalue')
        self.ValInput = tk.Entry(self.ValFrame, textvariable=self.ValVar, bd=2, width=10)
        self.ValShow = tk.Label(self.ValFrame, image=self.ValidatePhoto)
        
        self.FuncFrame = tk.Frame(self.Top,bg='#F0FFF0')
        self.Submit = tk.Button(self.FuncFrame, text='登陆', 
                                command=self.submit_login,
                                font=second_font, bd=2)
        self.Register = tk.Button(self.FuncFrame, text='注册', font=second_font,
                                  command=self.register)
        self.ForgetPw = tk.Button(self.FuncFrame, text='忘记密码？',
                                  font=second_font, command=self.forget_passwd)

        
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
        self.Register.grid(row=0, column=1, padx=10)
        self.ForgetPw.grid(row=0, column=2, padx=10)
        
        
    def recreate_validate(self):
        self.ValidateMsg = get_validate_img()
        self.ValidatePhoto = tk.PhotoImage(self.ValPath)
        self.ValShow.update()
        #self.Root.update()
        
    def submit_login(self):
        val = self.ValInput.get()
        if val!=self.ValidateMsg:
            msgbox.showerror(title='登陆失败', message='验证码错误!')
            self.ValInput.delete(0,tk.END)
            #self.recreate_validate()
            return False
        usr = self.UsrInput.get()
        pw = self.PwInput.get()
        sql = 'select PassWd from login where UserName=\'%s\''%usr
        self.Cur.execute(sql)
        if self.Cur.rowcount == 0:
            msg = '用户不存在！'
            msgbox.showerror(title='登陆失败', message=msg)
            self.UsrInput.delete(0,tk.END)
            self.PwInput.delete(0,tk.END)
            return False
        else:
            rs = self.Cur.fetchone()
            if rs[0]!=pw:
                msg = '密码错误！'
                msgbox.showerror(title='登陆失败', message=msg)
                self.PwInput.delete(0,tk.END)
                return False
        self.Root.destroy()
        new = tk.Tk()
        new.mainloop()   
        
    def forget_passwd(self):
        top = tk.Toplevel(self.Top)
        top.title('找回密码')
        top.geometry('600x300')
        
        frame1 = tk.Frame(top)
        name_label = tk.Label(frame1, text='用户名 ', font=self.Fonts[1])
        name = tk.StringVar(frame1, name='name')
        name_input = tk.Entry(frame1, textvariable=name, bd=2, width=15)
        frame1.pack(side=tk.TOP, pady=30)
        name_label.grid(row=0, column=0, padx=10)
        name_input.grid(row=0, column=1, padx=10)
        
        frame2 = tk.Frame(top)
        email_label = tk.Label(frame2, text='注册邮箱 ', font=self.Fonts[1])
        email = tk.StringVar(frame2, name='email')
        email_input = tk.Entry(frame2, textvariable=email, bd=2, width=20)
        frame2.pack(side=tk.TOP, pady=20)
        email_label.grid(row=0, column=0, padx=10)
        email_input.grid(row=0, column=1, padx=10)
        
        button = tk.Button(top, text='下一步', 
                           command=lambda:self.check_email(name_input, email_input, top),
                           font=self.Fonts[1])
        button.pack(side=tk.TOP, pady=20)
        
    def check_email(self, name, email, root):
        sql = 'select PassWd from login where UserName=\'%s\' and Email=\'%s\''%(name.get(),email.get())
        self.Cur.execute(sql)
        if self.Cur.rowcount == 1:
            rs = self.Cur.fetchone()
            msgbox.showinfo(title='找回密码成功', message='用户%s的密码为: %s'%(name.get(), rs[0]))
            name.delete(0,tk.END)
            email.delete(0,tk.END)
            root.destroy()
        else:
            msgbox.showerror(title='找回密码失败', message='用户名和邮箱不匹配！')
            
    def register(self):
        top = tk.Toplevel(self.Root)
        top.geometry('600x600')
        top.title('注册新用户')
        
        label = tk.Label(top, text='请输入以下信息', font=self.Fonts[0])
        label.pack(side=tk.TOP, pady=30)
        
        id_f = tk.Frame(top)
        id_label = tk.Label(id_f, text='ID(5位) ', font=self.Fonts[1])
        Id = tk.StringVar(id_f)
        id_input = tk.Entry(id_f, textvariable=Id, bd=2, width=10)
        id_f.pack(side=tk.TOP, pady=20)
        id_label.grid(row=0, column=0, padx=10)
        id_input.grid(row=0, column=1, padx=10)
        
        name_f = tk.Frame(top)
        name_label = tk.Label(name_f, text='姓名 ', font=self.Fonts[1])    
        name = tk.StringVar(name_f)
        name_input = tk.Entry(name_f, textvariable=name, bd=2, width=20)
        name_f.pack(side=tk.TOP, pady=20)
        name_label.grid(row=0, column=0, padx=10)
        name_input.grid(row=0, column=1, padx=10)
        
        pw_f = tk.Frame(top)
        pw_label = tk.Label(pw_f, text='密码 ', font=self.Fonts[1])    
        pw = tk.StringVar(pw_f)
        pw_input = tk.Entry(pw_f, textvariable=pw, bd=2, width=20)
        pw_f.pack(side=tk.TOP, pady=20)
        pw_label.grid(row=0, column=0, padx=10)
        pw_input.grid(row=0, column=1, padx=10)
        
        prior_f = tk.Frame(top)
        prior = tk.StringVar(prior_f, value='user')
        user_button = tk.Radiobutton(prior_f, text='普通用户', variable=prior, value='user', font=self.Fonts[1])
        admin_button = tk.Radiobutton(prior_f, text='管理员', variable=prior, value='admin', font=self.Fonts[1])
        prior_f.pack(side=tk.TOP, pady=20)
        user_button.grid(row=0, column=0, padx=10)
        admin_button.grid(row=0, column=1, padx=10)
        
        em_f = tk.Frame(top)
        em_label = tk.Label(em_f, text='邮箱 ', font=self.Fonts[1])
        email = tk.StringVar(em_f)
        em_input = tk.Entry(em_f, textvariable=email, bd=2, width=25)
        em_f.pack(side=tk.TOP, pady=20)
        em_label.grid(row=0, column=0, padx=10)
        em_input.grid(row=0, column=1, padx=10)
        
        button = tk.Button(top, text='注册', font=self.Fonts[1],
                           command=lambda: 
                               self.submit_register(id_input,name_input,pw_input,prior,em_input))
        button.pack(side=tk.TOP, pady=20)
        
    def submit_register(self, id, name, pw, prior, email):
        sql = 'insert into login values(\'%s\', \'%s\',\'%s\', \'%s\', \'%s\')'%\
                (id.get(),name.get(),pw.get(),prior.get(),email.get())
                
        duplicate_id_sql = 'select ID from login'
        duplicate_email_sql = 'select Email from login'
        try:
            self.Cur.execute(duplicate_id_sql)
            if self.Cur.rowcount != 0:
                rs = self.Cur.fetchall()
                for row in rs:
                    if row[0]==id.get():
                        msgbox.showerror(title='注册失败！', message='ID重复！')
                        raise RuntimeError
            self.Cur.execute(duplicate_email_sql)
            if self.Cur.rowcount != 0:
                rs = self.Cur.fetchall()
                for row in rs:
                    if row[0]==email.get():
                        msgbox.showerror(title='注册失败！', message='Email重复！')
                        raise RuntimeError
            self.Cur.execute(sql)
            self.Con.commit()
            msgbox.showinfo(title='注册成功', message='注册成功!')
        except RuntimeError:
            pass
        except mysql.MySQLError:
            msgbox.showerror(title='注册失败！', message='数据库操作失败！')
            self.Con.rollback()
        finally:
            id.delete(0, tk.END)
            name.delete(0, tk.END)
            pw.delete(0, tk.END)
            email.delete(0, tk.END)
            
            '''
        self.Cur.execute(sql)
        self.Con.commit()
        msgbox.showinfo(title='注册成功', message='注册成功!')'''
        
    '''
class MainFrame():
    def __init__(self, root):
        '''
    
    
    
    
    
    
    
    
        
        