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

def get_font(size, bold=True):
    if bold:
        isBold = tkf.BOLD
    else:
        isBold = tkf.NORMAL
    return tkf.Font(family='Fixdsys', size=size, weight=isBold)

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
        
        self.PriorFrame = tk.Frame(self.Top, bg='#F0FFF0')
        self.Prior = tk.StringVar(self.PriorFrame, name='prior', value='user')
        self.UserPrior = tk.Radiobutton(self.PriorFrame, text='用户登录', variable=self.Prior,
                                        value='user', font=self.Fonts[1], bg='#F0FFF0')
        self.AdminPrior = tk.Radiobutton(self.PriorFrame, text='管理员登录', variable=self.Prior,
                                        value='admin', font=self.Fonts[1], bg='#F0FFF0')
        
        
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
        
        self.PriorFrame.grid(row=4, column=0, pady=20)
        self.UserPrior.grid(row=0, column=0, ipadx=10, pady=10, sticky=tk.E+tk.W)
        self.AdminPrior.grid(row=0, column=1, ipadx=10,pady=10, sticky=tk.E+tk.W)     
        
        #self.FuncFrame.pack(side=tk.TOP, pady=10)
        self.FuncFrame.grid(row=5, column=0, pady=10)
        #self.Submit.pack(side=tk.TOP, padx=10, pady=10)
        self.Submit.grid(row=1,column=0,padx=20)
        self.Register.grid(row=1, column=1, padx=0)
        self.ForgetPw.grid(row=1, column=2, padx=20)
        
        
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
        sql = 'select * from login where UserName=\'%s\''%usr
        self.Cur.execute(sql)
        rs = self.Cur.fetchone()
        if self.Cur.rowcount == 0:
            msg = '用户不存在！'
            msgbox.showerror(title='登陆失败', message=msg)
            self.UsrInput.delete(0,tk.END)
            self.PwInput.delete(0,tk.END)
            return False
        elif rs[2]!=pw:
            msg = '密码错误！'
            print(rs)
            msgbox.showerror(title='登陆失败', message=msg)
            self.PwInput.delete(0,tk.END)
            return False
        else:
            if rs[3] == 'user' and self.Prior.get()=='admin':
                msgbox.showerror(title='登录失败', message='权限不够！')
                return False
        self.Root.destroy()
        new = tk.Tk()
        assert len(rs)==5, '数据库中出现重叠!'
        kwargs = {}
        #由于管理员可以用用户身份登录，因此权限不应该从数据库中得到
        kwargs['prior']=self.Prior.get()
        kwargs['id'] = rs[0]
        kwargs['name'] = rs[1]
        kwargs['pw'] = rs[2]
        kwargs['email'] = rs[4]
        kwargs['con'] = self.Con
        kwargs['cursor'] = self.Cur
        login_success = MainFrame(new, kwargs)
        new.mainloop()   
        
    def forget_passwd(self):
        top = tk.Toplevel(self.Top)
        top.title('找回密码')
        top.geometry('600x300')
        top.resizable(width=False,height=False)
        
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
        top.resizable(width=False,height=False)
        
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
        
        '''
        注册时要插入信息
        '''
        
    def submit_register(self, id, name, pw, prior, email):
        sql = 'insert into login values(\'%s\', \'%s\',\'%s\', \'%s\', \'%s\')'%\
                (id.get(),name.get(),pw.get(),prior.get(),email.get())
        try:        
            if id.get()=='' or name.get()=='' or pw.get()=='' or prior.get()=='' or email.get()=='':
                msgbox.showerror(title='注册失败', message='所有项均为必填项！')
                return
            if len(id.get())!=5:
                msgbox.showerror(title='注册失败', message='ID必须为5位!')
                id.delete(0, tk.END)
                return
            duplicate_id_sql = 'select ID from login'
            duplicate_email_sql = 'select Email from login'
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
            
            insert_sql = 'insert into studentinfo values(\'%s\',NULL,NULL,NULL,NULL,NULL)'%id.get()
            self.Cur.execute(insert_sql)
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
        
class MainFrame():
    def __init__(self, root, info):
        self.Info = info
        self.Top = root
        self.Top.title('菜单')
        self.Top.resizable(width=False,height=False)
        
        if info['prior']=='user':
            root.geometry('600x300')
        else:
            root.geometry('600x400')
        
        self.LoginMsg = tk.Label(self.Top, text='登陆成功', font=get_font(25), fg='red')
        #self.LoginMsg.grid(row=0, column=0, pady=10)
        self.LoginMsg.pack(side=tk.TOP, pady=10)
        self.Msg = tk.Label(self.Top, text='您好 %s %s %s'%(info['prior'],info['name'],info['id']),
                            font=get_font(20))
        #self.Msg.grid(row=1, column=0, pady=10)
        self.Msg.pack(side=tk.TOP, pady=10)
        
        self.OptionsFrame = tk.Frame(self.Top)
        #self.OptionsFrame.grid(row=3, column=0, pady=10)
        self.OptionsFrame.pack(side=tk.TOP, pady=10)
        
        if info['prior'] == 'user':
            self.InfoButton = tk.Button(self.OptionsFrame, text='查看我的信息', font=get_font(20),
                                        command=self.view_info)
            self.InfoButton.grid(row=0, column=0, pady=10)
        else:
            self.ModInfoButton = tk.Button(self.OptionsFrame, text='修改个人信息', 
                                           font=get_font(20), command=self.edit_info)
            self.ModInfoButton.grid(row=0, column=0, pady=10)
        
        if info['prior'] == 'user':
            self.GradeButton = tk.Button(self.OptionsFrame, text='查看我的课程成绩', font=get_font(20),
                                         command=self.view_grades)
            self.GradeButton.grid(row=1, column=0, pady=10)
        else:
            self.ModGradeButton = tk.Button(self.OptionsFrame, text='修改课程成绩', 
                                            font=get_font(20), command=self.edit_grade)
            self.ModGradeButton.grid(row=1, column=0, pady=10)
            self.InsGradeButton = tk.Button(self.OptionsFrame, text='添加课程成绩', 
                                            font=get_font(20), command=self.create_grade)
            self.InsGradeButton.grid(row=2, column=0, pady=10)
            
            
    def view_info(self):
        sql = 'select * from studentInfo where ID=%s'%self.Info['id']
        self.Info['cursor'].execute(sql)
        assert self.Info['cursor'].rowcount == 1, '查到的数据项不止一行！'
        rs = self.Info['cursor'].fetchone()
        
        top = tk.Toplevel(self.Top)
        top.geometry('600x600')
        top.title('我的信息')
        top.resizable(width=False,height=False)
        
        id_line = self.make_info_line(top, 0, 'ID ', rs[0])
        name_line = self.make_info_line(top, 1, '姓名', self.Info['name'])
        sex_line = self.make_info_line(top, 2, '性别 ', rs[3])
        dep_line = self.make_info_line(top, 3, '学院 ', rs[1])
        maj_line = self.make_info_line(top, 4, '专业 ', rs[2])
        add_line = self.make_info_line(top, 5, '地址 ', rs[4])
        tel_line = self.make_info_line(top, 6, '电话 ', rs[5])
        
    def view_grades(self):
        sql = 'select Class,Grade from takes where ID=%s'%self.Info['id']
        self.Info['cursor'].execute(sql)
        
        top = tk.Toplevel(self.Top)
        top.title('我的课程成绩')
        top.geometry('600x300')
        top.resizable(width=False, height=False)
        
        msg = tk.Label(top, text='%s 的课程成绩如下'%self.Info['name'], 
                       font=get_font(25), fg='red')
        msg.pack(side=tk.TOP)
        
        '''
        textframe = tk.Frame(top)
        textframe.pack(side=tk.TOP)
        text = tk.Listbox(textframe, width=60, height=10, font=get_font(20))
        for i in range(self.Info['cursor'].rowcount):
            rs = self.Info['cursor'].fetchone()
            text.insert(tk.END, '%s       %s\n'%(rs[0],rs[1]))
        
        scrollbar = tk.Scrollbar(textframe)
        
        scrollbar.config(command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, pady=10)
        '''
        
        showframe = tk.Frame(top)
        showframe.pack(side=tk.TOP)
               
        tree = tk.ttk.Treeview(showframe, columns=['科目', '分数'],
                               show='headings')
        tree.column('科目', width=200, anchor='center')
        tree.column('分数', width=200, anchor='center')
        tree.heading('科目', text='科目')
        tree.heading('分数', text='分数')
        
        scbar = tk.Scrollbar(showframe)
        scbar.config(command=tree.yview)
        
        for i in range(self.Info['cursor'].rowcount):
            rs = self.Info['cursor'].fetchone()
            tree.insert('', i, values=(rs[0],rs[1]))
        '''    
        def click_test(event):
            for item in tree.selection():
                content = tree.item(item, "values")
                print(content)
        tree.bind('<Double-Button-1>', click_test)
        '''
        scbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH)
        
        
    def edit_info(self):
        top = tk.Toplevel(self.Top)
        msg = tk.Label(top, text='请双击要修改的行', font=get_font(20), fg='red')
        msg.pack(side=tk.TOP, pady=20)
        top.geometry('800x600')
        top.title('修改个人信息')
        top.resizable(width=False, height=False)
        
        sql = 'select * from login natural join studentinfo'
        self.Info['cursor'].execute(sql)
        
        frame = tk.Frame(top)
        frame.pack(side=tk.TOP, pady=20)
        tree = tk.ttk.Treeview(frame, columns=['ID', '姓名', '性别', '学院', '专业', '地址', '电话'],
                               show='headings', height=200)
        tree.column('ID', width=50, anchor='center')
        tree.column('姓名', width=100, anchor='center')
        tree.column('性别', width=50, anchor='center')
        tree.column('学院', width=100, anchor='center')
        tree.column('专业', width=100, anchor='center')
        tree.column('地址', width=200, anchor='center')
        tree.column('电话', width=100, anchor='center')
        tree.heading('ID', text='ID')
        tree.heading('姓名', text='姓名')
        tree.heading('性别', text='性别')
        tree.heading('学院', text='学院')
        tree.heading('专业', text='专业')
        tree.heading('地址', text='地址')
        tree.heading('电话', text='电话')
        for i in range(self.Info['cursor'].rowcount):
            rs = self.Info['cursor'].fetchone()
            tree.insert('', i, values=(rs[0],rs[1],rs[7],rs[5],rs[6],rs[8],rs[9]))
                
        def edit_row(event):
            selection = tree.selection()
            if len(selection)==1:
                id = tree.item(selection[0], 'values')[0]
                self.edit_info_column(id)
        tree.bind('<Double-Button-1>', edit_row)
        tree.pack(side=tk.LEFT, fill=tk.BOTH)
        
        scbar = tk.Scrollbar(frame)
        scbar.config(command=tree.yview)
        scbar.pack(side=tk.RIGHT, fill=tk.Y)
        
                
    def edit_info_column(self, id):
        top = tk.Toplevel(self.Top)
        top.geometry('600x250')
        top.title('修改个人信息')
        
        msg = tk.Label(top, text='请选择要修改的项', font=get_font(20), fg='red')
        msg.pack(side=tk.TOP, pady=20)
        
        frame = tk.Frame(top)
        frame.pack(side=tk.TOP, pady=20)
        option = tk.StringVar(frame, value='ID')
        values = {'ID':'ID', 
                  '姓名':'UserName', 
                  '性别':'Sex', 
                  '学院':'Department',
                  '专业':'Majority', 
                  '地址':'Address', 
                  '电话':'Telephone'}
        button_list = []
        keys = values.keys()
        for i,key in enumerate(keys):
            button = tk.Radiobutton(frame, text=key, 
                                              variable=option, value=key, font=get_font(20))
            button.grid(row=0, column=i)
            button_list.append(button)
        
        ok_button = tk.Button(top, text='确定', font=get_font(20),
                              command=lambda:self.edit_info_values(top,
                                                                   id, 
                                                                   option.get(),
                                                                   values))
        ok_button.pack(side=tk.TOP, pady=20)
    
    def edit_info_values(self, src, id, val, val_dict):
        #先销毁上一层界面
        src.destroy()
        
        top = tk.Toplevel(self.Top)
        top.geometry('600x250')
        top.title('修改个人信息')
        
        label = tk.Label(top, text='请输入新的%s'%val, font=get_font(20), fg='red')
        label.pack(side=tk.TOP, pady=20)
        
        var = tk.StringVar(top)
        entry = tk.Entry(top, textvariable=var, bd=2, width=80)
        entry.pack(side=tk.TOP, pady=20)
        
        db_map = {'ID':'login', 
                        '姓名':'login', 
                        '性别':'studentinfo', 
                        '学院':'studentinfo',
                        '专业':'studentinfo', 
                        '地址':'studentinfo', 
                        '电话':'studentinfo'
                        }
        
        def update_info(database):
            sql = 'update %s set %s=\'%s\' where ID=\'%s\''%(database, val_dict[val], entry.get(), id)
            try:
                self.Info['cursor'].execute(sql)
                self.Info['con'].commit()
                msgbox.showinfo(title='修改个人信息', message='修改成功！')
            except:
                msgbox.showerror(title='修改个人信息', message='修改失败！数据库操作失败！')
                self.Info['con'].rollback()
            finally:
                #销毁上一层界面
                top.destroy()
        
        button = tk.Button(top, text='确定', font=get_font(20),
                           command=lambda:update_info(db_map[val]))
        button.pack(side=tk.TOP, pady=20)
        
    def edit_grade(self):
        top = tk.Toplevel(self.Top)
        top.geometry('600x400')
        top.title('修改课程成绩')
        top.resizable(width=False, height=False)
        
        msg = tk.Label(top, text='请双击要修改的项', font=get_font(20), fg='red')
        msg.pack(side=tk.TOP, pady=20)
        
        sql = 'select * from takes'
        self.Info['cursor'].execute(sql)
        
        frame = tk.Frame(top)
        frame.pack(side=tk.TOP, pady=20)
        tree = tk.ttk.Treeview(frame, 
                               columns=['ID', '课程', '分数'],
                               show='headings',
                               height=100)
        tree.column('ID', width=100, anchor='center')
        tree.column('课程', width=150, anchor='center')
        tree.column('分数', width=100, anchor='center')
        tree.heading('ID', text='ID')
        tree.heading('课程', text='课程')
        tree.heading('分数', text='分数')
        for i in range(self.Info['cursor'].rowcount):
            rs = self.Info['cursor'].fetchone()
            tree.insert('', i, values=(rs[0],rs[1],rs[2]))
                
        def edit_grade_row(event):
            selection = tree.selection()
            if len(selection)==1:
                ID = tree.item(selection[0], 'values')[0]
                Class = tree.item(selection[0], 'values')[1]
                self.edit_grade_value(ID, Class)
        tree.bind('<Double-Button-1>', edit_grade_row)
        tree.pack(side=tk.LEFT, fill=tk.BOTH)
        
        scbar = tk.Scrollbar(frame)
        scbar.config(command=tree.yview)
        scbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def edit_grade_value(self, ID, Class):
        top = tk.Toplevel(self.Top)
        top.geometry('600x300')
        top.resizable(width=False, height=False)
        top.title('修改课程成绩')
        
        msg = tk.Label(top, text='请输入新的成绩(0~100)', font=get_font(20), fg='red')
        msg.pack(side=tk.TOP, pady=20)
        
        grade = tk.StringVar(top)
        entry = tk.Entry(top, textvariable=grade, bd=2, width=30)
        entry.pack(side=tk.TOP, pady=20)
        
        def update_grade():
            g = int(grade.get())
            if g>100 or g<0:
                msgbox.showerror(title='修改课程成绩', message='请输入0~100内的数！')
                return
            G = str(g)
            top.destroy()
            sql = 'update takes set Grade=%s where ID=\'%s\' and Class=\'%s\''%(G,ID,Class)
            try:
                self.Info['cursor'].execute(sql)
                self.Info['con'].commit()
                msgbox.showinfo(title='修改课程成绩', message='修改成功！')
            except:
                msgbox.showerror(title='修改课程成绩', message='修改失败！数据库操作错误！')
                self.Info['con'].rollback()
        
        button = tk.Button(top, text='确定', font=get_font(20),
                           command=update_grade)
        button.pack(side=tk.TOP, pady=20)
        
    def create_grade(self):
        top = tk.Toplevel(self.Top)
        top.geometry('600x600')
        top.resizable(width=False, height=False)
        top.title('创建新的成绩')
        
        sql = 'select * from login natural join studentinfo'
        self.Info['cursor'].execute(sql)
        
        frame = tk.Frame(top)
        frame.pack(side=tk.TOP, pady=20)
        tree = tk.ttk.Treeview(frame, columns=['ID', '姓名', '性别', '学院', '专业', '地址', '电话'],
                               show='headings', height=200)
        tree.column('ID', width=50, anchor='center')
        tree.column('姓名', width=100, anchor='center')
        tree.column('性别', width=50, anchor='center')
        tree.column('学院', width=100, anchor='center')
        tree.column('专业', width=100, anchor='center')
        tree.column('地址', width=200, anchor='center')
        tree.column('电话', width=100, anchor='center')
        tree.heading('ID', text='ID')
        tree.heading('姓名', text='姓名')
        tree.heading('性别', text='性别')
        tree.heading('学院', text='学院')
        tree.heading('专业', text='专业')
        tree.heading('地址', text='地址')
        tree.heading('电话', text='电话')
        for i in range(self.Info['cursor'].rowcount):
            rs = self.Info['cursor'].fetchone()
            tree.insert('', i, values=(rs[0],rs[1],rs[7],rs[5],rs[6],rs[8],rs[9]))
                
        def create_grade_row(event):
            selection = tree.selection()
            if len(selection)==1:
                id = tree.item(selection[0], 'values')[0]
                self.create_grade_value(id)
        tree.bind('<Double-Button-1>', create_grade_row)
        tree.pack(side=tk.LEFT, fill=tk.BOTH)
        
        scbar = tk.Scrollbar(frame)
        scbar.config(command=tree.yview)
        scbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_grade_value(self, ID):
        top = tk.Toplevel(self.Top)
        top.geometry('600x400')
        top.resizable(width=False, height=False)
        top.title('创建新的成绩')
        
        msg = tk.Label(top, text='请输入以下信息', font=get_font(25), fg='red')
        msg.pack(side=tk.TOP, pady=20)
        
        cla_frame = tk.Frame(top)
        cla_frame.pack(side=tk.TOP, pady=20)
        cla_label = tk.Label(cla_frame, text='课程名 ', font=get_font(20))
        cla_val = tk.StringVar(cla_frame)
        cla_entry = tk.Entry(cla_frame, textvariable=cla_val, bd=2, width=60)
        cla_label.grid(row=0, column=0, padx=10)
        cla_entry.grid(row=0, column=1, padx=10)
        
        gra_frame = tk.Frame(top)
        gra_frame.pack(side=tk.TOP, pady=20)
        gra_label = tk.Label(gra_frame, text='成绩 ', font=get_font(20))
        gra_val = tk.StringVar(gra_frame)
        gra_entry = tk.Entry(gra_frame, textvariable=gra_val, bd=2, width=30)
        gra_label.grid(row=0, column=0, padx=10)
        gra_entry.grid(row=0, column=1, padx=10)
        
        def create_grade_upload():
            check_sql = 'select * from takes where ID=\'%s\' and Class=\'%s\''%(ID, cla_val.get())
            self.Info['cursor'].execute(check_sql)
            if self.Info['cursor'].rowcount !=0 :
                msgbox.showerror(title='创建新的成绩', message='创建的成绩对应的ID和课程已存在！')
                return
            g = int(gra_val.get())
            if g<0 or g>100:
                msgbox.showerror(title='创建新的成绩',message='成绩的范围不合要求！')
                return
            top.destroy()
            G = str(g)
            sql = 'insert into takes values(\'%s\',\'%s\',%s)'%(ID, cla_val.get(), G)
            try:
                self.Info['cursor'].execute(sql)
                self.Info['con'].commit()
                msgbox.showinfo(title='创建新的成绩', message='创建成功！')
            except:
                msgbox.showerror(title='创建新的成绩', message='创建失败！数据库操作失败！')
                self.Info['con'].rollback()
        
        button = tk.Button(top, text='确定', font=get_font(20), command=create_grade_upload)
        button.pack(side=tk.TOP, pady=20)

    
    def make_info_line(self, top, row, label, info, size=20):
        frame = tk.Frame(top)
        frame.pack(side=tk.TOP, pady=30)
        
        label_ = tk.Label(frame, text=label, font=get_font(size))
        label_.grid(row=0, column=0)
        
        info_ = tk.Label(frame, text=info, font=get_font(size))
        info_.grid(row=0, column=1, padx=10)
        
        return frame
    
    
    
    
    
    
    
    
    
        
        