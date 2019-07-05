# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:12:51 2019

@author: 唐郅杰
"""

from PIL import Image, ImageDraw, ImageFont
import random as rd

def get_validate_img(path=r'D:/ML_Py/数据库项目/image/valCache/val.png'):
    #背景颜色范围
    color = rd.randint(120, 255)
    img = Image.new(mode='RGB', size=(120,30), color=(color,color, color))
    drawer = ImageDraw.Draw(img, mode='RGB')
    font = ImageFont.truetype('C:/Windows/Fonts/Arial.ttf', 21)
    candidates = [chr(65+i) for i in range(26)]
    candidates += [str(i) for i in range(9)]
    #print(candidates)
    msg = str()
    
    for i in range(4):
        color = (rd.randint(0,255),rd.randint(0,255),rd.randint(0,255))
        char = rd.choice(candidates)[0]
        msg += char
        drawer.text([i*30,0], char, color, font=font)
    img.save(path, 'PNG')
    return msg
    
if __name__ == '__main__':
    print(get_validate_img())
        
    