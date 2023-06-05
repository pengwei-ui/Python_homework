#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Roc-xb
@desc: 将图片base流保存为图片文件
"""
import base64

if __name__ == '__main__':
    source_img = ""
    data = source_img.split(',')[1]
    image_data = base64.b64decode(data)
    with open('captcha.png', 'wb') as f:
        f.write(image_data)
        print("图片文件保存成功")