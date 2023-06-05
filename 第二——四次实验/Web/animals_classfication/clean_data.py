import os
from PIL import Image

# 源目录
project_dir = os.path.dirname(os.path.abspath(r"D:\data\archive\raw-img"))
input = os.path.join(project_dir, 'raw-img')
print(os.path.join(project_dir, 'raw-img'))
# 输出目录
output = os.path.join(project_dir, 'animals_224')

def change():
    # 切换目录
    os.chdir(input)
    # 遍历目录下所有的文件
    for file_name in os.listdir(os.getcwd()):
        for id,image_name in enumerate(os.listdir(os.path.join(input,file_name))):
            #遍历文件夹中的图片
            im = Image.open(os.path.join(input,file_name, image_name))
            im = im.convert("RGB")
            im = im.resize((256, 256))
            #调整图片大小
            isExists = os.path.exists(os.path.join(output,file_name))
            if not isExists:
                #判断文件路径是否存在，不存在则生成文件
                os.makedirs(os.path.join(output,file_name))
                print(os.path.join(output,file_name) + ' 创建成功')
            im.save(os.path.join(output, file_name,file_name+ str(id) +".jpg"))

if __name__ == '__main__':
    change()