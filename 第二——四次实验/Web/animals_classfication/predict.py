import torch
import timm
from PIL import Image
import json
from torch.autograd import Variable
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

with open(r"D:\python_program\Homework\Web\animals_classfication\dict.json", "r", encoding="utf-8") as f:
    data = json.load(f)
'''
    加载模型与参数
'''
#
# 加载模型
model = timm.create_model('resnet18', pretrained=True, num_classes=10).to(device)
model.load_state_dict(torch.load(r"D:\python_program\Homework\Web\animals_classfication\model_pth\pretrain_best_resnet.pth"))

# model = torch.load(r"C:\Users\13923\Desktop\best_resnet.pth")
# model = model.cuda()

transform = transforms.Compose([
    transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def run(img_path):
    img = Image.open(img_path)  # 打开图片
    # 图片转换为tensor
    img = transform(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()
    model.eval()
    # 不进行梯度更新
    with torch.no_grad():
        output_tensor = model(img)
        # 将输出通过softmax变为概率值
        output = torch.softmax(output_tensor, dim=1)
        # 输出可能性最大的那位
        pred_value, pred_index = torch.max(output, 1)
        pred_value = pred_value.detach().cpu().numpy()
        pred_index = pred_index.detach().cpu().numpy()
        # print("预测类别为： ", data[str(pred_index[0])], " 可能性为: ", pred_value[0] * 100, "%")
        return data[str(pred_index[0])], pred_value[0] * 100


if __name__ == '__main__':
    # img_path = r"D:\data\archive\animals_224\farfalla\farfalla62.jpg"
    # img_path = r"D:\data\archive\animals_224\cane\cane0.jpg"
    # img_path = r"D:\data\archive\animals_224\gatto\gatto0.jpg"
    img_path = r"D:\data\archive\animals_224\scoiattolo\scoiattolo20.jpg"
    # img_path = r"D:\data\archive\animals_224\cavallo\cavallo10.jpg"
    predict(img_path=img_path)