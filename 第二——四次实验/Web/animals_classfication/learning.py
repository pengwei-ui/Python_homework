import numpy as np
import timm
import timm.scheduler
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import create_dataset
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Train_transforms = torchvision.transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    # 标准化图像的每个通道
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

Test_transforms = torchvision.transforms.Compose([
    transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class Learn(object):
    def __init__(self, train_csv_path, test_csv_path, epochs, lr, batch_size, weight_decay, mode):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.mode = mode
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        pass

    def train(self):
        write = SummaryWriter('logs')
        trainloader = torch.utils.data.DataLoader(
            create_dataset.Train_Valid_Dataset(self.train_csv_path, transform=Train_transforms),
            batch_size=64, num_workers=1, drop_last=True, shuffle=True)
        testloader = torch.utils.data.DataLoader(
            create_dataset.Train_Valid_Dataset(self.test_csv_path, transform=Test_transforms),
            batch_size=32, num_workers=1, drop_last=True)

        model = timm.create_model('resnet18', pretrained=True, num_classes=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                                     t_initial=self.epochs,#训练总数
                                                     lr_min=0.0003,#余弦退火最低学习率
                                                     warmup_t=2,#学习预热阶段epoch数量
                                                     warmup_lr_init=0.001#学习率预热阶段的学习率起始值,最优为0.001
                                                     )
        total_train_step, total_test_step, best_test_acc = 0, 0, 0

        for epoch in range(self.epochs):
            scheduler.step(epoch)
            write.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            # 模型训练开始
            model.train()
            print(f'Starting epoch {epoch + 1}')
            train_losses = []
            for i, batch in enumerate(trainloader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_pre = model(x)
                loss = loss_fn(y_pre, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                step = epoch * len(trainloader) + i
                if step % 100 == 0:
                    write.add_scalar("train_loss", loss, step)
            train_loss = np.sum(train_losses) / len(train_losses)
            print(f"The {epoch + 1} train_loss: {train_loss:.3f}")
            if (epoch + 1) % 2 == 0:
                model.eval()
                print('Training process has finished. Saving trained model.')
                print('Starting validation')
                test_total_accuracy = 0
                test_data_size = 0
                with torch.no_grad():
                    for batch in tqdm(testloader):
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        y_pre = model(x)
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        test_total_accuracy += ((y_pre.argmax(dim=1) == y).sum().item())
                        test_data_size += len(y)
                    test_acc = test_total_accuracy / test_data_size
                    write.add_scalars("acc", {"test_acc": test_acc}, epoch)
                    temp_acc = best_test_acc
                    best_test_acc = max(test_acc, best_test_acc)
                    print(f"The final test_acc:{test_acc:.3f}")
                    print('--------------------------------------')
                    if best_test_acc > temp_acc:
                        save_path = f"D:/machine_learning/animals_classfication/model_pth/pretrain_best_resnet.pth"
                        torch.save(model.state_dict(), save_path)
                        print("Improve!!!!")
                    else:
                        print(f"Not improve!!,The best test_acc:{best_test_acc}")

                train_total_accuracy = 0
                train_data_size = 0
                with torch.no_grad():
                    for batch in tqdm(trainloader):
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        y_pre = model(x)
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        train_total_accuracy += ((y_pre.argmax(dim=1) == y).sum().item())
                        train_data_size += len(y)
                    train_acc = train_total_accuracy / train_data_size
                    write.add_scalars("acc", {"train_acc": train_acc}, epoch)
                    print(f"The final train_acc:{train_acc:.3f}")


    pass


if __name__ == '__main__':
    train_csv_path = r"D:\machine_learning\animals_classfication\train_data.csv"
    test_csv_path = r"D:\machine_learning\animals_classfication\test_data.csv"
    start = Learn(train_csv_path=train_csv_path, test_csv_path=test_csv_path, epochs=20, batch_size=64, lr=0.001,
                  weight_decay=0.0001, mode="train")
    start.train()
