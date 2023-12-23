import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 设置随机数种子
torch.manual_seed(0)
np.random.seed(0)
# 判断cuda是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义基准网络结构
class SKSNet(nn.Module):
    def __init__(self):
        super(SKSNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, int(20 * s), 3)
        self.bn1_1 = nn.BatchNorm2d(int(20 * s))

        self.conv1_2 = nn.Conv2d(int(20 * s), int(20 * s), 3)
        self.bn1_2 = nn.BatchNorm2d(int(20 * s))

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(int(20 * s), int(50 * s), 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bn2 = nn.BatchNorm2d(int(50 * s))

        self.fc1 = nn.Linear(int(4 * 4 * 50 * s), int(500 * s))
        self.relu = nn.ReLU()

        self.bn3 = nn.BatchNorm1d(int(500 * s))

        self.fc2 = nn.Linear(int(500 * s), 10)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.pool2(self.bn2(self.conv2(x)))
        x = x.view(-1, int(4 * 4 * 50 * s))
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label.squeeze()

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # 载入训练集和测试集的数据
    train_images = np.load('train_images.npy')[:10000]
    train_labels = np.load('train_labels.npy')[:10000]
    test_images = np.load('test_images.npy')
    test_labels = np.load('test_labels.npy')

    # 转换数据类型
    train_images = train_images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    test_images = test_images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(train_images, train_labels)
    test_dataset = CustomDataset(test_images, test_labels)

    s_arr = [2, 1.5, 1, 0.5, 0.2]

    for s in s_arr:
        print("s = {}".format(s))
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 创建模型实例
        model = SKSNet().to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # 训练模型
        num_epochs = 20
        print("Epochs: ", end='')
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(data)
                output = nn.functional.softmax(output, dim=1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx + 1 == len(train_loader):
                    print("*", end='')

        print()

        # 在测试集上评估模型
        model.eval()
        total_correct = 0
        total_samples = 0
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()

        print("Total samples: {}".format(total_samples))
        print("Total correct: {}".format(total_correct))
        test_accuracy = total_correct / total_samples
        test_error_rate = 1 - test_accuracy
        print("Test error rate: {:.2f}%".format(test_error_rate * 100))
