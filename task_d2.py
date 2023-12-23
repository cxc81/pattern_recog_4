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
class BN(nn.Module):
    def __init__(self):
        super(BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bn2 = nn.BatchNorm2d(50)

        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.relu = nn.ReLU()

        self.bn3 = nn.BatchNorm2d(500)

        self.fc2 = nn.Linear(500, 10)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.relu(self.bn3(x))
        x = x.squeeze(2).squeeze(2)
        x = self.fc2(x)
        x = self.dropout(x)
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

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 创建模型实例
    model = BN().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 20
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

            # Print the loss
            if batch_idx + 1 == len(train_loader):
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

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