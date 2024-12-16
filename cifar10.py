import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 下载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 类别标签
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 2. 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输入通道3（RGB），输出通道32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入通道32，输出通道64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输入通道64，输出通道128
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，2x2 最大池化
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 全连接层1
        self.fc2 = nn.Linear(512, 10)  # 输出层，10个类别
    def forward(self, x):
        # 卷积层 + 激活函数 + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        # 扁平化
        x = x.view(-1, 128 * 4 * 4)
        # 全连接层 + 激活函数
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 3. 初始化模型并将模型转移到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CNN().to(device)

# 4. 定义损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数


# 5. 定义训练和验证函数
def train_model(optimizer_name, num_epochs=100):
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay=0.0005)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=0.0005)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=0.0005)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    # 用于保存每个 epoch 的训练损失和验证准确率
    train_losses = []
    val_accuracies = []

    # 训练过程
    for epoch in range(num_epochs):
        # 训练阶段
        net.train()  # 设置为训练模式
        running_loss = 0.0
        loop = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100, unit="batch")

        for inputs, labels in loop:
            # 将数据转移到GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 反向传播 + 优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 更新进度条
            loop.set_postfix(loss=running_loss / (loop.n + 1))

        # 记录训练损失
        train_losses.append(running_loss / len(trainloader))

        # 每个epoch结束后计算验证集的准确率
        val_accuracy = validate_model()
        val_accuracies.append(val_accuracy)

        print(f"Accuracy: {val_accuracy:.2f}%")

    # 绘制训练损失与验证准确率
    plot_loss_and_accuracy(train_losses, val_accuracies, num_epochs)


def validate_model():
    # 进入评估模式
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            # 计算预测准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def plot_loss_and_accuracy(train_losses, val_accuracies, num_epochs):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制训练损失
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')  # 修改为 "Loss"
    ax1.plot(range(1, num_epochs + 1), train_losses, label='Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 创建第二个y轴来显示准确率
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:green')  # 修改为 "Accuracy"
    ax2.plot(range(1, num_epochs + 1), val_accuracies, label='Accuracy', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # 显示图例（Accuracy 放在 Loss 下方）
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.1, 0.1))  # 修改位置放到左上角下方

    # 显示图形
    plt.title('Loss and Accuracy')
    plt.tight_layout()
    plt.show()


# 6. 比较不同优化器的效果
optimizers = ['SGD', 'Adam', 'RMSprop']
results = {}

for optimizer_name in optimizers:
    print(f"\nTraining with {optimizer_name} optimizer...")
    net = CNN().to(device)  # 重新初始化模型
    train_model(optimizer_name)  # 训练
