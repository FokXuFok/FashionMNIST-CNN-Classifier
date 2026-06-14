import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

print("=" * 60)
print(f"训练集大小: {len(trainset)}, 测试集大小: {len(testset)}")
print(f"批次大小: 32")
print("=" * 60)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(1, 1)

        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 第1层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        # 第2层
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # 第3层
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # 第4层
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        # 第5层
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = CNN().to(device)
print(model)


# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# 定义标签平滑损失函数
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing  # 真实标签的置信度
        self.smoothing = smoothing  # 平滑值
        self.classes = classes
        self.dim = dim

    def forward(self, pred, target):
        # 将标签转换为one-hot编码
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        # 应用标签平滑
        # 真实标签: 1 - smoothing + smoothing / classes
        # 其他标签: smoothing / classes
        target_smooth = target_one_hot * self.confidence + self.smoothing / self.classes

        log_probs = F.log_softmax(pred, dim=self.dim)
        loss = -torch.sum(target_smooth * log_probs) / pred.size(0)

        return loss

# 使用标签平滑损失，平滑参数设为0.1（常用值）
criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)

# 使用AdamW优化器，固定学习率为0.001
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,  # 固定学习率
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-3    # 权重衰减
)

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # 计算标签平滑损失
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率（使用原始标签，不是平滑后的标签）
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_count += 1

        if batch_idx % 100 == 99:
            avg_loss = total_loss / batch_count
            accuracy = 100. * correct / total
            print(f'Epoch: {epoch} [{batch_idx + 1}/{len(dataloader)}] '
                  f'Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%')

            total_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0


# 测试函数
def test_model(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 计算标签平滑损失
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(dataloader)

    print(f'测试结果 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%')
    model.train()
    return accuracy, avg_loss


# 主训练循环
print("\n" + "=" * 60)
print("开始训练（使用标签平滑损失，smoothing=0.1）")
print("=" * 60)

best_acc = 0.0
train_history = []
test_history = []

# 训练10个epoch
for epoch in range(10):
    print(f'\nEpoch {epoch + 1}/10:')

    # 训练
    train_epoch(model, trainloader, criterion, optimizer, epoch + 1)

    # 测试
    test_acc, test_loss = test_model(model, testloader, criterion)

    # 记录历史
    test_history.append({
        'epoch': epoch + 1,
        'accuracy': test_acc,
        'loss': test_loss
    })

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': test_acc,
            'loss': test_loss,
            'smoothing': 0.1  # 保存标签平滑参数
        }, '5layer_cnn_labelsmooth_best_model.pth')
        print(f'✓ 保存最佳模型，准确率: {test_acc:.2f}%')

print("\n" + "=" * 60)
print("训练完成!")
print(f"最佳测试准确率: {best_acc:.2f}%")
print("=" * 60)

# 保存模型
torch.save(model.state_dict(), '5layer_cnn_labelsmooth_final_model.pth')


def plot_training_history(test_history):

    epochs = [item['epoch'] for item in test_history]
    accuracies = [item['accuracy'] for item in test_history]
    losses = [item['loss'] for item in test_history]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制准确率图表
    ax1.plot(epochs, accuracies, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy per Epoch', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # 在图表上标记数值
    for i, (epoch, acc) in enumerate(zip(epochs, accuracies)):
        ax1.annotate(f'{acc:.1f}%',
                     xy=(epoch, acc),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=9)

    # 绘制损失图表
    ax2.plot(epochs, losses, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Test Loss per Epoch', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    # 在图表上标记数值
    for i, (epoch, loss) in enumerate(zip(epochs, losses)):
        ax2.annotate(f'{loss:.4f}',
                     xy=(epoch, loss),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=9)

    plt.tight_layout()

    # 保存图表
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    # 显示图表
    plt.show()

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("训练统计摘要:")
    print("=" * 60)
    print(f"{'Epoch':<10}{'Accuracy':<15}{'Loss':<15}")
    print("-" * 40)
    for item in test_history:
        print(f"{item['epoch']:<10}{item['accuracy']:<15.2f}{item['loss']:<15.4f}")
    print("=" * 60)
    print(f"最高准确率: {max(accuracies):.2f}% (Epoch {epochs[accuracies.index(max(accuracies))]})")
    print(f"最低损失: {min(losses):.4f} (Epoch {epochs[losses.index(min(losses))]})")

# 调用可视化函数
if test_history:
    plot_training_history(test_history)
else:
    print("警告: 没有训练历史数据可供可视化")