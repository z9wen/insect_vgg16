import os
import torch
import random
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, fraction=1.0, random_seed=None):
        self.image_labels = []
        self.root_dir = root_dir
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            if random_seed is not None:
                random.seed(random_seed)
            if fraction < 1.0:
                lines = random.sample(lines, int(len(lines) * fraction))
            for line in lines:
                image_path, label = line.strip().split()
                self.image_labels.append((image_path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path, label = self.image_labels[idx]
        img_full_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用数据集的十分之一，随机选择
fraction = 0.1
random_seed = 42  # 保证每次运行随机选择相同的数据子集
root_dir = 'z/dataset/images'  # 根目录路径

# 加载数据
train_dataset = CustomDataset('z/dataset/train.txt', root_dir=root_dir, transform=transform, fraction=fraction, random_seed=random_seed)
val_dataset = CustomDataset('z/dataset/val.txt', root_dir=root_dir, transform=transform, fraction=fraction, random_seed=random_seed)
test_dataset = CustomDataset('z/dataset/test.txt', root_dir=root_dir, transform=transform, fraction=fraction, random_seed=random_seed)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的VGG16模型
from torchvision.models import vgg16, VGG16_Weights

# 加载预训练的VGG16模型
weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights)

# 冻结所有卷积层的参数
for param in model.features.parameters():
    param.requires_grad = False

# 修改最后的全连接层以匹配你的类别数
num_classes = len(open('z/dataset/classes.txt').readlines())
model.classifier[6] = nn.Linear(4096, num_classes)

# 使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 初始化 TensorBoard
writer = SummaryWriter('runs/vgg16_experiment')

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}/{num_epochs}...')
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (batch_idx + 1) % 10 == 0:  # 每10个batch输出一次
            print(f'Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # 将训练损失和准确率写入 TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

    # 验证模型
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

    # 将验证损失和准确率写入 TensorBoard
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

# 测试模型
print('Starting testing...')
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 关闭 TensorBoard
writer.close()
