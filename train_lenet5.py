import torch
import torch.nn as nn
import torch.optim as optim
from simple_cnn import load_data # 复用任务一的加载逻辑
from lenet5 import LeNet5

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练 LeNet-5...")
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} 完成")

    # 测试
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
    
    print(f"LeNet-5 最终准确率: {100 * correct / len(test_loader.dataset):.2f}%")

if __name__ == "__main__":
    run()