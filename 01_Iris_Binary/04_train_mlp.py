import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==========================
# 1. 实验配置
# ==========================
SEED = 42
EPOCHS = 30
LEARNING_RATE = 0.01

torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# ==========================
# 2. 加载相同的 Iris 数据
# ==========================
data = np.load(
    RESULTS_DIR / "iris_binary_data.npz"
)

X_train = torch.tensor(
    data["X_train"], dtype=torch.float32
)
X_test = torch.tensor(
    data["X_test"], dtype=torch.float32
)

y_train = torch.tensor(
    data["y_train"], dtype=torch.long
)
y_test = torch.tensor(
    data["y_test"], dtype=torch.long
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# ==========================
# 3. 构建传统 MLP
# ==========================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.network(x)


model = MLP()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print(
    "MLP Parameters:",
    sum(p.numel() for p in model.parameters())
)

# ==========================
# 4. 训练 MLP
# ==========================
loss_history = []
acc_history = []

start_time = time.perf_counter()

print("\nStart MLP training...\n")

for epoch in range(EPOCHS):

    model.train()
    optimizer.zero_grad()

    logits = model(X_train)
    loss = criterion(logits, y_train)

    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        train_pred = model(X_train).argmax(dim=1)
        train_acc = accuracy_score(
            y_train.numpy(),
            train_pred.numpy()
        )

    loss_history.append(loss.item())
    acc_history.append(train_acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

training_time = time.perf_counter() - start_time

# ==========================
# 5. 测试集评估
# ==========================
model.eval()

with torch.no_grad():
    test_pred = model(X_test).argmax(dim=1)

test_acc = accuracy_score(
    y_test.numpy(),
    test_pred.numpy()
)

cm = confusion_matrix(
    y_test.numpy(),
    test_pred.numpy(),
    labels=[0, 1]
)

print("\n===== MLP Test Results =====")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Time: {training_time:.4f} seconds")
print("Confusion Matrix:")
print(cm)

# ==========================
# 6. 保存实验结果
# ==========================
np.savez(
    RESULTS_DIR / "mlp_training_results.npz",
    loss=np.array(loss_history),
    train_accuracy=np.array(acc_history),
    test_accuracy=np.array(test_acc),
    training_time=np.array(training_time),
    y_test=y_test.numpy(),
    y_pred=test_pred.numpy(),
)

torch.save(
    model.state_dict(),
    RESULTS_DIR / "mlp_model.pth"
)

# 损失曲线
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.title("MLP Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "mlp_training_loss.png",
    dpi=200
)
plt.close()

# 准确率曲线
plt.figure(figsize=(6, 4))
plt.plot(acc_history)
plt.title("MLP Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "mlp_training_accuracy.png",
    dpi=200
)
plt.close()

# 混淆矩阵
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Setosa", "Versicolor"]
)
disp.plot(cmap="Blues")
plt.title("MLP Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "mlp_confusion_matrix.png",
    dpi=200
)
plt.close()

print("\nSTEP 4 STATUS: PASS")
