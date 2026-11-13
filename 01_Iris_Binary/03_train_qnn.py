
import time
from pathlib import Path

import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================
# 1. 实验配置
# ==========================
SEED = 42
EPOCHS = 30
LEARNING_RATE = 0.08
N_QUBITS = 2
N_LAYERS = 2

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ==========================
# 2. 加载第二步准备的数据
# ==========================
data = np.load(RESULTS_DIR / "iris_binary_data.npz")

X_train = torch.tensor(
    data["X_train"], dtype=torch.float64
)
X_test = torch.tensor(
    data["X_test"], dtype=torch.float64
)

y_train = data["y_train"]
y_test = data["y_test"]

# Setosa(0) -> +1
# Versicolor(1) -> -1
y_train_target = torch.tensor(
    1 - 2 * y_train, dtype=torch.float64
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# ==========================
# 3. 创建量子模拟器
# ==========================
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="best")
def quantum_circuit(x, weights):

    # 输入特征编码
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)

    # 两层可训练量子电路
    for layer in range(N_LAYERS):

        for wire in range(N_QUBITS):
            qml.RY(weights[layer, wire], wires=wire)
            qml.RZ(
                weights[layer, wire + N_QUBITS],
                wires=wire
            )

        # 双量子比特纠缠门
        qml.CNOT(wires=[0, 1])

    # 返回测量期望值
    return qml.expval(qml.PauliZ(0))


# ==========================
# 4. 初始化可训练参数
# ==========================
weights = torch.nn.Parameter(
    0.1 * torch.randn(
        N_LAYERS,
        N_QUBITS * 2,
        dtype=torch.float64
    )
)

optimizer = torch.optim.Adam(
    [weights],
    lr=LEARNING_RATE
)

# ==========================
# 5. 定义模型预测函数
# ==========================
def predict_scores(X):
    outputs = [
        quantum_circuit(x, weights)
        for x in X
    ]
    return torch.stack(outputs)


def to_labels(scores):
    # 大于等于 0 -> 类别 0
    # 小于 0 -> 类别 1
    return (scores < 0).long()


# ==========================
# 6. 训练 QNN
# ==========================
loss_history = []
acc_history = []

start_time = time.time()

print("\nStart QNN training...\n")

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    # 前向计算
    outputs = predict_scores(X_train)

    # 均方误差损失
    loss = torch.mean(
        (outputs - y_train_target) ** 2
    )

    # 反向传播
    loss.backward()
    optimizer.step()

    # 记录训练指标
    with torch.no_grad():
        train_pred = to_labels(
            predict_scores(X_train)
        )

        train_acc = accuracy_score(
            y_train,
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

training_time = time.time() - start_time

# ==========================
# 7. 测试集评估
# ==========================
with torch.no_grad():
    test_scores = predict_scores(X_test)
    test_pred = to_labels(test_scores).numpy()

test_acc = accuracy_score(y_test, test_pred)
cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

print("\n===== QNN Test Results =====")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
print("Confusion Matrix:")
print(cm)

# ==========================
# 8. 保存训练结果
# ==========================
np.savez(
    RESULTS_DIR / "qnn_training_results.npz",
    loss=np.array(loss_history),
    train_accuracy=np.array(acc_history),
    test_accuracy=np.array(test_acc),
    training_time=np.array(training_time),
    y_test=y_test,
    y_pred=test_pred,
    weights=weights.detach().numpy(),
)

# 绘制损失与准确率曲线
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(loss_history)
axes[0].set_title("QNN Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].grid(alpha=0.2)

axes[1].plot(acc_history)
axes[1].set_title("QNN Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1.05)
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "qnn_training_curves.png",
    dpi=200
)
plt.close()

# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Setosa", "Versicolor"]
)

disp.plot(cmap="Blues")
plt.title("QNN Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "qnn_confusion_matrix.png",
    dpi=200
)
plt.close()

print("\nSTEP 3 STATUS: PASS")
