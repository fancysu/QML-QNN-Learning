
import pennylane as qml
import torch
import sklearn

print("PennyLane:", qml.__version__)
print("PyTorch:", torch.__version__)
print("Scikit-learn:", sklearn.__version__)

# 创建一个双量子比特模拟器
dev = qml.device("default.qubit", wires=2)

# 定义量子电路
@qml.qnode(dev, interface="torch")
def quantum_circuit(x, weights):
    # 将两个输入特征编码到量子比特
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)

    # 让两个量子比特产生关联
    qml.CNOT(wires=[0, 1])

    # 可训练的量子门
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)

    # 返回第一个量子比特的 Z 期望值
    return qml.expval(qml.PauliZ(0))


# 模拟一条鸢尾花样本的两个特征
x = torch.tensor([0.5, 1.2], dtype=torch.float64)

# 可训练参数
weights = torch.tensor(
    [0.1, 0.2],
    dtype=torch.float64,
    requires_grad=True
)

# 执行量子电路
output = quantum_circuit(x, weights)

print("\nQuantum circuit output:", output.item())

# 测试能否反向传播
output.backward()
print("Parameter gradients:", weights.grad)

print("\nQuantum circuit test: PASS")
