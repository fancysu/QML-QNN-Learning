import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 创建输出文件夹
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 2. 加载 Iris 数据集
iris = load_iris()

# 只保留 Setosa(0) 和 Versicolor(1)
mask = iris.target < 2

# 只选择花瓣长度和花瓣宽度
X = iris.data[mask][:, [2, 3]]
y = iris.target[mask]

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Class distribution:", np.bincount(y))

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4. 归一化为量子门旋转角度 [0, pi]
# 注意：Scaler 只能使用训练集拟合，避免数据泄漏
scaler = MinMaxScaler(feature_range=(0, np.pi))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 保存处理后的数据
np.savez(
    RESULTS_DIR / "iris_binary_data.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)

# 6. 输出数据基本信息
print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))
print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))
print("\nFirst 5 training samples:")
print(X_train[:5])
print("\nFirst 5 labels:")
print(y_train[:5])

# 7. 绘制数据分布
plt.figure(figsize=(7, 5))

for label, name in [(0, "Setosa"), (1, "Versicolor")]:
    selected = y_train == label
    plt.scatter(
        X_train[selected, 0],
        X_train[selected, 1],
        label=name,
        alpha=0.7,
    )

plt.xlabel("Petal Length (encoded angle)")
plt.ylabel("Petal Width (encoded angle)")
plt.title("Iris Binary Classification - Training Data")
plt.legend()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "iris_data_distribution.png", dpi=200)
plt.close()

print("\nSTEP 2 STATUS: PASS")
