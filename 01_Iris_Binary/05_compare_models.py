from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# 读取已完成的实验结果
qnn = np.load(RESULTS_DIR / "qnn_training_results.npz")
mlp = np.load(RESULTS_DIR / "mlp_training_results.npz")

# 检查是否使用了相同测试集
assert np.array_equal(qnn["y_test"], mlp["y_test"])

# 汇总指标
df = pd.DataFrame({
    "Model": ["QNN", "MLP"],
    "Train Accuracy": [
        float(qnn["train_accuracy"][-1]),
        float(mlp["train_accuracy"][-1]),
    ],
    "Test Accuracy": [
        float(qnn["test_accuracy"]),
        float(mlp["test_accuracy"]),
    ],
    "Training Time (s)": [
        float(qnn["training_time"]),
        float(mlp["training_time"]),
    ],
    "Parameters": [8, 42],
})

print("\n===== QNN vs MLP =====")
print(df.to_string(index=False))

df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

# 绘制测试准确率对比
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df["Model"], df["Test Accuracy"])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Test Accuracy")
ax.set_title("QNN vs MLP: Iris Binary Classification")

for i, acc in enumerate(df["Test Accuracy"]):
    ax.text(i, acc + 0.02, f"{acc:.1%}", ha="center")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "model_accuracy_comparison.png",
    dpi=200
)
plt.close()

# 绘制训练耗时对比
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df["Model"], df["Training Time (s)"])
ax.set_ylabel("Training Time (seconds)")
ax.set_title("QNN vs MLP: Training Time")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "model_time_comparison.png",
    dpi=200
)
plt.close()

print("\nSTEP 5 STATUS: PASS")
