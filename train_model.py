"""
训练 MNIST 手写数字识别模型
使用 scikit-learn MLPClassifier (多层感知机)
"""
import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("Loading MNIST dataset...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
X = X / 255.0
y = y.astype(np.uint8)

print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Label distribution: 0~9, ~{X.shape[0] // 10} per class")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Train MLP model
print("\nTraining MLP neural network...")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=200,
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=100,
    random_state=42,
    verbose=True,
)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
model_path = "mnist_model.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")
