# 📊 数据可视化项目 — 手写数字识别 & UFO 国家预测

两个交互式数据可视化应用，基于 Streamlit 构建。

## ✍️ 手写数字识别

在画布上绘制数字（0-9），AI 神经网络模型实时识别并展示置信度。

- **模型**: 多层感知机 (MLP) — 3层隐藏层 (256, 128, 64)
- **训练数据**: MNIST (70,000 张手写数字图像)
- **准确率**: ~97.5%
- **功能**: 可调画笔大小、实时预测、置信度柱状图、图像预处理展示

## 🛸 UFO 目击国家预测

输入经纬度坐标，基于 KNN 地理分类器预测目击发生的国家。

- **模型**: K-近邻分类器 (K=5) + 哈弗辛球面距离
- **训练数据**: 全球 100+ 主要城市坐标
- **功能**: 交互式世界地图、最近城市展示、预测置信度、UFO 统计

## 🚀 本地运行

```bash
# 1. 克隆仓库
git clone https://github.com/xyx12369/mnist-ufo-app.git
cd mnist-ufo-app

# 2. 安装依赖
pip install -r requirements.txt

# 3. 训练 MNIST 模型（仅需一次）
python train_model.py

# 4. 启动应用
streamlit run app.py
```

## 📦 部署

本项目可一键部署到 [Streamlit Cloud](https://streamlit.io/cloud)：

1. 将代码推送到 GitHub
2. 在 [share.streamlit.io](https://share.streamlit.io) 连接仓库
3. Streamlit Cloud 自动部署并生成 `.streamlit.app` 网址

## 🧠 技术栈

| 功能 | 技术 |
|------|------|
| Web 框架 | Streamlit |
| MNIST 模型 | scikit-learn MLPClassifier |
| 手写画布 | streamlit-drawable-canvas |
| UFO 分类器 | scikit-learn KNeighborsClassifier |
| 地图可视化 | Folium + streamlit-folium |
| 图表 | Matplotlib + Plotly |

## 📄 许可

MIT License
