import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="手写数字识别", page_icon="✍️", layout="wide")

# ---- 加载/训练模型 ----
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mnist_model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.sidebar.success("✅ 模型已加载")
    else:
        st.sidebar.warning("⚠️ 模型文件未找到，请先运行 train_model.py")
        st.error("模型文件 (mnist_model.joblib) 不存在！请先在本地运行 `python train_model.py` 训练模型。")
        st.stop()
    return model

model = load_model()

# ---- UI 布局 ----
st.title("✍️ 手写数字识别")
st.markdown("在下方画布上写一个数字（0-9），AI 模型将识别并展示置信度 — 基于 MNIST 数据集训练的神经网络")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🎨 绘制画布")
    brush_size = st.slider("画笔大小", 8, 40, 20, key="brush")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # 透明填充
        stroke_width=brush_size,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        display_toolbar=True,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_btn = st.button("🔍 识别数字", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ 清除画布", use_container_width=True):
            st.rerun()

with col2:
    st.markdown("### 🤖 识别结果")

    if canvas_result.image_data is not None and predict_btn:
        # ---- 图像预处理 ----
        img = canvas_result.image_data  # shape: (H, W, 4) RGBA
        # 转为灰度
        img_gray = Image.fromarray(img.astype(np.uint8)).convert("L")
        # 缩放到 28x28
        img_28 = img_gray.resize((28, 28), Image.LANCZOS)
        # 转为 numpy 并归一化
        img_array = np.array(img_28).astype(np.float32) / 255.0
        # 展平为 784 维
        img_flat = img_array.reshape(1, -1)

        # ---- 模型预测 ----
        probs = model.predict_proba(img_flat)[0]  # 每个类别的概率
        pred_digit = np.argmax(probs)
        confidence = probs[pred_digit] * 100

        # ---- 显示结果 ----
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; color: white;">
            <div style="font-size: 0.9rem; opacity: 0.9;">预测结果</div>
            <div style="font-size: 5rem; font-weight: 800; line-height: 1.2;">{pred_digit}</div>
            <div style="font-size: 1.1rem;">置信度: <b>{confidence:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        # ---- 置信度柱状图 ----
        st.markdown("#### 📊 各类别置信度")
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ["#f5576c" if i == pred_digit else "#a0a0a0" for i in range(10)]
        bars = ax.bar(range(10), probs * 100, color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)
        # 在最佳预测上标注
        ax.text(pred_digit, probs[pred_digit] * 100 + 2, f"{confidence:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=11, color="#f5576c")
        ax.set_xticks(range(10))
        ax.set_xlabel("数字", fontsize=12)
        ax.set_ylabel("置信度 (%)", fontsize=12)
        ax.set_ylim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

        # 显示预处理后的图像
        with st.expander('🔍 查看模型"看到"的图像 (28×28)'):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**处理后图像 (28×28)**")
                fig2, ax2 = plt.subplots(figsize=(3, 3))
                ax2.imshow(img_array, cmap="gray")
                ax2.axis("off")
                st.pyplot(fig2)
            with c2:
                st.markdown("**像素值矩阵**")
                st.dataframe(
                    np.round(img_array * 255).astype(int),
                    use_container_width=True,
                )

    elif not predict_btn:
        st.info("👆 在画布上写一个数字，然后点击 **识别数字** 按钮")
        st.markdown("""
        ### 💡 使用提示
        1. 在左侧黑色画布上绘制一个数字 (0-9)
        2. 点击 **识别数字** 按钮
        3. 查看 AI 的预测结果和置信度

        写得越清晰，识别越准确！
        """)

# ---- 模型信息 ----
with st.expander("🧠 模型信息"):
    st.markdown(f"""
    - **模型架构**: 多层感知机 (MLP) — 3层隐藏层 (256, 128, 64 个神经元)
    - **训练数据**: MNIST 数据集 — 70,000 张手写数字图像
    - **测试准确率**: ~97.5%
    - **输入**: 28×28 灰度图像 (784 像素)
    - **输出**: 0-9 的概率分布
    - **框架**: scikit-learn MLPClassifier
    """)
