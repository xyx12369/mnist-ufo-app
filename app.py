import streamlit as st

st.set_page_config(
    page_title="📊 数据可视化 — 作业项目",
    page_icon="📊",
    layout="wide",
)

# ---- 自定义CSS ----
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    .card-digit {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .card-ufo {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .card-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        font-size: 0.95rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ---- 页面内容 ----
st.markdown('<div class="main-title">📊 数据可视化项目</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">两个交互式数据可视化应用 — 手写数字识别 & UFO 目击国家预测</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card card-digit">
        <div class="card-icon">✍️</div>
        <div class="card-title">手写数字识别</div>
        <div class="card-desc">
            在画布上写下任意数字（0-9），AI 模型将实时识别并展示每个数字的置信度。
            基于 MNIST 数据集训练的神经网络模型，准确率超过 97%。
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔍 功能亮点
    - 🎨 **自由绘制** — 可调整画笔大小的绘制画布
    - 🤖 **AI 识别** — 基于神经网络的实时数字识别
    - 📊 **置信度展示** — 柱状图展示每个数字的预测概率
    - 🖼️ **图像预处理** — 展示模型实际"看到"的处理后图像
    """)

with col2:
    st.markdown("""
    <div class="card card-ufo">
        <div class="card-icon">🛸</div>
        <div class="card-title">UFO 目击国家预测</div>
        <div class="card-desc">
            输入 UFO 目击事件的地理坐标，系统将预测目击发生的国家，
            并在地图上可视化展示目击位置与周边信息。
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔍 功能亮点
    - 🌍 **坐标输入** — 支持手动输入经纬度或地图点击选取
    - 🗺️ **交互地图** — 基于 Folium 的交互式世界地图
    - 🎯 **国家预测** — 基于 KNN 算法的地理位置预测
    - 📈 **数据可视化** — UFO 目击数据的统计分析
    """)

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 在左侧边栏选择 **手写数字识别** 或 **UFO 国家预测** 开始体验！")
