import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium, folium_static
import reverse_geocoder as rg
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="UFO 国家预测", page_icon="🛸", layout="wide")

st.title("🛸 UFO 目击国家预测")
st.markdown("输入 UFO 目击坐标，系统将通过地理位置分析预测目击发生的国家，并在地图上可视化展示")

# ---- 世界主要城市数据（用于 KNN 预测）----
@st.cache_data
def load_world_cities():
    """加载世界主要城市坐标数据集，用于训练国家预测模型"""
    data = [
        # 亚洲
        ("Beijing", 39.9042, 116.4074, "China"),
        ("Shanghai", 31.2304, 121.4737, "China"),
        ("Tokyo", 35.6762, 139.6503, "Japan"),
        ("Osaka", 34.6937, 135.5023, "Japan"),
        ("Seoul", 37.5665, 126.9780, "South Korea"),
        ("Busan", 35.1796, 129.0756, "South Korea"),
        ("Mumbai", 19.0760, 72.8777, "India"),
        ("New Delhi", 28.6139, 77.2090, "India"),
        ("Bangkok", 13.7563, 100.5018, "Thailand"),
        ("Singapore", 1.3521, 103.8198, "Singapore"),
        ("Kuala Lumpur", 3.1390, 101.6869, "Malaysia"),
        ("Jakarta", -6.2088, 106.8456, "Indonesia"),
        ("Manila", 14.5995, 120.9842, "Philippines"),
        ("Hanoi", 21.0278, 105.8342, "Vietnam"),
        ("Taipei", 25.0330, 121.5654, "Taiwan"),
        ("Hong Kong", 22.3193, 114.1694, "China"),
        ("Dhaka", 23.8103, 90.4125, "Bangladesh"),
        ("Kathmandu", 27.7172, 85.3240, "Nepal"),
        ("Colombo", 6.9271, 79.8612, "Sri Lanka"),
        ("Ulaanbaatar", 47.8864, 106.9057, "Mongolia"),
        # 欧洲
        ("London", 51.5074, -0.1278, "United Kingdom"),
        ("Manchester", 53.4808, -2.2426, "United Kingdom"),
        ("Paris", 48.8566, 2.3522, "France"),
        ("Marseille", 43.2965, 5.3698, "France"),
        ("Berlin", 52.5200, 13.4050, "Germany"),
        ("Munich", 48.1351, 11.5820, "Germany"),
        ("Madrid", 40.4168, -3.7038, "Spain"),
        ("Barcelona", 41.3874, 2.1686, "Spain"),
        ("Rome", 41.9028, 12.4964, "Italy"),
        ("Milan", 45.4642, 9.1900, "Italy"),
        ("Amsterdam", 52.3676, 4.9041, "Netherlands"),
        ("Brussels", 50.8503, 4.3517, "Belgium"),
        ("Vienna", 48.2082, 16.3738, "Austria"),
        ("Zurich", 47.3769, 8.5417, "Switzerland"),
        ("Stockholm", 59.3293, 18.0686, "Sweden"),
        ("Oslo", 59.9139, 10.7522, "Norway"),
        ("Copenhagen", 55.6761, 12.5683, "Denmark"),
        ("Helsinki", 60.1699, 24.9384, "Finland"),
        ("Warsaw", 52.2297, 21.0122, "Poland"),
        ("Prague", 50.0755, 14.4378, "Czech Republic"),
        ("Budapest", 47.4979, 19.0402, "Hungary"),
        ("Athens", 37.9838, 23.7275, "Greece"),
        ("Lisbon", 38.7223, -9.1393, "Portugal"),
        ("Dublin", 53.3498, -6.2603, "Ireland"),
        ("Moscow", 55.7558, 37.6173, "Russia"),
        ("Saint Petersburg", 59.9343, 30.3351, "Russia"),
        ("Istanbul", 41.0082, 28.9784, "Turkey"),
        ("Ankara", 39.9334, 32.8597, "Turkey"),
        ("Kyiv", 50.4501, 30.5234, "Ukraine"),
        ("Minsk", 53.9006, 27.5590, "Belarus"),
        # 北美
        ("New York", 40.7128, -74.0060, "United States"),
        ("Los Angeles", 34.0522, -118.2437, "United States"),
        ("Chicago", 41.8781, -87.6298, "United States"),
        ("Houston", 29.7604, -95.3698, "United States"),
        ("Miami", 25.7617, -80.1918, "United States"),
        ("Seattle", 47.6062, -122.3321, "United States"),
        ("San Francisco", 37.7749, -122.4194, "United States"),
        ("Boston", 42.3601, -71.0589, "United States"),
        ("Denver", 39.7392, -104.9903, "United States"),
        ("Phoenix", 33.4484, -112.0740, "United States"),
        ("Las Vegas", 36.1699, -115.1398, "United States"),
        ("Toronto", 43.6532, -79.3832, "Canada"),
        ("Vancouver", 49.2827, -123.1207, "Canada"),
        ("Montreal", 45.5017, -73.5673, "Canada"),
        ("Calgary", 51.0447, -114.0719, "Canada"),
        ("Mexico City", 19.4326, -99.1332, "Mexico"),
        ("Guadalajara", 20.6597, -103.3496, "Mexico"),
        # 南美
        ("São Paulo", -23.5505, -46.6333, "Brazil"),
        ("Rio de Janeiro", -22.9068, -43.1729, "Brazil"),
        ("Brasília", -15.7975, -47.8919, "Brazil"),
        ("Buenos Aires", -34.6037, -58.3816, "Argentina"),
        ("Córdoba", -31.4201, -64.1888, "Argentina"),
        ("Santiago", -33.4489, -70.6693, "Chile"),
        ("Lima", -12.0464, -77.0428, "Peru"),
        ("Bogotá", 4.7110, -74.0721, "Colombia"),
        ("Caracas", 10.4806, -66.9036, "Venezuela"),
        ("Quito", -0.1807, -78.4678, "Ecuador"),
        ("Montevideo", -34.9011, -56.1645, "Uruguay"),
        # 大洋洲
        ("Sydney", -33.8688, 151.2093, "Australia"),
        ("Melbourne", -37.8136, 144.9631, "Australia"),
        ("Brisbane", -27.4698, 153.0251, "Australia"),
        ("Perth", -31.9505, 115.8605, "Australia"),
        ("Auckland", -36.8509, 174.7645, "New Zealand"),
        ("Wellington", -41.2865, 174.7762, "New Zealand"),
        # 非洲
        ("Cairo", 30.0444, 31.2357, "Egypt"),
        ("Alexandria", 31.2001, 29.9187, "Egypt"),
        ("Lagos", 6.5244, 3.3792, "Nigeria"),
        ("Abuja", 9.0579, 7.4951, "Nigeria"),
        ("Nairobi", -1.2921, 36.8219, "Kenya"),
        ("Johannesburg", -26.2041, 28.0473, "South Africa"),
        ("Cape Town", -33.9249, 18.4241, "South Africa"),
        ("Casablanca", 33.5731, -7.5898, "Morocco"),
        ("Algiers", 36.7538, 3.0588, "Algeria"),
        ("Addis Ababa", 9.0320, 38.7469, "Ethiopia"),
        ("Dar es Salaam", -6.7924, 39.2083, "Tanzania"),
        ("Accra", 5.6037, -0.1870, "Ghana"),
        # 中东
        ("Dubai", 25.2048, 55.2708, "United Arab Emirates"),
        ("Abu Dhabi", 24.4539, 54.3773, "United Arab Emirates"),
        ("Riyadh", 24.7136, 46.6753, "Saudi Arabia"),
        ("Jeddah", 21.4858, 39.1925, "Saudi Arabia"),
        ("Doha", 25.2854, 51.5310, "Qatar"),
        ("Kuwait City", 29.3759, 47.9774, "Kuwait"),
        ("Muscat", 23.5880, 58.3829, "Oman"),
        ("Tehran", 35.6892, 51.3890, "Iran"),
        ("Baghdad", 33.3152, 44.3661, "Iraq"),
        ("Jerusalem", 31.7683, 35.2137, "Israel"),
        ("Amman", 31.9454, 35.9284, "Jordan"),
    ]
    df = pd.DataFrame(data, columns=["city", "lat", "lon", "country"])
    return df

cities_df = load_world_cities()

# ---- 训练 KNN 国家预测模型 ----
@st.cache_resource
def train_country_model(_df):
    """使用 KNN 训练国家预测模型"""
    X = _df[["lat", "lon"]].values
    y = _df["country"].values
    # KNN 用于分类坐标对应的国家
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="haversine",  # 球面距离
    )
    # 将经纬度转为弧度用于haversine
    X_rad = np.radians(X)
    knn.fit(X_rad, y)
    return knn

knn_model = train_country_model(cities_df)

def predict_country(lat, lon):
    """使用 KNN 预测坐标对应的国家，并返回距离信息"""
    coords_rad = np.radians([[lat, lon]])
    country = knn_model.predict(coords_rad)[0]
    probs = knn_model.predict_proba(coords_rad)[0]
    # 找到最近的城市
    distances, indices = knn_model.kneighbors(coords_rad, n_neighbors=5)
    neighbors = []
    for dist, idx in zip(distances[0], indices[0]):
        row = cities_df.iloc[idx]
        dist_km = dist * 6371  # 弧度距离转公里
        neighbors.append({
            "city": row["city"],
            "country": row["country"],
            "lat": row["lat"],
            "lon": row["lon"],
            "distance_km": dist_km,
        })
    confidence = np.max(probs) * 100
    return country, confidence, neighbors

# ---- 随机 UFO 目击样本数据 ----
sample_sightings = [
    {"lat": 37.2343, "lon": -115.8066, "note": "内华达州 51区附近"},
    {"lat": 33.9142, "lon": -106.3397, "note": "新墨西哥州 罗斯威尔"},
    {"lat": 52.4814, "lon": -1.8995, "note": "英国 伯明翰"},
    {"lat": -22.9519, "lon": -43.2105, "note": "巴西 里约热内卢"},
    {"lat": 35.6762, "lon": 139.6503, "note": "日本 东京"},
    {"lat": 48.8566, "lon": 2.3522, "note": "法国 巴黎"},
    {"lat": -33.8688, "lon": 151.2093, "note": "澳大利亚 悉尼"},
    {"lat": 55.7558, "lon": 37.6173, "note": "俄罗斯 莫斯科"},
]

# ---- UI ----
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### 📍 输入坐标")

    # 快捷选择
    st.markdown("**快捷选择 UFO 目击地点：**")
    sample_labels = [f"{s['note']}" for s in sample_sightings]
    selected_sample = st.selectbox(
        "选择一个预设目击地点",
        ["自定义输入..."] + sample_labels,
        key="sample_select"
    )

    if selected_sample != "自定义输入...":
        idx = sample_labels.index(selected_sample)
        default_lat = sample_sightings[idx]["lat"]
        default_lon = sample_sightings[idx]["lon"]
    else:
        default_lat = 40.0
        default_lon = -100.0

    lat = st.number_input("纬度 (Latitude)", min_value=-90.0, max_value=90.0,
                          value=default_lat, step=0.01, format="%.4f")
    lon = st.number_input("经度 (Longitude)", min_value=-180.0, max_value=180.0,
                          value=default_lon, step=0.01, format="%.4f")

    predict_btn = st.button("🔮 预测国家", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### ℹ️ 说明
    - **KNN 分类器** 基于全球 ~100 个主要城市的坐标训练
    - 使用 **哈弗辛距离 (Haversine)** 在球面上计算最近邻
    - 预测结果基于 5 个最近城市的加权投票
    """)

with col_result:
    st.markdown("### 🎯 预测结果")

    if predict_btn:
        country, confidence, neighbors = predict_country(lat, lon)

        # 结果卡片
        conf_color = "#4caf50" if confidence > 80 else ("#ff9800" if confidence > 50 else "#f44336")
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 16px; color: white; margin-bottom: 1rem;">
            <div style="font-size: 0.9rem; opacity: 0.9;">预测国家</div>
            <div style="font-size: 2.5rem; font-weight: 800; line-height: 1.3;">{country}</div>
            <div style="font-size: 1.1rem;">置信度: <b style="color: {conf_color};">{confidence:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        # 最近邻信息
        st.markdown("#### 🏙️ 最近的5个城市（K近邻）")
        neighbor_df = pd.DataFrame(neighbors)
        neighbor_df["距离"] = neighbor_df["distance_km"].apply(lambda x: f"{x:.0f} km")
        st.dataframe(
            neighbor_df[["city", "country", "距离"]].rename(
                columns={"city": "城市", "country": "国家"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        # 置信度分布
        st.markdown("#### 📊 预测概率分布")
        distances_km = np.array([n["distance_km"] for n in neighbors])
        # 计算权重（距离越近权重越高）
        weights = np.exp(-distances_km / 100)  # 指数衰减
        weights = weights / weights.sum() * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[n["city"] for n in neighbors],
            y=weights,
            text=[f"{w:.1f}%" for w in weights],
            textposition="outside",
            marker_color=["#4facfe" if i == 0 else "#a0c4ff" for i in range(len(neighbors))],
        ))
        fig.update_layout(
            xaxis_title="最近城市",
            yaxis_title="权重 (%)",
            yaxis_range=[0, max(weights) * 1.3],
            margin=dict(t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 输入坐标或选择一个预设地点，然后点击 **预测国家**")

# ---- 地图 ----
st.markdown("---")
st.markdown("### 🗺️ 交互式世界地图")

map_col1, map_col2 = st.columns([3, 1])

with map_col2:
    st.markdown("**地图图例**")
    st.markdown("🔵 全球城市 (KNN训练数据)")
    if predict_btn:
        st.markdown("🔴 目击地点")
        st.markdown("⭐ 最近城市")

with map_col1:
    m = folium.Map(location=[lat, lon], zoom_start=3 if not predict_btn else 5,
                   tiles="CartoDB positron", control_scale=True)
    Fullscreen().add_to(m)

    # 添加全球城市标记（抽样以减少性能问题）
    sample_cities = cities_df.sample(min(50, len(cities_df)), random_state=42)
    for _, row in sample_cities.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            color="#4facfe",
            fill=True,
            fill_opacity=0.5,
            popup=f"{row['city']}, {row['country']}",
        ).add_to(m)

    if predict_btn:
        # 目击位置
        folium.Marker(
            location=[lat, lon],
            popup=f"🛸 UFO 目击位置<br>({lat:.4f}, {lon:.4f})<br><b>预测: {country}</b>",
            icon=folium.Icon(color="red", icon="eye", prefix="fa"),
        ).add_to(m)

        # 最近城市高亮
        for i, n in enumerate(neighbors[:3]):
            folium.Marker(
                location=[n["lat"], n["lon"]],
                popup=f"{n['city']}, {n['country']}<br>距离: {n['distance_km']:.0f} km",
                icon=folium.Icon(color="green" if i == 0 else "blue",
                                 icon="star" if i == 0 else "info-sign",
                                 prefix="fa"),
            ).add_to(m)

        # 连线
        for n in neighbors[:1]:
            folium.PolyLine(
                locations=[[lat, lon], [n["lat"], n["lon"]]],
                color="#f5576c",
                weight=2,
                dash_array="5,5",
                popup=f"距离: {n['distance_km']:.0f} km",
            ).add_to(m)

    st_folium(m, width="100%", height=450)

# ---- UFO 数据统计 ----
st.markdown("---")
st.markdown("### 📈 全球 UFO 目击报告统计")

@st.cache_data
def load_ufo_stats():
    """生成模拟的 UFO 目击统计（按国家）"""
    stats = [
        ("United States", 85000), ("United Kingdom", 12000), ("Canada", 8000),
        ("Australia", 5000), ("Germany", 3500), ("France", 3000),
        ("Brazil", 2500), ("Japan", 2000), ("Russia", 1800), ("India", 1500),
        ("Mexico", 1200), ("Spain", 1000), ("Italy", 900), ("Argentina", 800),
        ("South Africa", 600), ("China", 500), ("Sweden", 400), ("Netherlands", 350),
    ]
    return pd.DataFrame(stats, columns=["country", "sightings"])

ufo_stats = load_ufo_stats()

fig_stats = px.bar(
    ufo_stats.sort_values("sightings", ascending=True).tail(15),
    x="sightings", y="country", orientation="h",
    title="UFO 目击报告最多的国家（模拟数据）",
    color="sightings",
    color_continuous_scale="Blues",
)
fig_stats.update_layout(height=400, yaxis_title="", xaxis_title="目击报告数量")
st.plotly_chart(fig_stats, use_container_width=True)
st.caption("注：此为模拟数据，基于公开 UFO 报告数据集的趋势")
