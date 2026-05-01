import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# إعدادات الصفحة (Minimalist)
st.set_page_config(layout="wide", page_title="Gradient Descent")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stSlider [data-baseweb="slider"] { margin-bottom: 20px; }
    .metric-container {
        display: flex;
        justify-content: space-around;
        padding: 10px;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 20px;
    }
    .metric-box { text-align: center; }
    .label { font-size: 12px; color: #999; text-transform: uppercase; }
    .value { font-size: 20px; font-weight: 600; color: #222; }
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# توليد البيانات
x = np.linspace(-5, 5, 20)
y = 0.8 * x + 1 + np.random.normal(0, 0.5, size=len(x))

# واجهة جانبية بسيطة جداً
with st.sidebar:
    st.markdown("### التحكم")
    lr = st.select_slider("سرعة التعلم", options=[0.001, 0.01, 0.05, 0.1], value=0.05)
    steps = st.slider("الخطوات", 10, 100, 50)
    start = st.button("ابدأ التدريب")

# منطقة العرض
metrics_area = st.empty()
plot_area = st.empty()

# نقطة البداية
w, b = -2.0, -2.0

if start:
    for i in range(steps):
        # 1. التوقع وحساب الخطأ
        y_p = w * x + b
        error = y_p - y
        loss = np.mean(error**2)
        
        # 2. المشتقات (The Calculus)
        dw = (2/len(x)) * np.dot(error, x)
        db = (2/len(x)) * np.sum(error)
        
        # 3. التحديث
        w -= lr * dw
        b -= lr * db

        # عرض المقاييس بتصميم نظيف
        metrics_area.markdown(f"""
            <div class="metric-container">
                <div class="metric-box"><div class="label">Step</div><div class="value">{i+1}</div></div>
                <div class="metric-box"><div class="label">Loss</div><div class="value">{loss:.3f}</div></div>
                <div class="metric-box"><div class="label">Weight</div><div class="value">{w:.2f}</div></div>
            </div>
            """, unsafe_allow_html=True)

        # الرسم البياني (16:9)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#333', size=8, opacity=0.4)))
        fig.add_trace(go.Scatter(x=[-6, 6], y=[w*(-6)+b, w*6+b], mode='lines', line=dict(color='#007BFF', width=3)))
        
        fig.update_layout(
            width=900, height=506, # نسبة 16:9
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False
        )
        
        plot_area.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        time.sleep(0.05)
else:
    st.markdown("<h2 style='text-align:center; color:#ccc; margin-top:150px;'>جاهز للبدء</h2>", unsafe_allow_html=True)
