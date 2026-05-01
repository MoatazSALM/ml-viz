import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 1. إعدادات الصفحة لتقليل الهوامش لأقصى حد
st.set_page_config(layout="centered")

st.markdown("""
    <style>
    .block-container { padding: 0.5rem; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stMetric { background-color: #f8f9fa; padding: 5px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 2. توليد بيانات Linear Regression ثابتة
np.random.seed(42)
x_data = np.linspace(-8, 8, 15)
y_true = 1.2 * x_data + 2 + np.random.normal(0, 1.5, size=len(x_data))

# 3. واجهة التحكم (السلايدرز تحت بعض لتحكم أسهل وأدق)
w = st.slider("Weight (w) - ميل الخط", -4.0, 4.0, 1.0, 0.01)
b = st.slider("Bias (b) - إزاحة الخط", -10.0, 10.0, 0.0, 0.1)

# حساب التوقعات والخطأ
y_pred = w * x_data + b
mse = np.mean((y_true - y_pred)**2)

# 4. عرض المعادلة والـ MSE بشكل أنيق
sign = "+" if b >= 0 else "-"
st.latex(f"Model: y = {w:.2f}x {sign} {abs(b):.1f}")
st.metric("Mean Squared Error (Loss)", f"{mse:.2f}")

# 5. إنشاء الرسم بالأبعاد المطلوبة 600x250
fig = go.Figure()

# نقاط البيانات الحقيقية
fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers', 
                         marker=dict(color='#FF8C00', size=9), name="Actual Data"))

# خط الانحدار الحالي
x_line = np.linspace(-10, 10, 100)
y_line = w * x_line + b
fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', 
                         line=dict(color='#1E90FF', width=4), name="Model Prediction"))

# ضبط التنسيق والأبعاد (600x250)
fig.update_layout(
    width=600,
    height=250,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(range=[-10, 10], zeroline=True, gridcolor='#eeeeee'),
    yaxis=dict(range=[-15, 15], zeroline=True, gridcolor='#eeeeee'),
    showlegend=False,
    template="plotly_white"
)

st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
