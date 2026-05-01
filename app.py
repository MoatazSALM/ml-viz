import streamlit as st
import numpy as np
import plotly.graph_objects as go

# إعدادات الصفحة
st.set_page_config(page_title="Linear Function Lab", layout="centered")
st.title("🧮 Linear Function: y = (w * x) + b")

# القائمة الجانبية للتحكم
st.sidebar.header("Parameters")
weight = st.sidebar.slider("الوزن (Weight - w)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
bias = st.sidebar.slider("الانحياز (Bias - b)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)

# توليد البيانات
x = np.linspace(-10, 10, 100)
y = (weight * x) + bias

# إنشاء الرسم البياني باستخدام Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Linear Function', line=dict(color='royalblue', width=4)))

# تنسيق محاور الرسم
fig.update_layout(
    xaxis_title="Input (x)",
    yaxis_title="Output (y)",
    xaxis=dict(range=[-10, 10]),
    yaxis=dict(range=[-30, 30]),
    template="plotly_white",
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# شرح بسيط يظهر تحت الرسم
st.info(f"المعادلة الحالية: y = ({weight})x + ({bias})")