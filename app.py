import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 1. إعدادات الصفحة
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .block-container { 
        padding: 0.5rem !important; 
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* تنسيق صندوق المعادلة */
    .equation-box {
        background-color: #f1f3f4;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 10px;
        border-left: 5px solid #1E90FF;
    }
    .eq-text {
        font-family: 'Courier New', Courier, monospace;
        font-size: 22px !important;
        font-weight: bold;
        color: #2c3e50;
    }
    .var-w { color: #1E90FF; } /* أزرق يطابق الخط */
    .var-b { color: #FF8C00; } /* برتقالي يطابق النقاط */
    </style>
    """, unsafe_allow_html=True)

# 2. البيانات
np.random.seed(42)
x_data = np.linspace(-8, 8, 15)
y_true = 1.2 * x_data + 2 + np.random.normal(0, 1.5, size=len(x_data))

# 3. التحكم (تحت بعض)
w = st.slider("Weight (w)", -4.0, 4.0, 1.0, 0.01)
b = st.slider("Bias (b)", -10.0, 10.0, 0.0, 0.1)

y_pred = w * x_data + b
mse = np.mean((y_true - y_pred)**2)

# 4. عرض المعادلة بشكل واضح جداً (HTML)
sign = "+" if b >= 0 else "-"
st.markdown(f"""
    <div class="equation-box">
        <span class="eq-text">
            y = <span class="var-w">{w:.2f}</span>x {sign} <span class="var-b">{abs(b):.1f}</span>
        </span>
        <br>
        <span style="color: #d32f2f; font-size: 14px;">Loss (MSE): {mse:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

# 5. الرسم البياني (Plotly)
fig = go.Figure()

fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers', 
                         marker=dict(color='#FF8C00', size=10), name="Actual"))

x_line = np.linspace(-10, 10, 100)
y_line = w * x_line + b
fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', 
                         line=dict(color='#1E90FF', width=4), name="Model"))

fig.update_layout(
    width=480,
    height=220,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(range=[-10, 10], zeroline=True, gridcolor='#f0f0f0'),
    yaxis=dict(range=[-15, 15], zeroline=True, gridcolor='#f0f0f0'),
    showlegend=False,
    template="plotly_white"
)

st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
