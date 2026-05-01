import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 1. إعدادات الصفحة - العرض يتناسب مع التضمين
st.set_page_config(layout="wide")

# 2. CSS مخصص لضبط الحجم (500x250) وإلغاء الفراغات
st.markdown("""
    <style>
    /* إلغاء الهوامش العلوية والجانبية */
    .block-container { 
        padding-top: 0.5rem !important; 
        padding-bottom: 0rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    /* إخفاء شريط Streamlit العلوي والسفلي */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* تنسيق السلايدرز لتبدو منظمة */
    .stSlider { margin-bottom: -15px; }
    </style>
    """, unsafe_allow_html=True)

# 3. بيانات ثابتة (Linear Regression Data)
np.random.seed(42)
x_data = np.linspace(-8, 8, 15)
y_true = 1.2 * x_data + 2 + np.random.normal(0, 1.5, size=len(x_data))

# 4. التحكم (السلايدرز تحت بعض - تحكم دقيق)
w = st.slider("W (Slope)", -4.0, 4.0, 1.0, 0.01)
b = st.slider("B (Bias)", -10.0, 10.0, 0.0, 0.1)

# الحسابات الرياضية
y_pred = w * x_data + b
mse = np.mean((y_true - y_pred)**2)

# 5. عرض المعلومات في سطر واحد لتوفير مساحة للرسم
sign = "+" if b >= 0 else "-"
st.caption(f"Equation: y = {w:.2f}x {sign} {abs(b):.1f} | Loss (MSE): {mse:.2f}")

# 6. الرسم البياني (Plotly) - مضبوط ليشغل 500x250
fig = go.Figure()

# نقاط البيانات
fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers', 
                         marker=dict(color='#FF8C00', size=8), name="Actual"))

# خط التوقع
x_line = np.linspace(-10, 10, 100)
y_line = w * x_line + b
fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', 
                         line=dict(color='#1E90FF', width=3), name="Model"))

# ضبط التنسيق النهائي ليتوافق مع الحجم المطلوب
fig.update_layout(
    width=480,  # أقل قليلاً من 500 لترك مساحة للهوامش
    height=200, # يوفر مساحة للسلايدرز في الأعلى ليكون المجموع 250
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(range=[-10, 10], zeroline=True, gridcolor='#f0f0f0'),
    yaxis=dict(range=[-15, 15], zeroline=True, gridcolor='#f0f0f0'),
    showlegend=False,
    template="plotly_white",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
