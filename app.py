import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# 1. إعدادات الصفحة
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .block-container { padding: 1rem !important; }
    .equation-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #FF4B4B;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .math-text { font-family: 'serif'; font-size: 1.2rem; color: #31333F; }
    </style>
    """, unsafe_allow_html=True)

# 2. البيانات الأساسية
np.random.seed(42)
x_data = np.linspace(-8, 8, 20)
y_true = 1.2 * x_data + 2 + np.random.normal(0, 1.5, size=len(x_data))

# 3. واجهة التحكم الجانبية
st.sidebar.header("إعدادات الـ Gradient Descent")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, format="%.3f")
iterations = st.sidebar.slider("عدد التكرارات", 10, 100, 50)
run_btn = st.sidebar.button("ابدأ التدريب 🚀")

# الحالة الابتدائية
if 'w' not in st.session_state: st.session_state.w = -3.0
if 'b' not in st.session_state: st.session_state.b = 0.0

# 4. منطق Gradient Descent
placeholder = st.empty()

if run_btn:
    w, b = -3.0, 0.0 # البدء من قيم عشوائية
    for i in range(iterations):
        y_pred = w * x_data + b
        error = y_pred - y_true
        
        # --- Calculus: الحساب الرياضي للمشتقات ---
        # dL/dw = (2/n) * Σ( (wx + b - y) * x )
        dw = (2/len(x_data)) * np.dot(error, x_data)
        # dL/db = (2/n) * Σ( wx + b - y )
        db = (2/len(x_data)) * np.sum(error)
        
        # تحديث الأوزان
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        
        mse = np.mean(error**2)
        
        # تحديث الرسم البياني
        with placeholder.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="equation-box">
                    <p class="math-text">الخطوة رقم: <b>{i+1}</b></p>
                    <p><b>Model:</b> y = {w:.2f}x + {b:.2f}</p>
                    <p style="color:red"><b>MSE:</b> {mse:.4f}</p>
                    <hr>
                    <small>dw: {dw:.2f} | db: {db:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_data, y=y_true, mode='markers', name="Data"))
                x_line = np.linspace(-10, 10, 100)
                fig.add_trace(go.Scatter(x=x_line, y=w*x_line+b, mode='lines', line=dict(color='#FF4B4B', width=3), name="Gradient Descent"))
                fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.05)
