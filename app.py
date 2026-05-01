import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# 1. إعدادات الصفحة والهوية البصرية
st.set_page_config(layout="wide", page_title="Gradient Descent Visualizer")

st.markdown("""
    <style>
    /* تحسين الخلفية والخطوط */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #fcfcfd;
    }

    /* حاوية البطاقة العصرية */
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #eff0f1;
        text-align: center;
        margin-bottom: 10px;
    }

    .stat-label { color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .stat-value { color: #1e293b; font-size: 1.5rem; font-weight: 800; }
    
    /* تنسيق شريط الجانب */
    .stSidebar { background-color: #ffffff; border-right: 1px solid #f1f5f9; }
    
    /* إخفاء القوائم الافتراضية */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. توليد البيانات
np.random.seed(42)
x_data = np.linspace(-8, 8, 25)
y_true = 1.2 * x_data + 2 + np.random.normal(0, 1.8, size=len(x_data))

# 3. القائمة الجانبية (Sidebar) بتصميم أنيق
with st.sidebar:
    st.title("⚙️ الإعدادات")
    st.markdown("---")
    lr = st.slider("معدل التعلم (Learning Rate)", 0.001, 0.2, 0.05, format="%.3f")
    epochs = st.slider("عدد الخطوات (Epochs)", 10, 150, 60)
    st.markdown("---")
    run_btn = st.button("تحديث النموذج 🔄", use_container_width=True, type="primary")

# 4. منطقة العرض الرئيسية
st.title("🧠 Gradient Descent Visualizer")
st.caption("ملاحظة التغير اللحظي في الأوزان بناءً على مشتقات دالة التكلفة")

# حقول فارغة للتحديث الحي
metrics_placeholder = st.empty()
plot_placeholder = st.empty()

# 5. منطق الخوارزمية
w, b = -4.0, -5.0  # نقطة البداية

if run_btn:
    for i in range(epochs + 1):
        # حساب التوقعات والخطأ
        y_pred = w * x_data + b
        error = y_pred - y_true
        mse = np.mean(error**2)
        
        # Calculus: حساب الميل (Gradients)
        dw = (2/len(x_data)) * np.dot(error, x_data)
        db = (2/len(x_data)) * np.sum(error)
        
        # تحديث المقاييس في الأعلى
        with metrics_placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="stat-card"><div class="stat-label">Epoch</div><div class="stat-value">{i}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-card"><div class="stat-label">Weight (w)</div><div class="stat-value">{w:.2f}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="stat-card"><div class="stat-label">Bias (b)</div><div class="stat-value">{b:.2f}</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="stat-card"><div class="stat-label">Loss (MSE)</div><div class="stat-value" style="color:#ef4444">{mse:.2f}</div></div>', unsafe_allow_html=True)

        # تحديث الرسم البياني بنسبة 16:9
        with plot_placeholder.container():
            fig = go.Figure()

            # البيانات الأصلية
            fig.add_trace(go.Scatter(
                x=x_data, y=y_true, mode='markers',
                marker=dict(color='#94a3b8', size=10, opacity=0.6, line=dict(width=1, color='white')),
                name="Actual Data"
            ))

            # خط التوقعات
            x_range = np.array([-10, 10])
            fig.add_trace(go.Scatter(
                x=x_range, y=w * x_range + b, mode='lines',
                line=dict(color='#6366f1', width=5),
                name="Model Prediction"
            ))

            fig.update_layout(
                width=1000,
                height=562, # تحقيق نسبة 16:9 (1000/562.5)
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[-10, 10], showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#e2e8f0'),
                yaxis=dict(range=[-15, 15], showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#e2e8f0'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # خطوة التحديث (Calculus Step)
        w -= lr * dw
        b -= lr * db
        
        time.sleep(0.03)

else:
    st.info("اضغط على زر 'تحديث النموذج' لمشاهدة السحر الرياضي يبدأ! ✨")
