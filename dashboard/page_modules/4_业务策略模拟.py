# ============================================================
# dashboard/page_modules/4_业务策略模拟.py
# 页面 4：业务策略模拟 — 基于真实测试集
# ============================================================

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils import load_model, load_test_data, get_test_predictions
from sklearn.metrics import precision_score, recall_score, f1_score

# ==================== 顶部栏 ====================

col_back, col_title = st.columns([1, 11])

with col_back:
    if st.button('返回', key='back1', use_container_width=True,
                 type='secondary'):
        st.session_state.page = 'home'
        st.rerun()

with col_title:
    st.markdown(
        '<h2 style="text-align: center; margin-top: 0;">业务策略模拟</h2>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #888;">拖拽阈值滑块，实时观察召回率、精确率变化及业务损失测算</p>',
        unsafe_allow_html=True
    )

st.divider()


# ==================== 阈值滑块（最显眼位置） ====================

st.markdown('### 违约判定阈值')
threshold = st.slider(
    '预测概率 ≥ 阈值时判定为违约。拖动滑块实时观察指标变化',
    min_value=0.10, max_value=0.90,
    value=0.558, step=0.01,
    help='阈值越低，越倾向拒绝（高 Recall，尽量不漏坏客户）；阈值越高，越倾向通过（高 Precision，仅拒绝最明显违约者）'
)

# 参考策略
col_ref1, col_ref2, col_ref3 = st.columns(3)
col_ref1.info(f'保守策略 ≈ 0.48\n\nRecall 高，尽量不漏坏客户')
col_ref2.success(f'平衡策略 ≈ 0.56\n\nF1 最优，均衡漏网与误拒')
col_ref3.warning(f'激进策略 ≈ 0.65\n\nPrecision 优先，仅拒绝明显违约者')

st.markdown('---')

# ==================== 加载真实测试集 ====================

with st.spinner('加载测试集数据...'):
    model_dict = load_model()
    X_test, y_test = load_test_data()
    proba = get_test_predictions(X_test)

n_total = len(y_test)
default_rate_true = y_test.mean()

st.caption(f'基于真实测试集：{n_total:,} 条样本，违约率 {default_rate_true*100:.1f}%')

# ==================== 实时计算 ====================

y_pred = (proba >= threshold).astype(int)

tp = ((y_test == 1) & (y_pred == 1)).sum()
fp = ((y_test == 0) & (y_pred == 1)).sum()
fn = ((y_test == 1) & (y_pred == 0)).sum()
tn = ((y_test == 0) & (y_pred == 0)).sum()

recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
n_pred_default = tp + fp

# ==================== 指标卡片 ====================

col1, col2, col3 = st.columns(3)
col1.metric('预测违约人数', f'{n_pred_default:,}', f'{n_pred_default/n_total*100:.1f}%')
col2.metric('Recall（捕获率）', f'{recall_val*100:.1f}%')
col3.metric('Precision（准确率）', f'{precision_val*100:.1f}%')

col4, col5, col6 = st.columns(3)
col4.metric('漏网违约者 (FN)', f'{fn:,}')
col5.metric('误拒好客户 (FP)', f'{fp:,}')
col6.metric('F1 分数', f'{f1_val*100:.2f}%')

# ==================== 阈值曲线（预计算） ====================

st.markdown('---')
st.subheader('阈值 vs 指标曲线')

thresh_range = np.linspace(0.1, 0.9, 100)
recalls = []
precisions = []
f1s = []

for t in thresh_range:
    yp = (proba >= t).astype(int)
    recalls.append(recall_score(y_test, yp, zero_division=0))
    precisions.append(precision_score(y_test, yp, zero_division=0))
    f1s.append(f1_score(y_test, yp, zero_division=0))

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=thresh_range, y=precisions, mode='lines',
    name='Precision', line=dict(color='#3498db', width=2.5)
))
fig.add_trace(go.Scatter(
    x=thresh_range, y=recalls, mode='lines',
    name='Recall', line=dict(color='#e74c3c', width=2.5)
))
fig.add_trace(go.Scatter(
    x=thresh_range, y=f1s, mode='lines',
    name='F1', line=dict(color='#2ecc71', width=3)
))

fig.add_vline(
    x=threshold, line_dash='dash', line_color='black', line_width=2,
    annotation_text=f'当前阈值={threshold:.2f}',
    annotation_position='top right'
)

fig.update_layout(
    xaxis_title='阈值',
    yaxis_title='分数',
    height=400,
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# ==================== 业务测算 ====================

st.markdown('---')
st.subheader('业务影响测算')

col_input1, col_input2 = st.columns(2)
with col_input1:
    monthly_volume = st.number_input(
        '月审批量（笔）',
        min_value=1000, max_value=100000,
        value=10000, step=1000
    )
with col_input2:
    avg_loan = st.number_input(
        '平均贷款金额 ($)',
        min_value=1000, max_value=100000,
        value=15000, step=1000
    )

fn_ratio = fn / n_total
tp_ratio = tp / n_total

loss_no_model = monthly_volume * default_rate_true * avg_loan
loss_with_model = monthly_volume * fn_ratio * avg_loan
loss_reduced = loss_no_model - loss_with_model
loss_reduced_pct = loss_reduced / loss_no_model * 100 if loss_no_model > 0 else 0

col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric('不使用模型：预期违约损失', f'${loss_no_model:,.0f}')
col_r2.metric('使用模型：预期违约损失', f'${loss_with_model:,.0f}',
              f'-{loss_reduced_pct:.1f}%')
col_r3.metric('减少损失', f'${loss_reduced:,.0f}')

st.caption(
    f'测试集真实违约率 {default_rate_true*100:.1f}%，平均贷款 ${avg_loan:,}。\n'
    f'当前阈值下，模型拦截了 {tp_ratio*100:.1f}% 的真实违约者（减少损失），'
    f'仍有 {fn_ratio*100:.1f}% 的违约者漏网（剩余损失）。'
    f'此为简化测算，未计入误拒好客户的机会成本。'
)
