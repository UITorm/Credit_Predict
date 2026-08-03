# ============================================================
# dashboard/pages/1_模型概览.py
# 页面 1：模型概览
# ============================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils import load_model

model_dict = load_model()
features_list = model_dict['features']
threshold = st.session_state.get('threshold', 0.558)

# ==================== 顶部栏 ====================

col_back, col_title = st.columns([1, 11])

with col_back:
    if st.button('返回', key='back1', use_container_width=True,
                 type='secondary'):
        st.session_state.page = 'home'
        st.rerun()

with col_title:
    st.markdown(
        '<h2 style="text-align: center; margin-top: 0;">模型概览</h2>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #888;">关键指标、特征重要性、优化历程与混淆矩阵</p>',
        unsafe_allow_html=True
    )

st.divider()

# ==================== 顶部：关键指标卡片 ====================

col1, col2, col3, col4 = st.columns(4)
col1.metric('AUC', '72.89%', help='测试集 AUC-ROC')
col2.metric('Recall', '57.04%', help='违约捕获率')
col3.metric('F1', '44.47%', help='精确率与召回率调和均值')
col4.metric('违约率', '19.91%', help='数据集整体违约比例')

st.markdown('---')

# ==================== 中部：特征重要性 + 优化历程 ====================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader('特征重要性 Top-10')
    importance_data = {
        '特征': ['subGrade', 'dti', 'monthly_burden', 'term', 'fico_mean',
                 'loanAmnt', 'balance_gap', 'regionCode', 'n6', 'n8'],
        '重要性': [0.188, 0.074, 0.073, 0.072, 0.057, 0.050, 0.040, 0.037, 0.035, 0.033]
    }
    df_imp = pd.DataFrame(importance_data)

    fig_imp = px.bar(
        df_imp, x='重要性', y='特征', orientation='h',
        title='特征重要性排名',
        color='重要性', color_continuous_scale='Reds',
        text=df_imp['重要性'].apply(lambda x: f'{x:.3f}')
    )
    fig_imp.update_traces(textposition='outside')
    fig_imp.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

with col_right:
    st.subheader('模型优化历程')
    stages = ['基线 XGBoost', 'Optuna 优化', '阈值调优', '测试集最终']
    auc_values = [72.81, 73.04, 73.04, 72.89]
    f1_values = [44.06, 44.17, 44.41, 44.47]

    fig_opt = go.Figure()
    fig_opt.add_trace(go.Scatter(
        x=stages, y=auc_values, mode='lines+markers',
        name='AUC (%)', line=dict(color='#2ecc71', width=3),
        marker=dict(size=10)
    ))
    fig_opt.add_trace(go.Scatter(
        x=stages, y=f1_values, mode='lines+markers',
        name='F1 (%)', line=dict(color='#3498db', width=3),
        marker=dict(size=10), yaxis='y2'
    ))
    fig_opt.update_layout(
        height=400,
        yaxis=dict(title='AUC (%)', range=[70, 75]),
        yaxis2=dict(title='F1 (%)', range=[42, 47], overlaying='y', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig_opt, use_container_width=True)

# ==================== 底部：混淆矩阵 ====================

st.markdown('---')
st.subheader('测试集混淆矩阵')
st.caption(f'76,609 条样本，阈值 = {threshold:.3f}')

cm_data = [[46551, 14804], [6356, 8898]]
labels_row = ['实际未违约', '实际违约']
labels_col = ['预测未违约', '预测违约']

fig_cm = px.imshow(
    cm_data,
    x=labels_col, y=labels_row,
    text_auto=True,
    color_continuous_scale='Blues',
    title='混淆矩阵'
)
fig_cm.update_layout(height=350)
st.plotly_chart(fig_cm, use_container_width=True)

# 解读
st.caption(
    f'正确拒绝（TN）：46,551 笔   |   '
    f'误拒（FP）：14,804 笔   |   '
    f'漏网（FN）：6,356 笔   |   '
    f'正确拦截（TP）：8,898 笔'
)
