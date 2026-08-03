# ============================================================
# dashboard/app.py
# 信贷违约预测 — 首页路由
# ============================================================

import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title='信贷违约预测仪表板',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# ==================== 初始化 page 状态 ====================

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ==================== 极简侧边栏 ====================

st.sidebar.markdown('## 信贷违约预测')
st.sidebar.markdown('---')
st.sidebar.caption('基于机器学习的贷款违约风险预测')
st.sidebar.caption(f'AUC 72.89% | 21 特征 | XGBoost')
st.sidebar.markdown('---')
st.sidebar.caption('数据来源：阿里天池')

# ==================== 首页 ====================

if st.session_state.page == 'home':

    st.markdown(
        '<h1 style="text-align: center;">信贷违约预测</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666;">基于机器学习的贷款违约风险预测模型 — 交互式仪表板</p>',
        unsafe_allow_html=True
    )
    st.markdown('---')

    col1, col2, col3, col4 = st.columns(4)

    # 卡片 1：模型概览
    with col1:
        with st.container(border=True, height=220):
            st.markdown('### 模型概览')
            st.caption('查看模型关键指标、特征重要性排名、优化历程和混淆矩阵')
            if st.button('进入 →', key='btn1', use_container_width=True):
                st.session_state.page = 'overview'
                st.rerun()

    # 卡片 2：单笔预测
    with col2:
        with st.container(border=True, height=220):
            st.markdown('### 单笔预测')
            st.caption('手动输入贷款信息，即时获得违约概率、风险等级和决策建议')
            if st.button('进入 →', key='btn2', use_container_width=True):
                st.session_state.page = 'single'
                st.rerun()

    # 卡片 3：批量分析
    with col3:
        with st.container(border=True, height=220):
            st.markdown('### 批量分析')
            st.caption('上传 CSV 文件，批量预测违约概率并下载完整结果')
            if st.button('进入 →', key='btn3', use_container_width=True):
                st.session_state.page = 'batch'
                st.rerun()

    # 卡片 4：业务策略模拟
    with col4:
        with st.container(border=True, height=220):
            st.markdown('### 业务策略模拟')
            st.caption('拖拽阈值滑块，实时观察召回率、精确率变化及业务损失测算')
            if st.button('进入 →', key='btn4', use_container_width=True):
                st.session_state.page = 'strategy'
                st.rerun()

    # 底部信息
    st.markdown('---')
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric('测试集 AUC', '72.89%')
    col_b.metric('Recall', '57.04%')
    col_c.metric('F1', '44.47%')
    col_d.metric('特征数', '21')

# ==================== 子页面 ====================

else:
    # 根据 page 状态加载对应页面文件
    page_files = {
        'overview': '1_模型概览.py',
        'single':   '2_单笔预测.py',
        'batch':    '3_批量分析.py',
        'strategy': '4_业务策略模拟.py',
    }

    page_file = page_files.get(st.session_state.page)
    if page_file:
        page_path = os.path.join(BASE_DIR, 'page_modules', page_file)
        with open(page_path, encoding='utf-8') as f:
            code = compile(f.read(), page_path, 'exec')
            exec(code)
