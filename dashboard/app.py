# ============================================================
# dashboard/app.py
# Streamlit 交互式仪表板 — 主入口
# ============================================================

import streamlit as st
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title='信贷违约预测仪表板',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 全局阈值（侧边栏）
st.sidebar.title('📊 信贷违约预测')
st.sidebar.markdown('---')
st.sidebar.markdown('### ⚙️ 全局阈值')
st.sidebar.caption('适用于页面 2（单笔预测）和页面 4（策略模拟）')

threshold = st.sidebar.slider(
    '违约判定阈值',
    min_value=0.10, max_value=0.90,
    value=0.558, step=0.01,
    help='预测概率 ≥ 阈值时判定为违约。页面 4 实时响应，页面 2 在提交时生效。'
)
st.session_state.threshold = threshold

st.sidebar.markdown(f'当前阈值：**{threshold:.2f}**')
st.sidebar.markdown('---')
st.sidebar.caption('数据来源：阿里天池信贷违约数据集')

st.sidebar.markdown('### 📁 页面导航')

pages = {
    '模型概览': '1_模型概览.py',
    '单笔预测': '2_单笔预测.py',
    '批量分析': '3_批量分析.py',
    '业务策略模拟': '4_业务策略模拟.py',
}

page = st.sidebar.radio('选择页面', list(pages.keys()))
page_file = pages[page]
page_path = os.path.join(BASE_DIR, 'page_modules', page_file)
with open(page_path, encoding='utf-8') as f:
    code = compile(f.read(), page_path, 'exec')
    exec(code)
