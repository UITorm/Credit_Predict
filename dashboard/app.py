# ============================================================
# dashboard/app.py
# Streamlit 交互式仪表板 — 主入口
# ============================================================

import streamlit as st
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

st.set_page_config(
    page_title='信贷违约预测仪表板',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ==================== 侧边栏 ====================

st.sidebar.title('信贷违约预测')

# 页面导航
st.sidebar.markdown('---')
st.sidebar.markdown('### 页面导航')

pages = {
    '模型概览': '1_模型概览.py',
    '单笔预测': '2_单笔预测.py',
    '批量分析': '3_批量分析.py',
    '业务策略模拟': '4_业务策略模拟.py',
}

page = st.sidebar.radio('选择页面', list(pages.keys()))

st.sidebar.markdown('---')
st.sidebar.caption('数据来源：阿里天池信贷违约数据集')

# ==================== 加载页面 ====================

page_file = pages[page]
page_path = os.path.join(BASE_DIR, 'page_modules', page_file)
with open(page_path, encoding='utf-8') as f:
    code = compile(f.read(), page_path, 'exec')
    exec(code)
