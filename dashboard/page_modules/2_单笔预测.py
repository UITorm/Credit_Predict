# ============================================================
# dashboard/page_modules/2_单笔预测.py
# 页面 2：单笔贷款违约预测
# ============================================================

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils import load_model, build_features, predict

# ==================== 顶部栏 ====================

col_back, col_title = st.columns([1, 11])

with col_back:
    if st.button('返回', key='back1', use_container_width=True,
                 type='secondary'):
        st.session_state.page = 'home'
        st.rerun()

with col_title:
    st.markdown(
        '<h2 style="text-align: center; margin-top: 0;">单笔预测</h2>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #888;">手动输入贷款信息，即时获得违约概率、风险等级和决策建议</p>',
        unsafe_allow_html=True
    )

st.divider()


# ==================== 阈值滑块（最显眼位置） ====================

st.markdown('### 违约判定阈值')
threshold = st.slider(
    '预测概率 ≥ 阈值时判定为违约',
    min_value=0.10, max_value=0.90,
    value=0.558, step=0.01,
    help='拖动调整决策阈值。阈值越低，越倾向拒绝（高 Recall）；阈值越高，越倾向通过（高 Precision）'
)
st.caption(f'当前阈值：**{threshold:.2f}**  |  低于阈值 → 建议通过  |  高于阈值 → 建议拒绝')
st.markdown('---')

# 加载模型
model_dict = load_model()
features_list = model_dict['features']

# ==================== 输入表单 ====================

with st.form('predict_form'):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**贷款信息**')
        loanAmnt = st.number_input('贷款金额 ($)', min_value=500, max_value=50000, value=15000, step=500)
        term = st.selectbox('贷款期限', ['3', '5'], index=0)
        st.markdown('**信用评级**')
        subGrade = st.selectbox('贷款等级',
                                [f'{g}{i}' for g in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] for i in range(1, 6)],
                                index=12)

    with col2:
        st.markdown('**还款能力**')
        annualIncome = st.number_input('年收入 ($)', min_value=5000, max_value=500000, value=60000, step=1000)
        dti = st.slider('债务收入比 (%)', min_value=0.0, max_value=40.0, value=20.0, step=0.5)
        employmentLength = st.slider('就业年限', min_value=0, max_value=10, value=5, step=1)
        st.markdown('**其他**')
        regionCode = st.number_input('地区编码', min_value=0, max_value=50, value=10, step=1)
        openAcc = st.number_input('未结信用额度数', min_value=0, max_value=50, value=8, step=1)

    submitted = st.form_submit_button('预测违约概率')

# ==================== 预测结果 ====================

if submitted:
    raw_input = {
        'loanAmnt': loanAmnt,
        'term': term,
        'subGrade': subGrade,
        'annualIncome': annualIncome,
        'dti': dti,
        'employmentLength': employmentLength,
        'regionCode': regionCode,
        'openAcc': openAcc,
    }

    df_features = build_features(raw_input, features_list)
    proba, decision, risk = predict(df_features, model_dict, threshold)
    proba_val = float(proba[0])
    decision_val = str(decision[0])
    risk_val = str(risk[0])

    st.markdown('---')
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=proba_val * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': '违约概率'},
            delta={'reference': threshold * 100, 'increasing': {'color': 'red'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#e74c3c' if proba_val >= threshold else '#27ae60'},
                'steps': [
                    {'range': [0, 30], 'color': '#d5f5e3'},
                    {'range': [30, 60], 'color': '#fdebd0'},
                    {'range': [60, 100], 'color': '#fadbd8'},
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 3},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown(f'### 风险等级：{risk_val}')
        st.markdown(f'### 决策建议：{decision_val}')

        if proba_val >= threshold:
            st.error(f'违约概率 {proba_val * 100:.1f}% ≥ 阈值 {threshold * 100:.1f}%')
        else:
            st.success(f'违约概率 {proba_val * 100:.1f}% < 阈值 {threshold * 100:.1f}%')

        st.markdown(f'''
        | 项目 | 数值 |
        |------|------|
        | 贷款金额 | ${loanAmnt:,} |
        | 期限 | {term} 年 |
        | 贷款等级 | {subGrade} |
        | 年收入 | ${annualIncome:,} |
        | DTI | {dti:.1f}% |
        ''')
