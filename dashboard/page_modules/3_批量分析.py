# ============================================================
# dashboard/pages/3_批量分析.py
# 页面 3：批量预测
# ============================================================

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from utils import load_model, predict

st.title('📂 批量分析')
st.markdown('上传 CSV 文件，批量预测违约概率并下载结果')

# 获取全局阈值
threshold = st.session_state.get('threshold', 0.558)

# 加载模型
model_dict = load_model()
features_list = model_dict['features']

# ==================== 文件上传 ====================

uploaded_file = st.file_uploader(
    '上传包含 21 个特征列的 CSV 文件',
    type=['csv'],
    help='列名需与模型训练特征一致'
)

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    # 校验列名
    missing_cols = [c for c in features_list if c not in df_input.columns]

    if missing_cols:
        st.error(f'❌ 缺少 {len(missing_cols)} 个特征列：{missing_cols[:10]}...')
    else:
        st.success(f'✅ 已加载 {len(df_input):,} 条样本')

        # 预测
        df_features = df_input[features_list]
        proba, decision, risk = predict(df_features, model_dict, threshold)

        df_input['违约概率'] = proba
        df_input['风险等级'] = risk
        df_input['决策建议'] = decision

        # ==================== 摘要统计 ====================

        n_total = len(df_input)
        n_default = (proba >= threshold).sum()
        avg_proba = proba.mean()

        col1, col2, col3 = st.columns(3)
        col1.metric('总样本', f'{n_total:,}')
        col2.metric('预测违约数', f'{n_default:,}', f'{n_default / n_total * 100:.1f}%')
        col3.metric('平均违约概率', f'{avg_proba * 100:.1f}%')

        # ==================== 分布图 ====================

        st.markdown('---')
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader('预测概率分布')
            fig_hist = px.histogram(
                df_input, x='违约概率', nbins=40,
                color=df_input['违约概率'] >= threshold,
                color_discrete_map={True: '#e74c3c', False: '#3498db'},
                labels={'color': '是否违约'},
                title='预测违约概率分布'
            )
            fig_hist.add_vline(x=threshold, line_dash='dash', line_color='black',
                               annotation_text=f'阈值={threshold:.2f}')
            fig_hist.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            st.subheader('风险等级分布')
            risk_counts = df_input['风险等级'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={'低风险': '#27ae60', '中风险': '#f39c12', '高风险': '#e74c3c'},
                title='风险等级占比'
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        # ==================== 数据预览 ====================

        st.markdown('---')
        st.subheader('预测结果预览（前 20 条）')
        st.dataframe(
            df_input[['违约概率', '风险等级', '决策建议']].head(20),
            use_container_width=True
        )

        # ==================== 下载 ====================

        csv = df_input.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label='📥 下载完整预测结果 (CSV)',
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv'
        )
