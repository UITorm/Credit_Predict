# ============================================================
# 模型加载、特征转换、预测
# ============================================================

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== 全局模型加载（缓存） ====================

@st.cache_resource
def load_model():
    """加载最终模型及配置"""
    model_path = os.path.join(BASE_DIR, 'models', 'final_xgb.pkl')
    model_dict = joblib.load(model_path)
    return model_dict

# ==================== 输入特征编码映射 ====================

# subGrade 有序编码：A1→1 ... G5→35
GRADE_MAP = {}
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for g in grades:
    for i in range(1, 6):
        GRADE_MAP[f'{g}{i}'] = grades.index(g) * 5 + i

# 独热编码列名映射（drop_first=True）
ONEHOT_COLS = {
    'homeOwnership_1': 'homeOwnership',
    'homeOwnership_2': 'homeOwnership',
    'verificationStatus_1': 'verificationStatus',
    'verificationStatus_2': 'verificationStatus',
    'purpose_4': 'purpose',
    'loan_quarter_2': 'loan_quarter',
    'loan_quarter_3': 'loan_quarter',
    'loan_quarter_4': 'loan_quarter',
}

# 独热编码中 category → 列名的映射
ONEHOT_MAP = {
    'homeOwnership': {
        'MORTGAGE': ['homeOwnership_1', False],  # 如果这里不确定具体 category 到列的具体对应，保留通用回退
        'OWN': ['homeOwnership_2', False],
        'RENT': [None, True],  # baseline
    },
    'verificationStatus': {
        'Not Verified': [None, True],
        'Source Verified': ['verificationStatus_1', False],
        'Verified': ['verificationStatus_2', False],
    },
    'purpose': {
        'debt_consolidation': [None, True],
        'credit_card': ['purpose_4', False],
    },
    'loan_quarter': {
        1: [None, True],
        2: ['loan_quarter_2', False],
        3: ['loan_quarter_3', False],
        4: ['loan_quarter_4', False],
    }
}


# ==================== 特征构建函数 ====================

def build_features(input_dict, features_list):
    """
    将用户原始输入转换为模型所需的 21 维特征向量。

    input_dict: dict, 原始业务字段
    features_list: list, 模型需要的 21 个特征列名
    """
    df = pd.DataFrame([input_dict])

    # --- 二值编码 ---
    if 'term' in df.columns:
        df['term'] = df['term'].map({'3': 0, '5': 1, 3: 0, 5: 1})

    # --- subGrade 有序编码 ---
    if 'subGrade' in df.columns:
        df['subGrade'] = df['subGrade'].map(GRADE_MAP).fillna(18)  # 默认 C3

    # --- 独热编码 ---
    for col_name, orig_col in ONEHOT_COLS.items():
        df[col_name] = 0  # 默认全部 0

    for orig_col, mapping in ONEHOT_MAP.items():
        if orig_col in df.columns:
            val = str(df[orig_col].iloc[0])
            for cat_val, (target_col, is_baseline) in mapping.items():
                if val == cat_val:
                    if not is_baseline and target_col:
                        df[target_col] = 1
                    break

    # --- 数值特征填充 ---
    # 对未提供的特征用合理默认值
    defaults = {
        'loanAmnt': 15000, 'interestRate': 12, 'installment': 350,
        'annualIncome': 60000, 'revolBal': 10000, 'revolUtil': 30,
        'openAcc': 8, 'totalAcc': 20, 'pubRec': 0, 'pubRecBankruptcies': 0,
        'delinquency_2years': 0, 'ficoRangeLow': 680, 'ficoRangeHigh': 700,
        'credit_history_years': 10, 'dti': 20, 'loan_year': 2020,
        'employmentLength': 5, 'regionCode': 1, 'initialListStatus': 0,
        'applicationType': 0, 'title': '', 'employmentTitle': '',
        'postCode': '', 'homeOwnership': 'RENT',
        'verificationStatus': 'Not Verified', 'purpose': 'debt_consolidation',
        'loan_quarter': 1, 'loan_month': 1,
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    # --- log 变换（与训练时一致） ---
    log10_cols = ['annualIncome']
    for c in log10_cols:
        if c in df.columns:
            df[c] = np.log10(max(df[c].iloc[0], 1))

    log1p_cols = ['revolBal', 'installment', 'credit_history_years',
                  'pubRec', 'pubRecBankruptcies', 'delinquency_2years',
                  'n0', 'n1', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8',
                  'n11', 'n12', 'n13', 'n14']
    for c in log1p_cols:
        if c in df.columns:
            df[c] = np.log1p(max(df[c].iloc[0], 0))

    # --- 特征创建 ---
    if 'ficoRangeLow' in df.columns and 'ficoRangeHigh' in df.columns:
        df['fico_mean'] = (df['ficoRangeLow'] + df['ficoRangeHigh']) / 2

    if 'openAcc' in df.columns and 'totalAcc' in df.columns:
        df['balance_gap'] = df['totalAcc'] - df['openAcc']
        df['credit_utilization'] = df['openAcc'] / df['totalAcc'].replace(0, 1)

    if 'installment' in df.columns and 'annualIncome' in df.columns:
        inc_orig = 10 ** df['annualIncome'].iloc[0]
        df['monthly_burden'] = df['installment'].iloc[0] / max(inc_orig / 12, 1)
        df['monthly_burden'] = min(df['monthly_burden'].iloc[0], 5)

    if 'dti' in df.columns:
        df['dti_risk_flag'] = (df['dti'].iloc[0] > 36).astype(int)

    if 'pubRec' in df.columns and 'pubRecBankruptcies' in df.columns:
        df['has_public_record'] = ((df['pubRec'] + df['pubRecBankruptcies']) > 0).astype(int)

    # --- 匿名特征 n 系列（默认填 0） ---
    for i in range(15):
        col = f'n{i}'
        if col not in df.columns and col in features_list:
            df[col] = 0

    # --- 频数编码列（title_freq, postCode_freq, employmentTitle_te）---
    # 简化处理：使用默认值
    for col in ['title_freq', 'postCode_freq', 'employmentTitle_te']:
        if col in features_list and col not in df.columns:
            df[col] = 0.0

    # --- 标记列 ---
    if 'n8_missing' in features_list and 'n8_missing' not in df.columns:
        df['n8_missing'] = 0

    # --- 只保留模型需要的特征 ---
    for f in features_list:
        if f not in df.columns:
            df[f] = 0

    return df[features_list]


def predict(df_features, model_dict, threshold):
    """预测并返回概率和决策"""
    xgb = model_dict['xgb']
    proba = xgb.predict_proba(df_features)[:, 1]
    decision = np.where(proba >= threshold, '建议拒绝', '建议通过')
    risk = np.where(proba < 0.3, '低风险',
                    np.where(proba < 0.6, '中风险', '高风险'))
    return proba, decision, risk


# ==================== 测试集数据加载（缓存） ====================

@st.cache_data
def load_test_data():
    """
    加载测试集特征和标签。
    """
    X_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'y_test.csv')).squeeze()
    return X_test, y_test

@st.cache_data
def get_test_predictions(X_test):
    """
    用模型预测测试集，返回概率和真实标签。
    缓存以加速页面 4 的阈值滑块交互。
    """
    model_dict = load_model()
    xgb = model_dict['xgb']
    proba = xgb.predict_proba(X_test)[:, 1]
    return proba

