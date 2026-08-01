# 信贷违约预测模型

基于机器学习的贷款违约风险预测项目。使用 80 万条真实信贷数据，完成从原始数据清洗到可部署模型的完整数据分析流程。

---

## 项目概述

信贷违约是金融机构面临的核心风险之一。本项目构建机器学习模型，在贷款审批环节预测申请人的违约概率，为贷前风控提供数据驱动的决策支持。

---

## 数据说明

| 项目 | 说明 |
|------|------|
| 来源 | 阿里天池信贷违约数据集 |
| 原始规模 | 800,000 条 × 47 列（46 个自变量 + 1 个目标变量 `isDefault`） |
| 清洗后规模 | 766,086 条 × 61 列 |
| 目标变量 | `isDefault`（0 = 未违约，1 = 违约） |
| 违约率 | 约 19.91% |
| 特征维度 | 贷款信息、信用评估、用户画像、行为特征、匿名特征（n0-n14） |

---

## 环境配置

### 方式一：pip 安装

```bash
pip install -r requirements.txt
```

### 方式二：conda 环境

```bash
conda create -n credit python=3.12
conda activate credit
pip install -r requirements.txt
```

---

## 使用方式

### 交互式仪表板

部署地址：https://zh-lee-credit-predict.streamlit.app/ （Streamlit Community Cloud）

注：Page 4(业务策略模拟)需要一段时间加载。

#### 页面功能

| 页面 | 功能 |
|------|------|
| **模型概览** | 关键指标卡片、特征重要性、优化历程、混淆矩阵 |
| **单笔预测** | 手动输入贷款信息，即时获得违约概率和决策建议 |
| **批量分析** | 上传 CSV 文件，批量预测并下载结果 |
| **业务策略模拟** | 拖拽阈值滑块，实时观察 Recall/Precision 变化 + 损失测算 |


### 加载模型进行预测

```python
import joblib
import pandas as pd

model_dict = joblib.load('models/final_xgb.pkl')
xgb = model_dict['xgb']
threshold = model_dict.get('threshold', 0.558)

# X 为 21 维特征 DataFrame
proba = xgb.predict_proba(X)[:, 1]
prediction = (proba >= threshold).astype(int)
```
