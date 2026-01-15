import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
import warnings

# 忽略部分 sklearn 版本警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 页面基本配置
# ==========================================
st.set_page_config(
    page_title="EFTM Multi-Model Prediction System",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自定义 CSS 样式
st.markdown("""
    <style>
    .stApp {
        max-width: 750px;
        margin: auto;
        background-color: #eef6ff;
        padding: 1rem 2rem 3rem 2rem;
    }
    h1 {
        color: #1565c0;
        font-weight: 700;
        font-size: 2.2rem;
    }
    .stMarkdown h3 {
        color: #0d47a1;
        border-bottom: 2px solid #90caf9;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
        font-size: 1.3rem;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    .result-box {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. 集成模型类 (保留智能匹配逻辑)
# ==========================================

class EFTMEnsembleModel:
    def __init__(self):
        self.weights = {
            'cb': 0.385412, 'xgb': 0.294103, 'lgbm': 0.211438, 'ab': 0.109047
        }
        self.models = {}
        self.feature_names = []

    def load_models(self):
        """加载模型并清洗特征名"""
        try:
            self.models['cb'] = joblib.load("model_cb.pkl")
            self.models['xgb'] = joblib.load("model_xgb.pkl")
            self.models['lgbm'] = joblib.load("model_lgbm.pkl")
            self.models['ab'] = joblib.load("model_ab.pkl")

            # 获取特征名，去空格并转字符串
            for m_name in ['lgbm', 'xgb', 'ab', 'cb']:
                model = self.models[m_name]
                if hasattr(model, 'feature_names_in_'):
                    self.feature_names = [str(x).strip() for x in model.feature_names_in_]
                    break
                elif hasattr(model, 'feature_name'):
                    self.feature_names = [str(x).strip() for x in model.feature_name()]
                    break

            return True, "所有模型加载成功"
        except Exception as e:
            return False, str(e)

    def predict(self, input_df):
        """
        执行预测 (自动匹配列名)
        """
        # 1. 建立映射：小写列名 -> 原始列名
        input_df.columns = input_df.columns.astype(str)
        input_map = {col.strip().lower(): col for col in input_df.columns}

        # 2. 构建符合模型顺序的数据
        final_df = pd.DataFrame()

        if self.feature_names:
            for req_col in self.feature_names:
                req_lower = str(req_col).strip().lower()

                if req_lower in input_map:
                    # 匹配成功：取对应数据
                    original_col = input_map[req_lower]
                    final_df[req_col] = input_df[original_col].values
                else:
                    # 匹配失败：填0 (静默处理)
                    final_df[req_col] = 0.0
        else:
            final_df = input_df.copy()

        # 3. 预测与加权
        try:
            pred_cb = self.models['cb'].predict(final_df)[0]
            pred_xgb = self.models['xgb'].predict(final_df)[0]
            pred_lgbm = self.models['lgbm'].predict(final_df)[0]
            pred_ab = self.models['ab'].predict(final_df)[0]

            final_pred = (
                    pred_cb * self.weights['cb'] +
                    pred_xgb * self.weights['xgb'] +
                    pred_lgbm * self.weights['lgbm'] +
                    pred_ab * self.weights['ab']
            )
            return final_pred
        except Exception as e:
            raise RuntimeError(f"模型计算错误: {str(e)}")


# 初始化
ensemble = EFTMEnsembleModel()
status, msg = ensemble.load_models()

# ==========================================
# 3. 界面逻辑
# ==========================================

st.title("🔬 污水厂AAO工艺智能预测系统 ")
st.markdown("基于 **CatBoost, XGBoost, LightGBM, AdaBoost** 集成模型预测。")

if not status:
    st.error(f"⚠️ 模型加载失败: {msg}")

# --- 表单输入 ---
with st.form("prediction_form"):
    st.markdown("### 1. 进水与时间 (Inflow & Time)")
    col1, col2, col3 = st.columns(3)
    with col1:
        date_input = st.date_input("📅 日期 (Date)", datetime.date.today())
    with col2:
        time_input = st.time_input("⏰ 时间 (Time)", datetime.datetime.now().time())
    with col3:
        inflow = st.number_input("💧 进水量 (m³)", value=1117.0, step=10.0, format="%.1f")

    st.markdown("### 2. 厌氧区 (Anaerobic Zone)")
    c1, c2, c3 = st.columns(3)
    with c1:
        ana_do_n = st.number_input("厌氧池北溶解氧 (DO)", value=0.20, step=0.01, format="%.2f")
    with c2:
        ana_orp_s = st.number_input("厌氧池南 ORP", value=-436.0, step=1.0, format="%.1f")
    with c3:
        ana_orp_n = st.number_input("厌氧池北 ORP", value=-461.5, step=1.0, format="%.1f")

    st.markdown("### 3. 缺氧区 (Anoxic Zone)")
    anox_ss_s = st.number_input("🧪 缺氧池南污泥浓度 (MLSS)", value=3408.0, step=10.0, format="%.1f")

    st.markdown("### 4. 好氧区 (Aerobic Zone)")
    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        aero_do_s = st.number_input("好氧南 DO", value=1.11, step=0.01)
        aero_ss_s = st.number_input("好氧南 MLSS", value=1165.0, step=10.0)
    with ac2:
        aero_orp_s = st.number_input("好氧南 ORP", value=124.5, step=1.0)
        aero_ss_n = st.number_input("好氧北 MLSS", value=2159.0, step=10.0)
    with ac3:
        aero_orp_n = st.number_input("好氧北 ORP", value=155.5, step=1.0)
        aero_ph_s = st.number_input("好氧南 pH", value=6.9, step=0.1)
    with ac4:
        # 已移除好氧北DO输入框
        aero_ph_n = st.number_input("好氧北 pH", value=6.9, step=0.1)
        st.write("")

    submit_btn = st.form_submit_button("🔍 开始预测 (Predict)")

# ==========================================
# 4. 预测与结果处理
# ==========================================

if submit_btn and status:
    # --- A. 时间特征编码 (Sin/Cos) ---
    feat_month = date_input.month
    feat_hour = time_input.hour
    feat_day = date_input.day

    month_sin = np.sin(2 * np.pi * feat_month / 12.0)
    month_cos = np.cos(2 * np.pi * feat_month / 12.0)
    day_sin = np.sin(2 * np.pi * feat_day / 31.0)
    day_cos = np.cos(2 * np.pi * feat_day / 31.0)
    hour_sin = np.sin(2 * np.pi * feat_hour / 24.0)
    hour_cos = np.cos(2 * np.pi * feat_hour / 24.0)

    # --- B. 构建 DataFrame ---
    # 包含了常见的命名格式，配合类的自动匹配功能
    data_dict = {
        # 传感器
        "进水量": [inflow],
        "厌氧池北溶解氧": [ana_do_n],
        "厌氧池南ORP": [ana_orp_s],
        "厌氧池北ORP": [ana_orp_n],
        "缺氧池南污泥浓度": [anox_ss_s],
        "好氧池南溶解氧": [aero_do_s],
        "好氧池南ORP": [aero_orp_s],
        "好氧池北ORP": [aero_orp_n],
        "好氧池南污泥浓度": [aero_ss_s],
        "好氧池北污泥浓度": [aero_ss_n],
        "好氧池南PH": [aero_ph_s],
        "好氧池北PH": [aero_ph_n],

        # 时间特征
        "month_sin": [month_sin], "Month_sin": [month_sin],
        "month_cos": [month_cos], "Month_cos": [month_cos],
        "day_sin": [day_sin], "Day_sin": [day_sin],
        "day_cos": [day_cos], "Day_cos": [day_cos],
        "hour_sin": [hour_sin], "Hour_sin": [hour_sin],
        "hour_cos": [hour_cos], "Hour_cos": [hour_cos]
    }

    input_df = pd.DataFrame(data_dict)

    try:
        # 调用预测
        prediction = ensemble.predict(input_df)

        # 1. 显示结果
        st.markdown(f"""
        <div class="result-box">
            <div style="color: #455a64; font-size: 1.1rem;">预测结果：好氧池北溶解氧 (Aerobic North DO)</div>
            <div class="result-value">{prediction:.4f} <span style="font-size:1rem; color:#666;">mg/L</span></div>
        </div>
        """, unsafe_allow_html=True)

        # 2. 导出 CSV
        export_df = input_df.copy()
        export_df['Predicted_Aerobic_North_DO'] = prediction
        # 只保留第一列同名列，避免导出时有重复的 Month_sin 等
        export_df = export_df.loc[:, ~export_df.columns.duplicated()]

        csv = export_df.to_csv(index=False).encode('utf-8-sig')

        st.download_button(
            "📥 导出结果 (Download CSV)",
            csv,
            "prediction_aerobic_north_do.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"预测错误: {e}")
