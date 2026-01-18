import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
import warnings

# 忽略部分 sklearn 版本警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 多语言配置字典 (Translation Dictionary)
# ==========================================
TRANSLATIONS = {
    "cn": {
        "page_title": "EFTM 多模型预测系统",
        "main_title": "🔬 污水厂AAO工艺智能预测系统",
        "sub_title": "基于 **CatBoost, XGBoost, LightGBM, AdaBoost** 集成模型预测。",
        "sec1_title": "### 1. 进水与时间",
        "date_label": "📅 日期",
        "time_label": "⏰ 时间",
        "inflow_label": "💧 进水量 (m³)",
        "sec2_title": "### 2. 厌氧区",
        "ana_do_n": "厌氧池北溶解氧",
        "ana_orp_s": "厌氧池南 ORP",
        "ana_orp_n": "厌氧池北 ORP",
        "sec3_title": "### 3. 缺氧区",
        "anox_ss_s": "🧪 缺氧池南污泥浓度",
        "sec4_title": "### 4. 好氧区",
        "aero_do_s": "好氧南溶解氧",
        "aero_ss_s": "好氧南污泥浓度",
        "aero_orp_s": "好氧南 ORP",
        "aero_ss_n": "好氧北污泥浓度",
        "aero_orp_n": "好氧北 ORP",
        "aero_ph_s": "好氧南 pH",
        "aero_ph_n": "好氧北 pH",
        "btn_predict": "🔍 开始预测",
        "res_title": "预测结果：好氧池北溶解氧",
        "res_unit": "mg/L",
        "btn_download": "📥 导出结果",
        "load_success": "所有模型加载成功",
        "load_fail": "⚠️ 模型加载失败: "
    },
    "en": {
        "page_title": "EFTM Prediction System",
        "main_title": "🔬 WWTP AAO Process Intelligent Prediction",
        "sub_title": "Prediction based on **CatBoost, XGBoost, LightGBM, AdaBoost** Ensemble.",
        "sec1_title": "### 1. Inflow & Time Parameters",
        "date_label": "📅 Date",
        "time_label": "⏰ Time",
        "inflow_label": "💧 Inflow Volume (m³)",
        "sec2_title": "### 2. Anaerobic Zone",
        "ana_do_n": "Anaerobic North DO",
        "ana_orp_s": "Anaerobic South ORP",
        "ana_orp_n": "Anaerobic North ORP",
        "sec3_title": "### 3. Anoxic Zone",
        "anox_ss_s": "🧪 Anoxic South MLSS",
        "sec4_title": "### 4. Aerobic Zone",
        "aero_do_s": "Aerobic South DO",
        "aero_ss_s": "Aerobic South MLSS",
        "aero_orp_s": "Aerobic South ORP",
        "aero_ss_n": "Aerobic North MLSS",
        "aero_orp_n": "Aerobic North ORP",
        "aero_ph_s": "Aerobic South pH",
        "aero_ph_n": "Aerobic North pH",
        "btn_predict": "🔍 Run Prediction",
        "res_title": "Prediction Result: Aerobic North DO",
        "res_unit": "mg/L",
        "btn_download": "📥 Download Results (CSV)",
        "load_success": "All models loaded successfully",
        "load_fail": "⚠️ Model loading failed: "
    }
}

# ==========================================
# 1. 页面基本配置与语言选择
# ==========================================
st.set_page_config(
    page_title="EFTM Multi-Model System",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自定义 CSS 样式 (修改版：Arial + 加粗)
st.markdown("""
    <style>
    /* 全局字体设置为 Arial */
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
    }

    /* 主容器样式 */
    .stApp {
        max-width: 750px;
        margin: auto;
        background-color: #eef6ff;
        padding: 1rem 2rem 3rem 2rem;
        font-family: 'Arial', sans-serif;
        font-weight: bold; /* 全局加粗 */
    }

    /* 强制所有文字、标签、段落加粗 */
    p, label, span, div, input {
        font-weight: bold !important;
    }

    /* 标题样式 */
    h1 {
        color: #1565c0;
        font-weight: 900 !important; /* 特粗 */
        font-size: 2.2rem;
        font-family: 'Arial', sans-serif;
    }
    .stMarkdown h3 {
        color: #0d47a1;
        border-bottom: 2px solid #90caf9;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
        font-size: 1.3rem;
        font-weight: 800 !important;
        font-family: 'Arial', sans-serif;
    }

    /* 按钮样式 */
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold !important;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }

    /* 结果框样式 */
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
        font-weight: 900 !important;
        color: #2e7d32;
        font-family: 'Arial', sans-serif;
    }

    /* 语言切换按钮样式微调 */
    div[data-testid="stRadio"] > label {
        display: none;
    }
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        justify-content: flex-end;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- 语言切换 (放置在主界面顶部) ---
col_blank, col_lang = st.columns([3, 1])
with col_lang:
    lang_option = st.radio(
        "语言选择",
        ["中文", "English"],
        index=0,
        horizontal=True
    )

current_lang = "cn" if lang_option == "中文" else "en"
t = TRANSLATIONS[current_lang]


# ==========================================
# 2. 集成模型类 (保留核心逻辑)
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

            return True, "Success"
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
            raise RuntimeError(f"Calculation Error: {str(e)}")


# 初始化
ensemble = EFTMEnsembleModel()
status, msg = ensemble.load_models()

# ==========================================
# 3. 界面逻辑
# ==========================================

st.title(t["main_title"])
st.markdown(t["sub_title"])

if not status:
    # 错误信息显示
    st.error(f"{t['load_fail']} {msg}")

# --- 表单输入 ---
with st.form("prediction_form"):
    st.markdown(t["sec1_title"])
    col1, col2, col3 = st.columns(3)
    with col1:
        date_input = st.date_input(t["date_label"], datetime.date.today())
    with col2:
        time_input = st.time_input(t["time_label"], datetime.datetime.now().time())
    with col3:
        inflow = st.number_input(t["inflow_label"], value=1117.0, step=10.0, format="%.1f")

    st.markdown(t["sec2_title"])
    c1, c2, c3 = st.columns(3)
    with c1:
        ana_do_n = st.number_input(t["ana_do_n"], value=0.20, step=0.01, format="%.2f")
    with c2:
        ana_orp_s = st.number_input(t["ana_orp_s"], value=-436.0, step=1.0, format="%.1f")
    with c3:
        ana_orp_n = st.number_input(t["ana_orp_n"], value=-461.5, step=1.0, format="%.1f")

    st.markdown(t["sec3_title"])
    anox_ss_s = st.number_input(t["anox_ss_s"], value=3408.0, step=10.0, format="%.1f")

    st.markdown(t["sec4_title"])
    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        aero_do_s = st.number_input(t["aero_do_s"], value=1.11, step=0.01)
        aero_ss_s = st.number_input(t["aero_ss_s"], value=1165.0, step=10.0)
    with ac2:
        aero_orp_s = st.number_input(t["aero_orp_s"], value=124.5, step=1.0)
        aero_ss_n = st.number_input(t["aero_ss_n"], value=2159.0, step=10.0)
    with ac3:
        aero_orp_n = st.number_input(t["aero_orp_n"], value=155.5, step=1.0)
        aero_ph_s = st.number_input(t["aero_ph_s"], value=6.9, step=0.1)
    with ac4:
        aero_ph_n = st.number_input(t["aero_ph_n"], value=6.9, step=0.1)
        st.write("")

    submit_btn = st.form_submit_button(t["btn_predict"])

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
    data_dict = {
        # 传感器 (Key 保持中文)
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
            <div style="color: #455a64; font-size: 1.1rem; font-weight: bold; font-family: 'Arial', sans-serif;">{t['res_title']}</div>
            <div class="result-value">{prediction:.4f} <span style="font-size:1rem; color:#666;">{t['res_unit']}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # 2. 导出 CSV
        export_df = input_df.copy()
        export_df['Predicted_Aerobic_North_DO'] = prediction
        export_df = export_df.loc[:, ~export_df.columns.duplicated()]

        csv = export_df.to_csv(index=False).encode('utf-8-sig')

        st.download_button(
            t["btn_download"],
            csv,
            "prediction_aerobic_north_do.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
