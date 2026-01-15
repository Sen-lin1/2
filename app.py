import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
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
# 2. 集成模型类定义 (已修复类型错误)
# ==========================================

class EFTMEnsembleModel:
    def __init__(self):
        # 定义权重
        self.weights = {
            'cb': 0.385412,
            'xgb': 0.294103,
            'lgbm': 0.211438,
            'ab': 0.109047
        }
        self.models = {}
        self.feature_names = None

    def load_models(self):
        """加载四个单独的模型文件"""
        try:
            self.models['cb'] = joblib.load("model_cb.pkl")
            self.models['xgb'] = joblib.load("model_xgb.pkl")
            self.models['lgbm'] = joblib.load("model_lgbm.pkl")
            self.models['ab'] = joblib.load("model_ab.pkl")

            # 尝试从其中一个模型获取特征名称，用于对齐列顺序
            # 优先尝试 LGBM 或 XGB，它们通常保留了 feature_names_in_
            for m_name in ['lgbm', 'xgb', 'ab', 'cb']:
                model = self.models[m_name]
                if hasattr(model, 'feature_names_in_'):
                    # 【关键修复】确保特征名是纯 Python 字符串列表，而不是 numpy.str_
                    self.feature_names = [str(x) for x in model.feature_names_in_]
                    break
                elif hasattr(model, 'feature_name'):  # Booster case
                    self.feature_names = [str(x) for x in model.feature_name()]
                    break

            return True, "所有模型加载成功"
        except Exception as e:
            return False, str(e)

    def predict(self, input_df):
        """
        执行加权预测
        input_df: 包含中文列名的 DataFrame
        """
        # =======================================================
        # 【关键修复】强制转换列名为标准字符串，解决混合类型错误
        # =======================================================
        input_df.columns = input_df.columns.astype(str)

        # 1. 特征对齐 (如果模型里保存了特征名，确保输入顺序一致)
        if self.feature_names is not None:
            # 找出缺失的列（主要是时间特征可能命名不一致，或模型有额外特征）
            for col in self.feature_names:
                if col not in input_df.columns:
                    # 如果是时间相关的列缺失，尝试用常见的默认值或 0
                    input_df[col] = 0

            # 严格按照模型训练时的列顺序重排
            input_df = input_df[self.feature_names]

        # 再次确保重排后的 DataFrame 列名也是纯字符串（双重保险）
        input_df.columns = input_df.columns.astype(str)

        # 2. 分别预测
        try:
            pred_cb = self.models['cb'].predict(input_df)[0]
            pred_xgb = self.models['xgb'].predict(input_df)[0]
            pred_lgbm = self.models['lgbm'].predict(input_df)[0]
            pred_ab = self.models['ab'].predict(input_df)[0]

            # 3. 加权求和
            final_pred = (
                    pred_cb * self.weights['cb'] +
                    pred_xgb * self.weights['xgb'] +
                    pred_lgbm * self.weights['lgbm'] +
                    pred_ab * self.weights['ab']
            )
            return final_pred
        except Exception as e:
            # 打印错误详情到后台以便调试
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"预测计算时发生错误: {str(e)}")


# 初始化并加载模型
ensemble = EFTMEnsembleModel()
status, msg = ensemble.load_models()

# ==========================================
# 3. 界面逻辑
# ==========================================

st.title("🔬 污水处理出水指标预测 (EFTM)")
st.markdown("基于 **CatBoost, XGBoost, LightGBM, AdaBoost** 集成模型预测。")

if not status:
    st.error(f"⚠️ 模型加载失败: {msg}\n\n请确保目录中包含 model_cb.pkl, model_xgb.pkl, model_lgbm.pkl, model_ab.pkl")

# --- 表单输入区域 ---
with st.form("prediction_form"):
    # -------------------------------------------------------
    # 1. 进水与时间 (Inflow & Time)
    # -------------------------------------------------------
    st.markdown("### 1. 进水与时间 (Inflow & Time)")
    col1, col2, col3 = st.columns(3)

    with col1:
        date_input = st.date_input("📅 日期 (Date)", datetime.date.today())
    with col2:
        time_input = st.time_input("⏰ 时间 (Time)", datetime.datetime.now().time())
    with col3:
        inflow = st.number_input("💧 进水量 (m³)", value=1117.0, step=10.0, format="%.1f")

    # -------------------------------------------------------
    # 2. 厌氧区 (Anaerobic Zone)
    # -------------------------------------------------------
    st.markdown("### 2. 厌氧区 (Anaerobic Zone)")
    c1, c2, c3 = st.columns(3)

    with c1:
        ana_do_n = st.number_input("厌氧池北溶解氧 (DO)", value=0.20, step=0.01, format="%.2f")
    with c2:
        ana_orp_s = st.number_input("厌氧池南 ORP", value=-436.0, step=1.0, format="%.1f")
    with c3:
        ana_orp_n = st.number_input("厌氧池北 ORP", value=-461.5, step=1.0, format="%.1f")

    # -------------------------------------------------------
    # 3. 缺氧区 (Anoxic Zone)
    # -------------------------------------------------------
    st.markdown("### 3. 缺氧区 (Anoxic Zone)")
    anox_ss_s = st.number_input("🧪 缺氧池南污泥浓度 (MLSS)", value=3408.0, step=10.0, format="%.1f")

    # -------------------------------------------------------
    # 4. 好氧区 (Aerobic Zone)
    # -------------------------------------------------------
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
        aero_do_n = st.number_input("好氧北 DO", value=1.85, step=0.01)
        aero_ph_n = st.number_input("好氧北 pH", value=6.9, step=0.1)

    submit_btn = st.form_submit_button("🔍 开始预测 (Predict)")

# ==========================================
# 4. 预测与结果处理
# ==========================================

if submit_btn and status:
    # --- A. 时间特征编码 ---
    feat_month = date_input.month
    feat_hour = time_input.hour
    feat_day = date_input.day
    feat_weekday = date_input.weekday()  # 0=Monday, 6=Sunday

    # --- B. 构建 DataFrame 并使用中文列名 ---
    # 这里的 Key 必须与模型训练时的列名完全一致
    data_dict = {
        # 1. 传感器数据 (Sensor Data)
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
        "好氧池南PH": [aero_ph_s],  # 注意大小写，根据经验 PH 常见大写
        "好氧池北PH": [aero_ph_n],
        "好氧池北溶解氧": [aero_do_n],

        # 2. 时间特征 (Time Features)
        # 提供多种常见格式以匹配不同模型需求
        "Month": [feat_month],
        "Hour": [feat_hour],
        "Day": [feat_day],
        "Weekday": [feat_weekday],
        "month": [feat_month],
        "hour": [feat_hour]
    }

    input_df = pd.DataFrame(data_dict)

    try:
        # 调用集成模型进行预测
        prediction = ensemble.predict(input_df)

        # 显示结果
        st.markdown(f"""
        <div class="result-box">
            <div>加权预测出水指标 / Weighted Prediction</div>
            <div class="result-value">{prediction:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        # 导出 CSV
        export_df = input_df.copy()
        export_df['Prediction_Result'] = prediction
        csv = export_df.to_csv(index=False).encode('utf-8-sig')

        st.download_button(
            "📥 导出结果 (Download CSV)",
            csv,
            "EFTM_ensemble_prediction.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"预测过程中发生错误: {e}")