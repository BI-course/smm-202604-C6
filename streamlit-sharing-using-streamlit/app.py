"""
Streamlit app -- one tab per model (Advanced tier, Leg 5).

Serves the SAME saved artifacts as api.py and the Gradio Space. Nothing is
retrained here: models are loaded once and only .predict() is called.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# model/ sits at the REPOSITORY ROOT, one level above this folder. Building the
# path from this file's location works no matter where Streamlit is started.
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


# -----------------------------------------------------------------------
# Streamlit re-runs this whole script on every click. @st.cache_resource makes
# the artifacts load ONCE instead of on every interaction.
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    regression = joblib.load(MODEL_DIR / "best_revenue_pipeline.joblib")
    classification = joblib.load(MODEL_DIR / "distress_risk_pipeline.joblib")
    clustering = joblib.load(MODEL_DIR / "customer_segmentation_bundle.joblib")
    arm = joblib.load(MODEL_DIR / "recommendation_rules_pipeline.joblib")
    return regression, classification, clustering, arm


regression_pipeline, classification_pipeline, clustering_bundle, arm_artifact = load_artifacts()
recommendation_lookup = arm_artifact["recommendation_lookup"]
item_lookup = arm_artifact["item_lookup"]  # index = show_id, column "item_name"

# The pipelines remember the exact column order they were trained on.
REGRESSION_FEATURES = list(regression_pipeline.feature_names_in_)
CLASSIFICATION_FEATURES = list(classification_pipeline.feature_names_in_)

CLUSTER_FEATURES = clustering_bundle["features"]
# ['recency_days', 'frequency_per_year', 'avg_order_value_kes', 'num_categories_purchased']

st.set_page_config(page_title="Model Validation Demos", page_icon="\U0001F4CA")
st.title("Model Validation Demos")

tab_reg, tab_clf, tab_clu, tab_arm = st.tabs([
    "Retail Revenue Regressor", "Financial Distress Classifier",
    "Customer Segmenter", "Show Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression -- predict a retail store's daily revenue (KES)
# -----------------------------------------------------------------------
with tab_reg:
    with st.form("regression_form"):
        c1, c2 = st.columns(2)
        store_format = c1.selectbox("Store format",
                                    ["Convenience", "Hypermarket", "Kiosk", "Supermarket"], index=3)
        location_type = c2.selectbox("Location type", ["CBD", "Mall", "Residential", "Street"], index=1)
        delivery_available = c1.selectbox("Delivery available", ["Yes", "No"])
        store_size_category = c2.selectbox("Store size category", ["Small", "Medium", "Large"], index=1)
        store_age_years = c1.number_input("Store age (years)", min_value=0.0, value=0.5, step=0.5)
        store_size_sqm = c2.number_input("Store size (sqm)", min_value=1.0, value=250.0)
        promotions_active = c1.number_input("Active promotions", min_value=0, value=2, step=1)
        footfall_count = c2.number_input("Daily footfall (visitors)", min_value=0.0, value=450.0)
        online_order_count = c1.number_input("Online orders", min_value=0.0, value=28.0)
        avg_basket_size_kes = c2.number_input("Average basket size (KES)", min_value=0.0, value=950.0)
        basket_unknown = st.checkbox("Average basket size is unknown")
        submitted_reg = st.form_submit_button("Predict daily revenue")

    if submitted_reg:
        payload = {
            "store_format": store_format, "location_type": location_type,
            "delivery_available": delivery_available,
            "store_size_category": store_size_category,
            "store_age_years": store_age_years, "store_size_sqm": store_size_sqm,
            "promotions_active": promotions_active, "footfall_count": footfall_count,
            "online_order_count": online_order_count,
            # NaN (not None) is what the pipeline's imputer understands.
            "avg_basket_size_kes": np.nan if basket_unknown else avg_basket_size_kes,
        }
        input_row = pd.DataFrame([payload])[REGRESSION_FEATURES]
        predicted_log_revenue = regression_pipeline.predict(input_row)[0]
        # The model predicts log1p(revenue); expm1 converts it back to KES.
        st.success(f"Predicted daily revenue: KES {float(np.expm1(predicted_log_revenue)):,.2f}")

# -----------------------------------------------------------------------
# 2. Classification -- predict corporate financial distress risk
# -----------------------------------------------------------------------
with tab_clf:
    with st.form("classification_form"):
        c1, c2 = st.columns(2)
        industry_sector = c1.selectbox("Industry sector",
                                       ["Construction", "Energy", "Manufacturing", "Retail", "Technology"],
                                       index=2)
        listing_status = c2.selectbox("Listing status", ["Private", "Public"])
        credit_rating_band = c1.selectbox("Credit rating band",
                                          ["High Risk", "Speculative", "Investment Grade"], index=1)
        audit_going_concern_flag = c2.selectbox("Auditor going-concern flag",
                                                ["No", "Not Applicable", "Yes"])
        years_in_operation = c1.number_input("Years in operation", min_value=0.0, value=15.0)
        num_employees = c2.number_input("Number of employees", min_value=0, value=300, step=1)
        debt_to_equity_ratio = c1.number_input("Debt-to-equity ratio", value=1.5)
        current_ratio = c2.number_input("Current ratio", value=1.4)
        interest_coverage_ratio = c1.number_input("Interest coverage ratio", value=3.0)
        revenue_growth_pct = c2.number_input("Revenue growth (%)", value=5.0)
        net_profit_margin_pct = c1.number_input("Net profit margin (%)", value=6.0)
        working_capital_to_assets_ratio = c2.number_input("Working capital / total assets", value=0.15)
        macroeconomic_interest_rate_pct = c1.number_input("Macro interest rate (%)", value=10.0)
        submitted_clf = st.form_submit_button("Predict distress risk")

    if submitted_clf:
        payload = {
            "industry_sector": industry_sector, "listing_status": listing_status,
            "credit_rating_band": credit_rating_band,
            "audit_going_concern_flag": audit_going_concern_flag,
            "years_in_operation": years_in_operation, "num_employees": num_employees,
            "debt_to_equity_ratio": debt_to_equity_ratio, "current_ratio": current_ratio,
            "interest_coverage_ratio": interest_coverage_ratio,
            "revenue_growth_pct": revenue_growth_pct,
            "net_profit_margin_pct": net_profit_margin_pct,
            "working_capital_to_assets_ratio": working_capital_to_assets_ratio,
            "macroeconomic_interest_rate_pct": macroeconomic_interest_rate_pct,
        }
        input_row = pd.DataFrame([payload])[CLASSIFICATION_FEATURES]
        predicted_class = classification_pipeline.predict(input_row)[0]
        probabilities = classification_pipeline.predict_proba(input_row)[0]
        st.success(f"Predicted distress risk: {predicted_class}")
        st.bar_chart(pd.Series(probabilities, index=[str(c) for c in classification_pipeline.classes_]))

# -----------------------------------------------------------------------
# 3. Clustering -- assign a customer to a segment
#
# K-Means was trained on STANDARDIZED data, so we scale the inputs with the
# saved scaler first and only then predict. All four inputs are required.
# -----------------------------------------------------------------------
with tab_clu:
    with st.form("clustering_form"):
        c1, c2 = st.columns(2)
        recency_days = c1.number_input("Days since last purchase", min_value=0.0, value=20.0)
        frequency_per_year = c2.number_input("Purchases per year", min_value=0.0, value=20.0)
        avg_order_value_kes = c1.number_input("Average order value (KES)", min_value=0.0, value=9000.0)
        num_categories_purchased = c2.number_input("Number of categories purchased", min_value=0.0, value=6.0)
        submitted_clu = st.form_submit_button("Predict segment")

    if submitted_clu:
        input_row = pd.DataFrame([{
            "recency_days": recency_days,
            "frequency_per_year": frequency_per_year,
            "avg_order_value_kes": avg_order_value_kes,
            "num_categories_purchased": num_categories_purchased,
        }])[CLUSTER_FEATURES]
        scaled_row = clustering_bundle["scaler"].transform(input_row)
        raw_cluster = int(clustering_bundle["kmeans"].predict(scaled_row)[0])
        # Translate K-Means' raw label into the numbered segment used in the report.
        cluster_id = clustering_bundle["label_map"][raw_cluster]
        st.success(f"Segment {cluster_id}: {clustering_bundle['segment_names'][cluster_id]}")

# -----------------------------------------------------------------------
# 4. Association rules -- recommend shows for a viewer's watch list
# -----------------------------------------------------------------------
with tab_arm:
    with st.form("arm_form"):
        watched = st.multiselect(
            "Shows already watched", options=list(item_lookup.index),
            default=["S04"],
            format_func=lambda sid: f"{item_lookup.loc[sid, 'item_name']} ({sid})")
        top_n = st.number_input("How many recommendations?", min_value=1, max_value=10, value=3, step=1)
        submitted_arm = st.form_submit_button("Get recommendations")

    if submitted_arm:
        if not watched:
            st.error("Please pick at least one show.")
        else:
            watched_set = set(watched)
            candidates = {}
            for antecedent, consequents in recommendation_lookup.items():
                if antecedent.issubset(watched_set):  # the rule's "if" part is fully satisfied
                    for show_id, confidence, lift in consequents:
                        if show_id not in watched_set:
                            if show_id not in candidates or confidence > candidates[show_id][0]:
                                candidates[show_id] = (confidence, lift)
            ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:int(top_n)]
            if not ranked:
                st.warning("No rule matches this combination -- try a different set of shows.")
            else:
                for show_id, (confidence, lift) in ranked:
                    st.write(f"**{item_lookup.loc[show_id, 'item_name']}** ({show_id}) -- "
                             f"confidence {confidence:.1%}, lift {lift:.2f}")