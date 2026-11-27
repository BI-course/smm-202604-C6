"""
Gradio app -- one tab per model (Advanced tier, Leg 5).

Serves the SAME saved artifacts as api.py. Nothing is retrained here: every
model is loaded once from its .joblib file and then only .predict() is called.
"""
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Hugging Face ZeroGPU workaround (see the course notes). The `spaces`
# package only exists on Hugging Face, so the import is guarded: this same
# file also runs on your laptop.
# -----------------------------------------------------------------------
try:
    import spaces

    @spaces.GPU
    def _warm_zerogpu():
        return None

    _warm_zerogpu()
except ImportError:
    pass

# -----------------------------------------------------------------------
# Load the persisted artifacts ONCE at start-up (same rule as api.py).
# On the Space the .joblib files sit next to app.py; in the repo they live
# in ./model, so we check both places.
# -----------------------------------------------------------------------
MODEL_DIR = "." if os.path.exists("best_revenue_pipeline.joblib") else "./model"


def load_artifact(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


regression_pipeline = load_artifact("best_revenue_pipeline.joblib")
classification_pipeline = load_artifact("distress_risk_pipeline.joblib")
clustering_bundle = load_artifact("customer_segmentation_bundle.joblib")
arm_artifact = load_artifact("recommendation_rules_pipeline.joblib")

recommendation_lookup = arm_artifact["recommendation_lookup"]
item_lookup = arm_artifact["item_lookup"]  # index = show_id, column "item_name"

# The pipelines remember the exact column order they were trained on.
REGRESSION_FEATURES = list(regression_pipeline.feature_names_in_)
CLASSIFICATION_FEATURES = list(classification_pipeline.feature_names_in_)


def to_number(value, label, required=True):
    """A blank Gradio Number box arrives as None. Required boxes raise a
    friendly error; optional boxes become NaN, which the pipelines' imputers
    know how to fill (None would crash them)."""
    if value is None:
        if required:
            raise gr.Error(f"Please fill in '{label}'.")
        return np.nan
    return float(value)


# -----------------------------------------------------------------------
# 1. Regression -- predict a retail store's daily revenue (KES)
# -----------------------------------------------------------------------
def predict_revenue(store_format, location_type, delivery_available,
                    store_size_category, store_age_years, store_size_sqm,
                    promotions_active, footfall_count, online_order_count,
                    avg_basket_size_kes):
    payload = {
        "store_format": store_format,
        "location_type": location_type,
        "delivery_available": delivery_available,
        "store_size_category": store_size_category,
        "store_age_years": to_number(store_age_years, "Store age"),
        "store_size_sqm": to_number(store_size_sqm, "Store size"),
        "promotions_active": to_number(promotions_active, "Active promotions"),
        "footfall_count": to_number(footfall_count, "Footfall"),
        # These two may legitimately be blank (the pipeline imputes them).
        "online_order_count": to_number(online_order_count, "Online orders", required=False),
        "avg_basket_size_kes": to_number(avg_basket_size_kes, "Average basket", required=False),
    }
    input_row = pd.DataFrame([payload])[REGRESSION_FEATURES]
    predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    # The model predicts log1p(revenue); expm1 converts it back to KES.
    predicted_revenue_kes = float(np.expm1(predicted_log_revenue))
    return f"KES {predicted_revenue_kes:,.2f} per day"


# -----------------------------------------------------------------------
# 2. Classification -- predict corporate financial distress risk
# -----------------------------------------------------------------------
def predict_distress(industry_sector, listing_status, credit_rating_band,
                     audit_going_concern_flag, years_in_operation, num_employees,
                     debt_to_equity_ratio, current_ratio, interest_coverage_ratio,
                     revenue_growth_pct, net_profit_margin_pct,
                     working_capital_to_assets_ratio, macroeconomic_interest_rate_pct):
    payload = {
        "industry_sector": industry_sector,
        "listing_status": listing_status,
        "credit_rating_band": credit_rating_band,
        "audit_going_concern_flag": audit_going_concern_flag,
        "years_in_operation": to_number(years_in_operation, "Years in operation"),
        "num_employees": to_number(num_employees, "Employees"),
        "debt_to_equity_ratio": to_number(debt_to_equity_ratio, "Debt-to-equity"),
        "current_ratio": to_number(current_ratio, "Current ratio"),
        "interest_coverage_ratio": to_number(interest_coverage_ratio, "Interest coverage"),
        "revenue_growth_pct": to_number(revenue_growth_pct, "Revenue growth"),
        "net_profit_margin_pct": to_number(net_profit_margin_pct, "Net profit margin"),
        "working_capital_to_assets_ratio": to_number(working_capital_to_assets_ratio, "Working capital / assets"),
        "macroeconomic_interest_rate_pct": to_number(macroeconomic_interest_rate_pct, "Interest rate"),
    }
    input_row = pd.DataFrame([payload])[CLASSIFICATION_FEATURES]
    probabilities = classification_pipeline.predict_proba(input_row)[0]
    # gr.Label shows the top class and a bar for each probability.
    return {str(label): float(p)
            for label, p in zip(classification_pipeline.classes_, probabilities)}


# -----------------------------------------------------------------------
# 3. Clustering -- assign a customer to a segment
#
# The bundle saves everything the model needs: the scaler, the K-Means model,
# the 4 feature names and the segment names. K-Means was trained on
# STANDARDIZED data, so we scale the inputs first and only then predict.
# (The bundle also contains an imputer for blank values; it cannot run on the
# pinned scikit-learn version, so all four inputs are simply required.)
# -----------------------------------------------------------------------
CLUSTER_FEATURES = clustering_bundle["features"]
# ['recency_days', 'frequency_per_year', 'avg_order_value_kes', 'num_categories_purchased']


def predict_segment(recency_days, frequency_per_year, avg_order_value_kes,
                    num_categories_purchased):
    values = {
        "recency_days": to_number(recency_days, "Days since last purchase"),
        "frequency_per_year": to_number(frequency_per_year, "Purchases per year"),
        "avg_order_value_kes": to_number(avg_order_value_kes, "Average order value"),
        "num_categories_purchased": to_number(num_categories_purchased, "Categories purchased"),
    }
    input_row = pd.DataFrame([values])[CLUSTER_FEATURES]
    scaled_row = clustering_bundle["scaler"].transform(input_row)
    raw_cluster = int(clustering_bundle["kmeans"].predict(scaled_row)[0])
    # Translate K-Means' raw label into the numbered segment used in the report.
    cluster_id = clustering_bundle["label_map"][raw_cluster]
    segment_name = clustering_bundle["segment_names"][cluster_id]
    return f"Segment {cluster_id}: {segment_name}"


# -----------------------------------------------------------------------
# 4. Association rules -- recommend shows for a viewer's watch list
# -----------------------------------------------------------------------
SHOW_CHOICES = [(f"{row['item_name']} ({show_id})", show_id)
                for show_id, row in item_lookup.iterrows()]


def recommend_shows(watched_ids, top_n):
    if not watched_ids:
        raise gr.Error("Please pick at least one show.")
    watched_set = set(watched_ids)
    candidates = {}
    for antecedent, consequents in recommendation_lookup.items():
        if antecedent.issubset(watched_set):  # the rule's "if" part is fully satisfied
            for show_id, confidence, lift in consequents:
                if show_id not in watched_set:
                    if show_id not in candidates or confidence > candidates[show_id][0]:
                        candidates[show_id] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:int(top_n)]
    if not ranked:
        return "No rule matches this combination -- try a different set of shows."
    return "\n".join(
        f"{item_lookup.loc[sid, 'item_name']} ({sid}) -- "
        f"confidence {conf:.1%}, lift {lift:.2f}"
        for sid, (conf, lift) in ranked
    )


# -----------------------------------------------------------------------
# The app: one gr.Blocks, four tabs, one shared Space URL
# -----------------------------------------------------------------------
with gr.Blocks(title="Model Validation Demos") as demo:
    gr.Markdown("# Model Validation Demos\nSwitch tabs to try each model.")

    with gr.Tab("Retail Revenue Regressor"):
        gr.Interface(
            fn=predict_revenue,
            inputs=[
                gr.Dropdown(["Convenience", "Hypermarket", "Kiosk", "Supermarket"],
                            label="Store format", value="Supermarket"),
                gr.Dropdown(["CBD", "Mall", "Residential", "Street"],
                            label="Location type", value="Mall"),
                gr.Dropdown(["Yes", "No"], label="Delivery available", value="Yes"),
                gr.Dropdown(["Small", "Medium", "Large"],
                            label="Store size category", value="Medium"),
                gr.Number(label="Store age (years)", value=0.5),
                gr.Number(label="Store size (sqm)", value=250),
                gr.Number(label="Active promotions", value=2),
                gr.Number(label="Daily footfall (visitors)", value=450),
                gr.Number(label="Online orders (blank if unknown)", value=28),
                gr.Number(label="Average basket size (KES, blank if unknown)", value=950),
            ],
            outputs=gr.Textbox(label="Predicted daily revenue"),
            flagging_mode="never",
        )

    with gr.Tab("Financial Distress Classifier"):
        gr.Interface(
            fn=predict_distress,
            inputs=[
                gr.Dropdown(["Construction", "Energy", "Manufacturing", "Retail", "Technology"],
                            label="Industry sector", value="Manufacturing"),
                gr.Dropdown(["Private", "Public"], label="Listing status", value="Private"),
                gr.Dropdown(["High Risk", "Speculative", "Investment Grade"],
                            label="Credit rating band", value="Speculative"),
                gr.Dropdown(["No", "Not Applicable", "Yes"],
                            label="Auditor going-concern flag", value="No"),
                gr.Number(label="Years in operation", value=15),
                gr.Number(label="Number of employees", value=300),
                gr.Number(label="Debt-to-equity ratio", value=1.5),
                gr.Number(label="Current ratio", value=1.4),
                gr.Number(label="Interest coverage ratio", value=3.0),
                gr.Number(label="Revenue growth (%)", value=5.0),
                gr.Number(label="Net profit margin (%)", value=6.0),
                gr.Number(label="Working capital / total assets", value=0.15),
                gr.Number(label="Macro interest rate (%)", value=10.0),
            ],
            outputs=gr.Label(label="Predicted distress risk", num_top_classes=3),
            flagging_mode="never",
        )

    with gr.Tab("Customer Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[
                gr.Number(label="Days since last purchase", value=20),
                gr.Number(label="Purchases per year", value=20),
                gr.Number(label="Average order value (KES)", value=9000),
                gr.Number(label="Number of categories purchased", value=6),
            ],
            outputs=gr.Textbox(label="Predicted segment"),
            flagging_mode="never",
        )

    with gr.Tab("Show Recommender"):
        gr.Interface(
            fn=recommend_shows,
            inputs=[
                gr.CheckboxGroup(choices=SHOW_CHOICES, label="Shows already watched",
                                 value=["S04"]),
                gr.Number(label="How many recommendations?", value=3, precision=0),
            ],
            outputs=gr.Textbox(label="Recommendations"),
            flagging_mode="never",
        )

if __name__ == "__main__":
    demo.launch()