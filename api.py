"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- SME monthly revenue prediction
    2. Classification   -- SME credit risk category
    3. Clustering       -- Mall customer segment assignment
    4. Association Rule -- Product recommendation from a cart
       Mining (ARM)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "best_revenue_pipeline.joblib")
)

def load_optional(filename):
    """Load a sample artifact that its owner has not replaced yet.
    Returns None instead of crashing the whole API if it cannot load."""
    try:
        return joblib.load(os.path.join(MODEL_DIR, filename))
    except Exception as exc:
        print(f"WARNING: could not load {filename}: {exc}")
        return None


classification_pipeline = load_optional("distress_risk_pipeline.joblib")
kmeans_pipeline = load_optional("kmeans_cluster_pipeline.joblib")
clustering_artifact = load_optional("kmeans_mall_customer_segmentation.joblib")
arm_artifact = load_optional("apriori_recommendation_rules.joblib")

clustering_scaler = clustering_artifact["scaler"] if clustering_artifact else None
clustering_kmeans = clustering_artifact["kmeans"] if clustering_artifact else None
recommendation_lookup = arm_artifact["recommendation_lookup"] if arm_artifact else None
sku_lookup = arm_artifact["sku_lookup"] if arm_artifact else None

print("Regression model loaded. Other artifacts load only if compatible.")


REGRESSION_FEATURES = list(regression_pipeline.feature_names_in_)


def bad_request(message):
    """Return a consistent JSON error for invalid input."""
    return jsonify({"error": message}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "regression_model_loaded": True})


# 1. REGRESSION -- predict daily revenue (KES) for a retail store

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/retail-revenue-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"store_format": "Supermarket", "location_type": "Mall",
#             "delivery_available": "Yes", "store_size_category": "Medium",
#             "store_age_years": 0.5, "store_size_sqm": 250.0,
#             "promotions_active": 2, "footfall_count": 450.0,
#             "online_order_count": 28.0, "avg_basket_size_kes": 950.0}'
@app.route("/api/v1/models/retail-revenue-regressor/predictions", methods=["POST"])
def predict_revenue():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return bad_request("Request body must be a JSON object.")
    missing = [c for c in REGRESSION_FEATURES if c not in payload]
    if missing:
        return bad_request("Missing required fields: " + ", ".join(missing))
    input_row = pd.DataFrame([payload])[REGRESSION_FEATURES]

    try:
        predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    except ValueError as exc:
        return bad_request(f"Could not score this input: {exc}")
    # The pipeline predicts log1p(revenue); np.expm1 converts back to KES.
    predicted_revenue_kes = float(np.expm1(predicted_log_revenue))

    return jsonify({
        "predicted_daily_revenue_kes": round(predicted_revenue_kes, 2),
        "model": "Ridge Regressor",
    })


# 2. CLASSIFICATION -- predict SME credit risk category (Low/Medium/High)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-credit-risk-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"sector": "Retail", "county": "Nairobi", "quarter": "Q2",
#             "loan_purpose": "Working Capital", "owner_education_level": "Secondary",
#             "business_age_years": 2.1, "num_employees": 4,
#             "monthly_revenue_kes": 850000, "loan_amount_kes": 150000,
#             "loan_term_months": 12, "interest_rate_pct": 16.5,
#             "collateral_value_kes": null, "debt_to_income_ratio": 0.62,
#             "credit_score": 590, "num_previous_loans": 1,
#             "missed_payments_count": 3, "online_presence_score": 40.0,
#             "customer_satisfaction_score": 3.4, "competitor_density": 6}'
#
#
# Note: collateral_value_kes is null here because "Working Capital" loans are
# more often unsecured (structural missingness, not missing at random) --
# again, the pipeline's imputer + missingness indicator already handles this
# internally; this route does not need special code for it.
# 2. CLASSIFICATION -- predict corporate financial distress risk

CLASSIFICATION_FEATURES = [
    "industry_sector",
    "listing_status",
    "credit_rating_band",
    "audit_going_concern_flag",
    "years_in_operation",
    "num_employees",
    "debt_to_equity_ratio",
    "current_ratio",
    "interest_coverage_ratio",
    "revenue_growth_pct",
    "net_profit_margin_pct",
    "working_capital_to_assets_ratio",
    "macroeconomic_interest_rate_pct",
]
@app.route("/api/v1/models/corporate-financial-distress-classifier/predictions", methods=["POST"])
def predict_distress_risk():
    if classification_pipeline is None:
        return jsonify({"error": "Classification model not available"}), 503

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return bad_request("Request body must be a JSON object.")

    missing = [c for c in CLASSIFICATION_FEATURES if c not in payload]

    if missing:
        return bad_request("Missing required fields: " + ", ".join(missing))

    input_row = pd.DataFrame([payload])[CLASSIFICATION_FEATURES]

    try:
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    except ValueError as exc:
        return bad_request(f"Could not score this input: {exc}")

    class_labels = classification_pipeline.classes_

    return jsonify({
        "predicted_distress_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Tuned Logistic Regression",
        "note": "Decision-support only; human review required."
    })


# 3. CLUSTERING -- assign a new mall customer to an existing segment

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/mall-customer-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"Age": 27, "Annual Income (k$)": 45, "Spending Score (1-100)": 72}'
#
# Note: unlike the two Pipelines above, K-Means does not bundle its own
# scaler, so this is the one endpoint where the scaling step is applied
# explicitly (scaler.transform(...) then kmeans.predict(...)) -- exactly the
# two-object pattern the clustering notebook persisted together for this
# reason. There is still no manual encoding or feature engineering: the
# three numeric fields are used exactly as the client sends them.
@app.route("/api/v1/models/mall-customer-segmenter/predictions", methods=["POST"])
def predict_customer_segment():
    if clustering_kmeans is None:
        return jsonify({"error": "Clustering model not available yet"}), 503
    payload = request.get_json()
    input_row = pd.DataFrame([payload])

    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": "K-Means (k=5, standardized features)",
    })


# 4. ASSOCIATION RULE MINING -- recommend products for a shopping cart

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["H06", "H07"]}'
#
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["P04", "P01"]}'
#
# Note: there is no scikit-learn ".predict()" here -- the mined association
# rules were precomputed into a lookup dictionary
# (antecedent SKU-set -> ranked candidate consequents)
# during the notebook's persistence stage. This
# endpoint performs the exact same lookup+rank step a production recommender
# microservice would perform on every incoming request; it does not mine
# for the rules again.
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    if recommendation_lookup is None:
        return jsonify({"error": "Recommender model not available yet"}), 503
    payload = request.get_json()
    cart = set(payload.get("cart", []))
    top_n = int(payload.get("top_n", 3))

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "sku": str(sku),
            "product_name": sku_lookup.loc[sku, "product_name"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for sku, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# K-MEANS CUSTOMER SEGMENTATION ENDPOINT
# -----------------------------------------------------------------------------
@app.route("/api/v1/models/customer-segmentation/predict", methods=["POST"])
def predict_customer_segmentation():
    if kmeans_pipeline is None:
        return bad_request("Customer segmentation model not available")

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return bad_request("Request body must be a JSON object.")

    try:
        input_row = pd.DataFrame([payload])
        predicted_cluster = kmeans_pipeline.predict(input_row)[0]
        return jsonify({"cluster": int(predicted_cluster)})
    except Exception as e:
        return bad_request(f"Could not score this input: {e}")
if __name__ == "__main__":
    # Render and Docker supply the port through the PORT environment variable.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

   
