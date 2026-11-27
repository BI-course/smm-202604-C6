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

from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parent / "model"

# Load all four artifacts once when the API starts.
regression_pipeline = joblib.load(
    MODEL_DIR / "best_revenue_pipeline.joblib"
)

classification_pipeline = joblib.load(
    MODEL_DIR / "distress_risk_pipeline.joblib"
)

# The bundle holds the scaler, the K-Means model, the 4 feature names and the
# human-readable segment names, all saved together.
clustering_bundle = joblib.load(
    MODEL_DIR / "customer_segmentation_bundle.joblib"
)

arm_artifact = joblib.load(
    MODEL_DIR / "recommendation_rules_pipeline.joblib"
)

# The recommender artifact is a dictionary, not a sklearn Pipeline.
recommendation_lookup = arm_artifact["recommendation_lookup"]
item_lookup = arm_artifact["item_lookup"]

print("All four model artifacts loaded successfully.")


def bad_request(message):
    """Return a consistent JSON error (HTTP 400) for invalid input."""
    return jsonify({"error": message}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": {
            "regression": regression_pipeline is not None,
            "classification": classification_pipeline is not None,
            "clustering": clustering_bundle is not None,
            "recommender": recommendation_lookup is not None
        }
    }), 200

# 1. REGRESSION -- predict daily revenue (KES) for a retail store

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/retail-revenue-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"store_format": "Supermarket", "location_type": "Mall",
#             "delivery_available": "Yes", "store_size_category": "Medium",
#             "store_age_years": 0.5, "store_size_sqm": 250.0,
#             "promotions_active": 2, "footfall_count": 450.0,
#             "online_order_count": 28.0, "avg_basket_size_kes": 950.0}'
REGRESSION_FEATURES = list(regression_pipeline.feature_names_in_)

@app.route(
    "/api/v1/models/retail-revenue-regressor/predictions",
    methods=["POST"]
)
def predict_revenue():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return bad_request("Request body must be a JSON object.")

    missing = [f for f in REGRESSION_FEATURES if f not in payload]
    if missing:
        return bad_request("Missing required fields: " + ", ".join(missing))

    input_row = pd.DataFrame([payload])[REGRESSION_FEATURES]

    try:
        # JSON whole numbers (e.g. 28) arrive as ints, but the pipeline was fitted on floats.
        input_row["online_order_count"] = input_row["online_order_count"].astype("float64")
        predicted_log_revenue = regression_pipeline.predict(input_row)[0]
        predicted_revenue_kes = float(np.expm1(predicted_log_revenue))
    except (ValueError, TypeError) as exc:
        return bad_request(f"Could not score this input: {exc}")

    return jsonify({
        "predicted_daily_revenue_kes": round(predicted_revenue_kes, 2),
        "model": "Ridge Regressor"
    }), 200


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
    "macroeconomic_interest_rate_pct"
]

@app.route(
    "/api/v1/models/corporate-financial-distress-classifier/predictions",
    methods=["POST"]
)
def predict_distress_risk():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return bad_request("Request body must be a JSON object.")

    missing = [f for f in CLASSIFICATION_FEATURES if f not in payload]
    if missing:
        return bad_request("Missing required fields: " + ", ".join(missing))

    input_row = pd.DataFrame([payload])[CLASSIFICATION_FEATURES]

    try:
        predicted_class = classification_pipeline.predict(input_row)[0]
        probabilities = classification_pipeline.predict_proba(input_row)[0]
    except (ValueError, TypeError) as exc:
        return bad_request(f"Could not score this input: {exc}")

    return jsonify({
        "predicted_distress_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(
                classification_pipeline.classes_,
                probabilities
            )
        },
        "model": "Tuned Logistic Regression",
        "note": "Decision-support only; human review required."
    }), 200

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
CLUSTERING_FEATURES = clustering_bundle["features"]
# ['recency_days', 'frequency_per_year', 'avg_order_value_kes', 'num_categories_purchased']

@app.route(
    "/api/v1/models/customer-segmentation/predict",
    methods=["POST"]
)
def predict_customer_segmentation():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return bad_request("Request body must be a JSON object.")

    missing = [f for f in CLUSTERING_FEATURES if f not in payload]
    if missing:
        return bad_request("Missing required fields: " + ", ".join(missing))

    input_row = pd.DataFrame([payload])[CLUSTERING_FEATURES]

    try:
        input_row = input_row.astype("float64")
        # K-Means was trained on standardized data, so scale first, then predict.
        scaled_row = clustering_bundle["scaler"].transform(input_row)
        raw_cluster = int(clustering_bundle["kmeans"].predict(scaled_row)[0])
    except (ValueError, TypeError) as exc:
        return bad_request(f"Could not score this input: {exc}")

    # Translate K-Means' raw label into the numbered segment used in the report.
    cluster_id = clustering_bundle["label_map"][raw_cluster]

    return jsonify({
        "cluster": cluster_id,
        "segment_name": clustering_bundle["segment_names"][cluster_id],
        "model": "K-Means Customer Segmentation"
    }), 200


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
@app.route(
    "/api/v1/models/content-recommender/recommendations",
    methods=["POST"]
)
def recommend_content():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return bad_request("Request body must be a JSON object.")

    watched_items = payload.get("watched_items")

    if not isinstance(watched_items, list) or not watched_items:
        return bad_request(
            "'watched_items' must be a non-empty list of show IDs."
        )

    if not all(isinstance(item, str) for item in watched_items):
        return bad_request("Every watched item ID must be a string.")

    # Confirm the IDs exist in the recommender's item lookup.
    known_items = set(item_lookup.index.astype(str))
    unknown_items = set(watched_items) - known_items

    if unknown_items:
        return bad_request(
            "Unknown show IDs: " + ", ".join(sorted(unknown_items))
        )

    try:
        top_n = int(payload.get("top_n", 3))
    except (ValueError, TypeError):
        return bad_request("'top_n' must be an integer.")

    if not 1 <= top_n <= 10:
        return bad_request("'top_n' must be between 1 and 10.")

    watched_set = set(watched_items)
    candidates = {}

    # Find rules whose antecedents are contained in the watch history.
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(watched_set):
            for show_id, confidence, lift in consequent_list:
                show_id = str(show_id)

                if show_id not in watched_set:
                    # Keep the strongest confidence for each candidate.
                    if (
                        show_id not in candidates
                        or confidence > candidates[show_id][0]
                    ):
                        candidates[show_id] = (confidence, lift)

    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True
    )[:top_n]

    recommendations = []

    for show_id, (confidence, lift) in ranked:
        recommendations.append({
            "show_id": show_id,
            "item_name": item_lookup.loc[show_id, "item_name"],
            "confidence": round(float(confidence), 4),
            "lift": round(float(lift), 4)
        })

    return jsonify({
        "watched_items": sorted(watched_set),
        "recommendations": recommendations,
        "message": (
            "Recommendations generated."
            if recommendations
            else "No matching recommendation rules were found."
        ),
        "model": "Apriori Association Rules"
    }), 200


if __name__ == "__main__":
    # Render and Docker supply the port through the PORT environment variable.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)