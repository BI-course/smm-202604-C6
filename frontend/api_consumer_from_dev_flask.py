"""
api_consumer_from_dev_flask.py

A frontend-style client that calls each of the four prediction endpoints
exposed by api.py, running on the local Flask development server. Start
api.py first (python api.py), then run this script separately.

This mirrors api.py's four artifacts one-to-one:

    1. Regression        -- /api/v1/models/sme-revenue-regressor/predictions
    2. Classification     -- /api/v1/models/sme-credit-risk-classifier/predictions
    3. Clustering         -- /api/v1/models/mall-customer-segmenter/predictions
    4. Association Rule   -- /api/v1/models/ecommerce-recommender/predictions
       Mining (ARM)

Each call uses the same requests.post(..., json=payload, headers=headers)
pattern, the same header set, and the same try/except structure as the
single-endpoint version this file replaces -- only the URL, payload, and the
key read back out of the JSON response differ per endpoint, because each
endpoint returns a different shape (a regression figure, a class + class
probabilities, a cluster id, or a list of recommendations).
"""

import requests

BASE_URL = "http://127.0.0.1:5000"

HEADERS = {
    'Content-Type': 'application/json',  # This header tells the server that the payload is in JSON format
    'Accept': 'application/json',        # This header tells the server that the client expects a JSON response
    'User-Agent': 'ServingMLModels-Client/1.0',  # Identifies the client application (useful for backend logging/debugging)
    # 'Authorization': 'Bearer <token>',  # This can be used for token-based authentication if the API requires it
}


def call_endpoint(path, payload):
    """Send one POST request to the given API path and return the parsed
    JSON response, or None if the request failed. Centralizing the
    try/except here means each of the four calls below gets identical
    error handling without repeating it four times."""
    url = f"{BASE_URL}{path}"
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raises an HTTPError if the status code is 4xx or 5xx
        return response.json()

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e)
        if e.response is not None:
            print("Response body:", e.response.text)

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

    return None


# 1. REGRESSION -- predict monthly revenue (KES) for an SME

regression_payload = {
    "sector": "Technology", "county": "Nairobi", "quarter": "Q3",
    "business_age_years": 4.5, "num_employees": 12,
    "owner_education_level": "Secondary",
    "monthly_marketing_spend_kes": 45000,
    "avg_daily_foot_traffic": None,  # Technology sector: structurally not applicable
    "online_presence_score": 78.0, "loan_amount_kes": 250000,
    "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5,
    "competitor_density": 5, "customer_satisfaction_score": 4.2,
}

result = call_endpoint("/api/v1/models/sme-revenue-regressor/predictions", regression_payload)
if result is not None:
    print("Predicted monthly revenue (KES) =", result["predicted_monthly_revenue_kes"])

print("-" * 60)



# 2. CLASSIFICATION -- predict SME credit risk category (Low/Medium/High)

classification_payload = {
    "sector": "Retail", "county": "Nairobi", "quarter": "Q2",
    "loan_purpose": "Working Capital", "owner_education_level": "Secondary",
    "business_age_years": 2.1, "num_employees": 4,
    "monthly_revenue_kes": 850000, "loan_amount_kes": 150000,
    "loan_term_months": 12, "interest_rate_pct": 16.5,
    "collateral_value_kes": None,  # Working Capital loans are more often unsecured
    "debt_to_income_ratio": 0.62,
    "credit_score": 590, "num_previous_loans": 1,
    "missed_payments_count": 3, "online_presence_score": 40.0,
    "customer_satisfaction_score": 3.4, "competitor_density": 6,
}

result = call_endpoint("/api/v1/models/sme-credit-risk-classifier/predictions", classification_payload)
if result is not None:
    print("Predicted credit risk category =", result["predicted_credit_risk_category"])
    print("Class probabilities =", result["class_probabilities"])

print("-" * 60)



# 3. CLUSTERING -- assign a new mall customer to an existing segment

clustering_payload = {
    "Age": 27,
    "Annual Income (k$)": 45,
    "Spending Score (1-100)": 72,
}

result = call_endpoint("/api/v1/models/mall-customer-segmenter/predictions", clustering_payload)
if result is not None:
    print("Assigned cluster =", result["assigned_cluster"])

print("-" * 60)



# 4. ASSOCIATION RULE MINING -- recommend products for a shopping cart

recommendation_payload = {
    "cart": ["P04", "P01"],
    "top_n": 3,
}

result = call_endpoint("/api/v1/models/ecommerce-recommender/predictions", recommendation_payload)
if result is not None:
    print("Cart =", result["cart"])
    print("Recommendations =", result["recommendations"])