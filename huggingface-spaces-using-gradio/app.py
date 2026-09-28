import gradio as gr
import joblib
import numpy as np
import pandas as pd
import spaces

# -----------------------------------------------------------------------
# Hugging Face's free tier now provisions new Gradio Spaces on ZeroGPU
# hardware only (see the note in Step 2 below). None of this app's four
# models needs a GPU -- every prediction below runs on plain scikit-learn
# arrays -- but some free-tier Spaces fail to start reliably unless at
# least one function is decorated with @spaces.GPU. This function does
# nothing GPU-related; it exists only to satisfy that platform check.
# Calling it once, here, registers it before Gradio launches below.
@spaces.GPU
def _warm_zerogpu():
    return None

_warm_zerogpu()

# -----------------------------------------------------------------------
# Load all four persisted artifacts ONCE, at import time -- the same
# principle api.py uses: load once at startup, reuse across every request
# (here, across every tab and every submit click), never inside a function.
# -----------------------------------------------------------------------
regression_pipeline = joblib.load("lasso_regressor_for_sme_revenue.joblib")
classification_pipeline = joblib.load("svc_classifier_for_sme_credit_risk.joblib")

clustering_artifact = joblib.load("kmeans_mall_customer_segmentation.joblib")
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load("apriori_recommendation_rules.joblib")
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]


# -----------------------------------------------------------------------
# 1. Regression -- predict monthly revenue (KES) for an SME
# -----------------------------------------------------------------------
def predict_revenue(sector, county, quarter, business_age_years, num_employees,
                     owner_education_level, monthly_marketing_spend_kes,
                     avg_daily_foot_traffic, online_presence_score,
                     loan_amount_kes, interest_rate_pct, inflation_rate_pct,
                     competitor_density, customer_satisfaction_score):
    payload = {
        "sector": sector, "county": county, "quarter": quarter,
        "business_age_years": business_age_years, "num_employees": num_employees,
        "owner_education_level": owner_education_level,
        "monthly_marketing_spend_kes": monthly_marketing_spend_kes,
        # A Gradio Number field cannot submit a literal null; 0 is used here
        # as a stand-in for "not applicable" when the field is left blank.
        "avg_daily_foot_traffic": avg_daily_foot_traffic or None,
        "online_presence_score": online_presence_score,
        "loan_amount_kes": loan_amount_kes, "interest_rate_pct": interest_rate_pct,
        "inflation_rate_pct": inflation_rate_pct,
        "competitor_density": competitor_density,
        "customer_satisfaction_score": customer_satisfaction_score,
    }
    input_row = pd.DataFrame([payload])
    predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    predicted_revenue_kes = float(np.expm1(predicted_log_revenue))
    return f"KES {predicted_revenue_kes:,.2f} / month"


# -----------------------------------------------------------------------
# 2. Classification -- predict SME credit risk category
# -----------------------------------------------------------------------
def predict_credit_risk(sector, county, quarter, loan_purpose, owner_education_level,
                         business_age_years, num_employees, monthly_revenue_kes,
                         loan_amount_kes, loan_term_months, interest_rate_pct,
                         collateral_value_kes, debt_to_income_ratio, credit_score,
                         num_previous_loans, missed_payments_count,
                         online_presence_score, customer_satisfaction_score,
                         competitor_density):
    payload = {
        "sector": sector, "county": county, "quarter": quarter,
        "loan_purpose": loan_purpose, "owner_education_level": owner_education_level,
        "business_age_years": business_age_years, "num_employees": num_employees,
        "monthly_revenue_kes": monthly_revenue_kes, "loan_amount_kes": loan_amount_kes,
        "loan_term_months": loan_term_months, "interest_rate_pct": interest_rate_pct,
        "collateral_value_kes": collateral_value_kes or None,
        "debt_to_income_ratio": debt_to_income_ratio, "credit_score": credit_score,
        "num_previous_loans": num_previous_loans,
        "missed_payments_count": missed_payments_count,
        "online_presence_score": online_presence_score,
        "customer_satisfaction_score": customer_satisfaction_score,
        "competitor_density": competitor_density,
    }
    input_row = pd.DataFrame([payload])
    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_
    probability_lines = "\n".join(
        f"  {label}: {prob:.1%}" for label, prob in zip(class_labels, predicted_probabilities)
    )
    return f"Predicted category: {predicted_class}\n\n{probability_lines}"


# -----------------------------------------------------------------------
# 3. Clustering -- assign a new mall customer to an existing segment
# -----------------------------------------------------------------------
def predict_segment(age, annual_income_k, spending_score):
    input_row = pd.DataFrame([{
        "Age": age,
        "Annual Income (k$)": annual_income_k,
        "Spending Score (1-100)": spending_score,
    }])
    scaled_input = clustering_scaler.transform(input_row)
    cluster = int(clustering_kmeans.predict(scaled_input)[0])
    return f"Assigned segment: Cluster {cluster}"


# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend products for a shopping cart
# -----------------------------------------------------------------------
def recommend_products(cart_text, top_n):
    cart = {sku.strip() for sku in cart_text.split(",") if sku.strip()}
    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:int(top_n)]
    if not ranked:
        return "No recommendations for this cart -- try a SKU combination seen in the training orders."

    lines = [
        f"{sku_lookup.loc[sku, 'product_name']} ({sku})  --  "
        f"confidence {confidence:.1%}, lift {lift:.2f}"
        for sku, (confidence, lift) in ranked
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------
# One gr.Blocks() app, four gr.Tab()s -- one shared Space URL, no linking
# between separately-hosted demos required.
# -----------------------------------------------------------------------
with gr.Blocks(title="BBT 4206 Model Validation Demos") as demo:
    gr.Markdown("# BBT 4206 -- Model Validation Demos\nSwitch tabs to try a different model.")

    with gr.Tab("SME Revenue Regressor"):
        gr.Interface(
            fn=predict_revenue,
            inputs=[
                gr.Textbox(label="Sector", value="Technology"),
                gr.Textbox(label="County", value="Nairobi"),
                gr.Textbox(label="Quarter", value="Q3"),
                gr.Number(label="Business Age (years)", value=4.5),
                gr.Number(label="Number of Employees", value=12),
                # These four categories are the exact set the pipeline's
                # OrdinalEncoder was fitted on -- any other string raises
                # "Found unknown categories" at predict time.
                gr.Dropdown(label="Owner Education Level",
                            choices=["Primary", "Secondary", "Tertiary", "Postgraduate"],
                            value="Secondary"),
                gr.Number(label="Monthly Marketing Spend (KES)", value=45000),
                gr.Number(label="Avg Daily Foot Traffic (leave 0 if not applicable)", value=0),
                gr.Number(label="Online Presence Score", value=78.0),
                gr.Number(label="Loan Amount (KES)", value=250000),
                gr.Number(label="Interest Rate (%)", value=14.5),
                gr.Number(label="Inflation Rate (%)", value=6.5),
                gr.Number(label="Competitor Density", value=5),
                gr.Number(label="Customer Satisfaction Score", value=4.2),
            ],
            outputs=gr.Textbox(label="Predicted Monthly Revenue"),
        )

    with gr.Tab("SME Credit Risk Classifier"):
        gr.Interface(
            fn=predict_credit_risk,
            inputs=[
                gr.Textbox(label="Sector", value="Retail"),
                gr.Textbox(label="County", value="Nairobi"),
                gr.Textbox(label="Quarter", value="Q2"),
                gr.Textbox(label="Loan Purpose", value="Working Capital"),
                gr.Dropdown(label="Owner Education Level",
                            choices=["Primary", "Secondary", "Tertiary", "Postgraduate"],
                            value="Secondary"),
                gr.Number(label="Business Age (years)", value=2.1),
                gr.Number(label="Number of Employees", value=4),
                gr.Number(label="Monthly Revenue (KES)", value=850000),
                gr.Number(label="Loan Amount (KES)", value=150000),
                gr.Number(label="Loan Term (months)", value=12),
                gr.Number(label="Interest Rate (%)", value=16.5),
                gr.Number(label="Collateral Value (leave 0 if unsecured)", value=0),
                gr.Number(label="Debt-to-Income Ratio", value=0.62),
                gr.Number(label="Credit Score", value=590),
                gr.Number(label="Number of Previous Loans", value=1),
                gr.Number(label="Missed Payments Count", value=3),
                gr.Number(label="Online Presence Score", value=40.0),
                gr.Number(label="Customer Satisfaction Score", value=3.4),
                gr.Number(label="Competitor Density", value=6),
            ],
            outputs=gr.Textbox(label="Predicted Credit Risk Category"),
        )

    with gr.Tab("Mall Customer Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[
                gr.Number(label="Age", value=27),
                gr.Number(label="Annual Income (k$)", value=45),
                gr.Number(label="Spending Score (1-100)", value=72),
            ],
            outputs=gr.Textbox(label="Prediction"),
        )

    with gr.Tab("E-commerce Recommender"):
        gr.Interface(
            fn=recommend_products,
            inputs=[
                gr.Textbox(label="Cart (comma-separated SKUs)", value="H06, H07"),
                gr.Number(label="Top N Recommendations", value=3),
            ],
            outputs=gr.Textbox(label="Recommendations"),
        )

if __name__ == "__main__":
    demo.launch()