import streamlit as st
from AdvEMICalculator import (
    calculate_emi, generate_amortization_schedule,
    validate_input, loan_model
)
import pandas as pd

st.title("📊 Advanced EMI Calculator with ML")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
debt = st.number_input("Existing Debt ($)", min_value=0.0, value=300.0)
loan = st.number_input("Desired Loan Amount ($)", min_value=1000.0, value=100000.0)
tenure = st.slider("Tenure (months)", min_value=6, max_value=60, value=36)

if st.button("Calculate Loan Approval & EMI"):
    valid, msg = validate_input(credit_score, income, debt, loan, tenure)
    if not valid:
        st.error(f"Input error: {msg}")
    else:
        df = pd.DataFrame([[credit_score, income, debt, loan]], columns=['Credit_Score', 'Income', 'Existing_Debt', 'Loan_Amount'])
        approval, rate = loan_model.predict(df)
        emi = calculate_emi(loan, rate, tenure)
        schedule = generate_amortization_schedule(loan, rate, tenure)

        st.success("✅ Loan Approved!" if approval else "❌ Loan Rejected")
        st.metric("Predicted Interest Rate (%)", f"{rate:.2f}")
        st.metric("Monthly EMI ($)", f"{emi:,.2f}")

        if approval and not schedule.empty:
            st.subheader("📄 Amortization Schedule (First 12 Months)")
            st.dataframe(schedule.head(12))

# Run with:
# streamlit run emi_app.py
