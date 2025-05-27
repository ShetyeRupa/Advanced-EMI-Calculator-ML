# Import necessary libraries
import numpy as np  # For numerical operations (e.g., arrays)
import pandas as pd  # For handling data in tabular format
import seaborn as sns  # For data visualization (optional in your code but imported)
import matplotlib.pyplot as plt  # For plotting graphs (optional in your code but imported)
from sklearn.model_selection import train_test_split  # To split the dataset into training and test sets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Import models for classification and regression
from sklearn.metrics import accuracy_score, mean_absolute_error  # For performance evaluation of models
import warnings  # To suppress warnings in the output for cleaner presentation

# Suppressing warnings to make the output cleaner, especially when you're presenting
warnings.filterwarnings("ignore")  

# Function to calculate EMI (Equated Monthly Installment) based on the loan amount, interest rate, and tenure.
# EMI is the amount a borrower has to pay each month to repay the loan.
def calculate_emi(principal, rate, tenure):
    if rate == 0:  # If interest rate is 0%, simply divide the principal by tenure
        return round(principal / tenure, 2)  # Return the result rounded to 2 decimal places
    rate = rate / (12 * 100)  # Convert annual interest rate to monthly rate (divide by 12 months and 100 to get the percentage)
    emi = (principal * rate * (1 + rate) ** tenure) / ((1 + rate) ** tenure - 1)  # EMI formula
    return round(emi, 2)  # Return the calculated EMI rounded to 2 decimal places

# Loading a sample dataset that contains the details of past loan applications.
# The dataset has 'Credit_Score', 'Income', 'Loan_Amount', 'Interest_Rate', and the status of 'Loan_Approval'.
data = pd.DataFrame({
    'Credit_Score': [750, 650, 800, 550, 700, 600, 720, 580, 680, 740],  # Sample credit scores
    'Income': [50000, 40000, 80000, 25000, 55000, 30000, 60000, 27000, 52000, 73000],  # Monthly incomes
    'Loan_Amount': [200000, 150000, 300000, 100000, 180000, 120000, 220000, 90000, 175000, 250000],  # Requested loan amounts
    'Interest_Rate': [7.5, 9.2, 6.8, 12.5, 8.0, 11.0, 7.2, 13.0, 8.5, 6.9],  # Interest rates for loans
    'Loan_Approval': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]  # 1 = Approved, 0 = Rejected (Target variable)
})

# Splitting the dataset into features (X) and the target variable (y).
# The target variable (y) is 'Loan_Approval' which indicates if the loan was approved or rejected.
X = data[['Credit_Score', 'Income', 'Loan_Amount']]  # Features used for loan approval prediction (independent variables)
y = data['Loan_Approval']  # Target variable: 1 (Approved), 0 (Rejected)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and test sets

# Training a Random Forest Classifier model to predict loan approval.
loan_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Using 100 trees for the classifier
loan_model.fit(X_train, y_train)  # Fit the model to the training data

# Predicting loan approval status on the test set and checking the model's accuracy
y_pred = loan_model.predict(X_test)  # Predict loan approval (1 or 0) on test data
print(f'Loan Approval Prediction Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')  # Print accuracy of the model

# Now, let's predict the interest rates for loans using the same features.
# We will train a Random Forest Regressor for this task (since we are predicting a continuous value - interest rate).
X_reg = data[['Credit_Score', 'Income', 'Loan_Amount']]  # Features used to predict interest rate
y_reg = data['Interest_Rate']  # Target variable: Interest rate (continuous value)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)  # Split data

rate_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Create a Random Forest Regressor model
rate_model.fit(X_train_reg, y_train_reg)  # Fit the regressor to the training data

# Check the performance of the interest rate prediction model
y_pred_reg = rate_model.predict(X_test_reg)  # Predict interest rates on test data
print(f'Mean Absolute Error in Interest Rate Prediction: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}')  # Print mean absolute error

# Training another Random Forest Regressor to predict the maximum loan amount a person is eligible for.
X_loan = data[['Credit_Score', 'Income']]  # Features to predict the maximum loan amount (no loan amount here)
y_loan = data['Loan_Amount']  # Target variable: Loan amount (continuous value)
X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(X_loan, y_loan, test_size=0.2, random_state=42)  # Split data

loan_amount_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Create a Random Forest Regressor for loan amount
loan_amount_model.fit(X_train_loan, y_train_loan)  # Fit the model to the training data

# Now, we will take the user's input and apply the models for predictions.
try:
    # Taking input for the user's credit score, income, loan amount, and tenure.
    credit_score = int(input("Enter Credit Score: "))  # Input for credit score (integer)
    
    # If the credit score is below 600, the loan is automatically rejected.
    if credit_score < 600:  # Check if the credit score is below 600
        print(f'Loan Approval Status: Rejected')  # Print rejection message
        print("Reason: Credit score is below the minimum required threshold of 600.")  # Explanation of rejection reason
        exit()  # Exit the program as no further processing is needed

    income = int(input("Enter Monthly Income: "))  # Input for monthly income (integer)
    loan_amount = int(input("Enter Loan Amount: "))  # Input for loan amount (integer)
    tenure = int(input("Enter Tenure (in months): "))  # Input for loan tenure (integer)

    # We convert the user's input into a DataFrame for making predictions.
    input_data = pd.DataFrame([[credit_score, income, loan_amount]], columns=['Credit_Score', 'Income', 'Loan_Amount'])
    input_data_loan = pd.DataFrame([[credit_score, income]], columns=['Credit_Score', 'Income'])

    # Predicting the maximum loan amount the user is eligible for using the trained model.
    max_loan_amount = loan_amount_model.predict(input_data_loan)[0]  # Predict the maximum loan amount

    # Predicting loan approval status using the trained classifier model.
    predicted_approval = loan_model.predict(input_data)[0]  # Predict loan approval status (1: Approved, 0: Rejected)
    
    # Predicting the interest rate based on the user's input.
    predicted_rate = rate_model.predict(input_data)[0]  # Predict interest rate for the loan

    # Checking if the loan amount exceeds the maximum eligibility, and if so, reject the loan.
    if credit_score < 600:  # If the credit score is low, reject the loan
        print(f'Loan Approval Status: Rejected')  # Print rejection message
        print("Reason: Credit score is below the minimum required threshold of 600.")  # Explanation
    elif loan_amount > max_loan_amount:  # If the loan amount exceeds the eligibility, reject the loan
        print(f'Loan Approval Status: Rejected')  # Print rejection message
        print(f"Reason: Requested loan amount exceeds the estimated maximum eligibility of ${max_loan_amount:.2f}.")  # Explanation
    else:
        # If loan is approved, calculate and display the EMI based on the predicted interest rate.
        emi = calculate_emi(loan_amount, predicted_rate, tenure)  # Calculate EMI based on the formula
        print("\n--- Loan Prediction Results ---")  # Print results header
        print(f'Loan Approval Status: Approved')  # Print approval message
        print(f'Predicted Interest Rate: {predicted_rate:.2f}%')  # Display the predicted interest rate
        print(f'Calculated EMI: ${emi:.2f}')  # Display the calculated EMI
    
    # If the loan is rejected, display the maximum loan eligibility
    if loan_amount > max_loan_amount or credit_score < 600:  # If loan is rejected, display max eligibility
        print(f"Estimated Maximum Loan Amount You Can Take: ${max_loan_amount:.2f}")  # Display eligibility
    
except ValueError:  # Handle invalid input (non-numeric input)
    print("Invalid input! Please enter numeric values only.")  # Print error message for invalid input
