"""
Advanced Loan EMI Calculator with Machine Learning
Author: Rupali Ravindra Shetye

This program combines financial calculations with ML models to:
1. Predict loan approval using Random Forest Classifier
2. Predict interest rates using Random Forest Regressor
3. Calculate EMI and generate amortization schedules
4. Save all results with visualizations
"""

# =============================================
# 1. IMPORT LIBRARIES
# =============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, 
    confusion_matrix, classification_report, 
    roc_curve, auc, mean_squared_error, r2_score
)
import warnings
from faker import Faker  # For realistic data generation
import os
import datetime
from sklearn.inspection import permutation_importance

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set matplotlib to interactive mode
plt.ion()

# =============================================
# 2. DATA GENERATION AND PREPARATION
# =============================================
def generate_realistic_loan_data(num_records=1000):
    """
    Generate realistic loan application dataset with:
    - Credit scores (300-850)
    - Incomes (log-normal distribution)
    - Existing debts
    - Calculated loan amounts
    - Interest rates based on creditworthiness
    - Approval decisions with realistic rules
    
    Args:
        num_records: Number of records to generate
        
    Returns:
        DataFrame with generated loan data
    """
    data = {
        'Credit_Score': np.random.normal(650, 100, num_records).astype(int),
        'Income': np.random.lognormal(10.5, 0.4, num_records).astype(int),
        'Existing_Debt': np.random.lognormal(8, 0.3, num_records).astype(int),
        'Loan_Amount': [],
        'Interest_Rate': [],
        'Loan_Approval': []
    }
    
    # Clip credit scores to realistic range (300-850)
    data['Credit_Score'] = np.clip(data['Credit_Score'], 300, 850)
    
    # Generate loan data for each record
    for i in range(num_records):
        # Calculate loan amount based on income and credit score
        base_amount = data['Income'][i] * np.random.uniform(2, 5)
        credit_modifier = data['Credit_Score'][i] / 700  # 700 as baseline
        loan_amount = int(base_amount * credit_modifier)
        data['Loan_Amount'].append(loan_amount)
        
        # Generate interest rates with penalty for lower credit scores
        base_rate = 5.0  # Best available rate
        rate_penalty = (850 - data['Credit_Score'][i]) / 100
        interest_rate = round(base_rate + rate_penalty + np.random.uniform(0, 3), 1)
        data['Interest_Rate'].append(interest_rate)
        
        # Determine approval with realistic business rules
        debt_to_income = data['Existing_Debt'][i] / data['Income'][i]
        approval = 1 if (
            data['Credit_Score'][i] > 620 and 
            debt_to_income < 0.5 and 
            np.random.random() > 0.2  # 20% rejection chance even if qualified
        ) else 0
        data['Loan_Approval'].append(approval)
    
    return pd.DataFrame(data)

# Generate and store the dataset
data = generate_realistic_loan_data(1000)

# =============================================
# 3. FINANCIAL CALCULATIONS
# =============================================
def calculate_emi(principal, rate, tenure):
    """
    Calculate Equated Monthly Installment (EMI)
    
    Args:
        principal: Loan amount
        rate: Annual interest rate
        tenure: Loan term in months
        
    Returns:
        EMI amount rounded to 2 decimal places
    """
    if rate == 0:  # Handle zero-interest case
        return round(principal / tenure, 2)
    
    monthly_rate = rate / (12 * 100)  # Convert annual rate to monthly decimal
    emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure) / ((1 + monthly_rate) ** tenure - 1)
    return round(emi, 2)

def generate_amortization_schedule(principal, rate, tenure):
    """
    Generate full amortization schedule with validation
    
    Args:
        principal: Loan amount
        rate: Annual interest rate
        tenure: Loan term in months
        
    Returns:
        DataFrame with monthly payment breakdown or empty DataFrame if invalid
    """
    # Validate inputs
    if principal <= 0 or tenure <= 0:
        return pd.DataFrame(columns=['Month', 'EMI', 'Principal', 'Interest', 'Balance'])
    
    try:
        # Calculate EMI and initialize variables
        monthly_rate = rate / (12 * 100)
        emi = calculate_emi(principal, rate, tenure)
        balance = principal
        schedule = []
        
        # Calculate payment breakdown for each month
        for month in range(1, tenure + 1):
            interest = balance * monthly_rate
            principal_paid = emi - interest
            balance -= principal_paid
            
            schedule.append({
                'Month': month,
                'EMI': emi,
                'Principal': principal_paid,
                'Interest': interest,
                'Balance': max(balance, 0)  # Prevent negative balance
            })
        
        return pd.DataFrame(schedule)
    
    except Exception as e:
        print(f"Amortization error: {str(e)}")
        return pd.DataFrame(columns=['Month', 'EMI', 'Principal', 'Interest', 'Balance'])

# =============================================
# 4. MACHINE LEARNING MODEL SETUP
# =============================================
# Prepare features and target variables
X = data[['Credit_Score', 'Income', 'Existing_Debt', 'Loan_Amount']]
y = data['Loan_Approval']
X_reg = data[['Credit_Score', 'Income', 'Existing_Debt', 'Loan_Amount']]
y_reg = data['Interest_Rate']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Initialize models
loan_model = RandomForestClassifier(n_estimators=200, random_state=42)
rate_model = RandomForestRegressor(n_estimators=200, random_state=42)

# Train models
loan_model.fit(X_train, y_train)
rate_model.fit(X_train_reg, y_train_reg)

# =============================================
# 5. MODEL EVALUATION AND METRICS
# =============================================
def evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name=""):
    """
    Evaluate classification model performance with comprehensive metrics
    
    Args:
        model: Trained classifier
        X_train, X_test: Features
        y_train, y_test: Targets
        model_name: Identifier for the model
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    return {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_report': classification_report(y_train, y_train_pred, output_dict=True),
        'test_report': classification_report(y_test, y_test_pred, output_dict=True),
        'train_cm': confusion_matrix(y_train, y_train_pred),
        'test_cm': confusion_matrix(y_test, y_test_pred),
        'fpr_train': roc_curve(y_train, y_train_prob)[0],
        'tpr_train': roc_curve(y_train, y_train_prob)[1],
        'fpr_test': roc_curve(y_test, y_test_prob)[0],
        'tpr_test': roc_curve(y_test, y_test_prob)[1],
        'roc_auc_train': auc(roc_curve(y_train, y_train_prob)[0], roc_curve(y_train, y_train_prob)[1]),
        'roc_auc_test': auc(roc_curve(y_test, y_test_prob)[0], roc_curve(y_test, y_test_prob)[1]),
        'feature_importance': model.feature_importances_
    }

def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name=""):
    """
    Evaluate regression model performance with comprehensive metrics
    
    Args:
        model: Trained regressor
        X_train, X_test: Features
        y_train, y_test: Targets
        model_name: Identifier for the model
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    return {
        'model_name': model_name,
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'feature_importance': model.feature_importances_
    }

def save_model_metrics(results, model_type="classification"):
    """
    Save model evaluation metrics and visualizations to files
    
    Args:
        results: Dictionary of evaluation metrics
        model_type: Either "classification" or "regression"
    """
    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = f"results/model_eval_{timestamp}"
    os.makedirs(base_path, exist_ok=True)
    
    if model_type == "classification":
        # Confusion Matrix plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(results['train_cm'], annot=True, fmt='d', cmap='Blues')
        plt.title('Train Confusion Matrix')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(results['test_cm'], annot=True, fmt='d', cmap='Blues')
        plt.title('Test Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{base_path}/confusion_matrix.png")
        plt.close()
        
        # ROC Curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(results['fpr_train'], results['tpr_train'], 
                label=f'Train ROC (AUC = {results["roc_auc_train"]:.2f})')
        plt.plot(results['fpr_test'], results['tpr_test'], 
                label=f'Test ROC (AUC = {results["roc_auc_test"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f"{base_path}/roc_curve.png")
        plt.close()
        
        # Feature Importance plot
        plt.figure(figsize=(10, 6))
        features = X_train.columns
        indices = np.argsort(results['feature_importance'])[::-1]
        plt.bar(range(len(features)), results['feature_importance'][indices])
        plt.xticks(range(len(features)), features[indices], rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{base_path}/feature_importance.png")
        plt.close()
        
        # Save metrics to text file
        with open(f"{base_path}/classification_metrics.txt", 'w') as f:
            f.write(f"Model: {results['model_name']}\n")
            f.write(f"\nTrain Accuracy: {results['train_accuracy']:.4f}")
            f.write(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
            f.write("\n\nTrain Classification Report:\n")
            f.write(classification_report(y_train, loan_model.predict(X_train)))
            f.write("\nTest Classification Report:\n")
            f.write(classification_report(y_test, loan_model.predict(X_test)))
    
    elif model_type == "regression":
        # Feature Importance plot
        plt.figure(figsize=(10, 6))
        features = X_train_reg.columns
        indices = np.argsort(results['feature_importance'])[::-1]
        plt.bar(range(len(features)), results['feature_importance'][indices])
        plt.xticks(range(len(features)), features[indices], rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{base_path}/feature_importance.png")
        plt.close()
        
        # Save metrics to text file
        with open(f"{base_path}/regression_metrics.txt", 'w') as f:
            f.write(f"Model: {results['model_name']}\n")
            f.write(f"\nTrain MAE: {results['train_mae']:.4f}")
            f.write(f"\nTest MAE: {results['test_mae']:.4f}")
            f.write(f"\nTrain MSE: {results['train_mse']:.4f}")
            f.write(f"\nTest MSE: {results['test_mse']:.4f}")
            f.write(f"\nTrain R2: {results['train_r2']:.4f}")
            f.write(f"\nTest R2: {results['test_r2']:.4f}")

# Evaluate and save model metrics
loan_metrics = evaluate_classification_model(loan_model, X_train, X_test, y_train, y_test, "Loan Approval Classifier")
rate_metrics = evaluate_regression_model(rate_model, X_train_reg, X_test_reg, y_train_reg, y_test_reg, "Interest Rate Regressor")

save_model_metrics(loan_metrics, "classification")
save_model_metrics(rate_metrics, "regression")

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
loan_cv_scores = cross_val_score(loan_model, X, y, cv=cv, scoring='accuracy')
rate_cv_scores = cross_val_score(rate_model, X_reg, y_reg, cv=cv, scoring='neg_mean_absolute_error')

# =============================================
# 6. USER INTERFACE AND LOAN CALCULATION
# =============================================
def get_user_input():
    """
    Get and validate user input for loan calculation
    
    Returns:
        Tuple of (credit_score, income, existing_debt, loan_amount, tenure)
    """
    while True:
        try:
            print("\n" + "="*50)
            print("LOAN ELIGIBILITY CALCULATOR".center(50))
            print("="*50)
            
            credit_score = int(input("\nEnter Credit Score (300-850): "))
            if not 300 <= credit_score <= 850:
                print("Error: Credit score must be between 300 and 850")
                continue
                
            income = int(input("Enter Monthly Income ($): "))
            if income <= 0:
                print("Error: Income must be greater than 0")
                continue
                
            existing_debt = int(input("Enter Existing Monthly Debt Payments ($): "))
            if existing_debt < 0:
                print("Error: Debt cannot be negative")
                continue
                
            loan_amount = int(input("Enter Desired Loan Amount ($): "))
            if loan_amount <= 0:
                print("Error: Loan amount must be greater than 0")
                continue
                
            tenure = int(input("Enter Loan Tenure (months, 6-60): "))
            if tenure < 6 or tenure > 60:
                print("Error: Tenure must be between 6 and 60 months")
                continue
                
            return credit_score, income, existing_debt, loan_amount, tenure
            
        except ValueError:
            print("Invalid input! Please enter numbers only.")

def save_loan_results(approval, predicted_rate, emi, amortization_df, max_loan=None):
    """
    Save loan calculation results and visualizations
    
    Args:
        approval: Loan approval status (1 or 0)
        predicted_rate: Predicted interest rate
        emi: Calculated EMI amount
        amortization_df: Amortization schedule DataFrame
        max_loan: Maximum eligible loan amount
    """
    # Create timestamped directory for this loan calculation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results/loan_result_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    try:
        # Save results to text file
        with open(f"{result_dir}/loan_results.txt", 'w', encoding='utf-8') as f:
            f.write("LOAN DECISION RESULTS\n")
            f.write("="*50 + "\n")
            status = "APPROVED" if approval else "REJECTED"
            f.write(f"\n{status}\n")
            f.write(f"• Predicted Interest Rate: {predicted_rate:.2f}%\n")
            
            if approval and not amortization_df.empty:
                f.write(f"• Monthly EMI: ${emi:,.2f}\n")
                amortization_df.to_csv(f"{result_dir}/amortization_schedule.csv", index=False)
                
        # Save plot if approved
        if approval and not amortization_df.empty:
            plt.figure(figsize=(12, 6))
            plt.stackplot(amortization_df['Month'], 
                         amortization_df['Principal'], 
                         amortization_df['Interest'],
                         labels=['Principal', 'Interest'])
            plt.title('EMI Composition Over Time')
            plt.xlabel('Month')
            plt.ylabel('Amount ($)')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.savefig(f"{result_dir}/emi_composition.png")
            plt.close()
            
    except Exception as e:
        print(f"\n⚠️ Warning: Could not save results - {str(e)}")

def display_loan_results(approval, predicted_rate, emi, amortization_df, max_loan=None):
    """
    Display loan calculation results to user
    
    Args:
        approval: Loan approval status (1 or 0)
        predicted_rate: Predicted interest rate
        emi: Calculated EMI amount
        amortization_df: Amortization schedule DataFrame
        max_loan: Maximum eligible loan amount
    """
    print("\n" + "="*50)
    print("LOAN DECISION RESULTS".center(50))
    print("="*50)
    
    if approval == 1:
        print(f"\n✅ LOAN APPROVED")
        print(f"• Predicted Interest Rate: {predicted_rate:.2f}%")
        
        if not amortization_df.empty:
            print(f"• Monthly EMI: ${emi:,.2f} for {len(amortization_df)} months")
            
            # Plot amortization schedule
            plt.figure(figsize=(12, 6))
            plt.stackplot(amortization_df['Month'], 
                         amortization_df['Principal'], 
                         amortization_df['Interest'],
                         labels=['Principal', 'Interest'])
            plt.title('EMI Composition Over Time')
            plt.xlabel('Month')
            plt.ylabel('Amount ($)')
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.show()
            
            # Print first 12 months amortization
            print("\nFirst 12 Months Amortization Schedule:")
            formatted_df = amortization_df.head(12).copy()
            formatted_df['Balance'] = formatted_df['Balance'].apply(lambda x: f"{x:,.2f}")
            print(formatted_df.to_string(index=False, formatters={
                'Month': '{:,.0f}'.format,
                'EMI': '{:,.2f}'.format,
                'Principal': '{:,.2f}'.format,
                'Interest': '{:,.2f}'.format
            }))
    else:
        print(f"\n❌ LOAN REJECTED")
        if max_loan:
            print(f"• Maximum Eligible Loan Amount: ${max_loan:,.2f}")

    # Add clear separation before continuation prompt
    print("\n" + "="*50)

def validate_loan_parameters(credit_score, income, existing_debt, loan_amount, tenure):
    """
    Validate loan parameters against business rules
    
    Args:
        credit_score: Applicant's credit score
        income: Monthly income
        existing_debt: Monthly debt payments
        loan_amount: Requested loan amount
        tenure: Loan term in months
        
    Returns:
        tuple: (is_valid, rejection_reason)
    """
    # Check debt-to-income ratio
    debt_to_income = existing_debt / income
    if debt_to_income >= 0.5:
        return False, f"Debt-to-income ratio too high ({debt_to_income:.2f})"
    
    # Check maximum eligible loan amount (10x annual income)
    max_eligible = income * 10 * 12
    if loan_amount > max_eligible:
        return False, f"Loan amount exceeds maximum eligibility (${max_eligible:,.2f})"
    
    # Check credit score minimum
    if credit_score < 620:
        return False, "Credit score below minimum (620)"
    
    return True, ""

def prepare_input_data(credit_score, income, existing_debt, loan_amount):
    """
    Prepare input data for predictions
    
    Args:
        credit_score: Applicant's credit score
        income: Monthly income
        existing_debt: Monthly debt payments
        loan_amount: Requested loan amount
        
    Returns:
        DataFrame with input features
    """
    return pd.DataFrame([[credit_score, income, existing_debt, loan_amount]],
                      columns=['Credit_Score', 'Income', 'Existing_Debt', 'Loan_Amount'])

def make_predictions(input_data):
    """
    Make loan approval and interest rate predictions
    
    Args:
        input_data: DataFrame with input features
        
    Returns:
        tuple: (approval_status, predicted_rate)
    """
    approval_prob = loan_model.predict_proba(input_data)[0][1]
    approval = 1 if approval_prob > 0.5 else 0
    predicted_rate = rate_model.predict(input_data)[0]
    return approval, predicted_rate

def calculate_loan_details(loan_amount, rate, tenure):
    """
    Calculate EMI and generate amortization schedule
    
    Args:
        loan_amount: Loan amount
        rate: Annual interest rate
        tenure: Loan term in months
        
    Returns:
        tuple: (emi, amortization_df)
    """
    try:
        if loan_amount <= 0 or rate < 0 or tenure <= 0:
            raise ValueError("Invalid loan parameters")
            
        emi = calculate_emi(loan_amount, rate, tenure)
        amortization_df = generate_amortization_schedule(loan_amount, rate, tenure)
        return emi, amortization_df
    except Exception as e:
        print(f"\n⚠️ EMI Calculation Error: {str(e)}")
        return None, pd.DataFrame()

def prompt_to_continue():
    """Prompt user to continue or exit"""
    while True:
        try:
            choice = input("\nCalculate another loan? (y/n): ").strip().lower()
            if choice == 'n':
                print("\nThank you for using the Loan Eligibility Calculator!")
                return False
            if choice == 'y':
                return True
            print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return False
        except:
            print("Invalid input! Please try again")

def print_model_performance():
    """Print model metrics at startup"""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS".center(50))
    print("="*50)
    print(f"\nLoan Approval Model - Cross-Validated Accuracy: {loan_cv_scores.mean():.2f} (±{loan_cv_scores.std():.2f})")
    print(f"Interest Rate Model - Cross-Validated MAE: {-rate_cv_scores.mean():.2f}")
    print("="*50)

# =============================================
# 7. MAIN APPLICATION FLOW
# =============================================
def main():
    """Main application workflow"""
    # Initial setup
    if not os.path.exists('results'):
        os.makedirs('results')
    
    print_model_performance()
    
    # Main application loop
    while True:
        try:
            # Get user input
            credit_score, income, existing_debt, loan_amount, tenure = get_user_input()
            
            # Validate against business rules
            is_valid, rejection_reason = validate_loan_parameters(
                credit_score, income, existing_debt, loan_amount, tenure
            )
            
            if not is_valid:
                print(f"\n❌ LOAN REJECTED: {rejection_reason}")
                max_loan = income * 10 * 12
                print(f"• Maximum Eligible Loan Amount: ${max_loan:,.2f}")
                print("\n" + "="*50)
                if not prompt_to_continue():
                    break
                continue
            
            # Prepare input data and make predictions
            input_data = prepare_input_data(credit_score, income, existing_debt, loan_amount)
            approval, predicted_rate = make_predictions(input_data)
            
            # Calculate EMI and generate schedule
            emi, amortization_df = calculate_loan_details(loan_amount, predicted_rate, tenure)
            max_loan = income * 10 * 12
            
            # Display and save results
            display_loan_results(approval, predicted_rate, emi, amortization_df, max_loan)
            save_loan_results(approval, predicted_rate, emi, amortization_df, max_loan)
            
            if not prompt_to_continue():
                break
                
        except ValueError as ve:
            print(f"\nError: {str(ve)} - Please enter valid numbers only")
        
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Exiting...")
            break
       
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
            print("Please try again or contact support")

# =============================================
# 8. APPLICATION ENTRY POINT
# =============================================
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Run main application
    main()
    
    # Keep window open after completion
    input("\nPress Enter to exit...")