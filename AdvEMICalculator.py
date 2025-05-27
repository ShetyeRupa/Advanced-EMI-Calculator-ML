"""
Advanced Loan EMI Calculator with Machine Learning (Production-Ready)
Author: Rupali Ravindra Shetye
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
import logging
from typing import Tuple, Dict, Optional
from functools import lru_cache
from dataclasses import dataclass
import yaml
from cryptography.fernet import Fernet
import os
import datetime
from pathlib import Path
from tenacity import retry, stop_after_attempt
import argparse
from tabulate import tabulate
from prometheus_client import Counter, start_http_server
import joblib

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('loan_calculator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize metrics
APPROVAL_COUNTER = Counter('loan_approvals', 'Total loan approvals/rejections', ['status'])

# =============================================
# 2. CONFIGURATION MANAGEMENT
# =============================================
@dataclass
class AppConfig:
    MIN_CREDIT_SCORE: int = 620
    MAX_DEBT_TO_INCOME: float = 0.5
    LOAN_MULTIPLIER: int = 10
    MODEL_PATHS: Dict[str, str] = None
    ENCRYPTION_KEY: str = None

def load_config(config_path: str = 'config.yaml') -> AppConfig:
    """Load configuration from YAML file"""
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            return AppConfig(**config_data)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults")
        return AppConfig()

config = load_config()

# =============================================
# 3. DATA ENCRYPTION
# =============================================
class DataEncryptor:
    def __init__(self, key: str = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()

encryptor = DataEncryptor(config.ENCRYPTION_KEY)

# =============================================
# 4. DATA GENERATION AND PREPARATION
# =============================================
def generate_realistic_loan_data(num_records: int = 1000) -> pd.DataFrame:
    """Generate realistic loan application dataset"""
    data = {
        'Credit_Score': np.clip(np.random.normal(650, 100, num_records).astype(int), 300, 850),
        'Income': np.random.lognormal(10.5, 0.4, num_records).astype(int),
        'Existing_Debt': np.random.lognormal(8, 0.3, num_records).astype(int),
        'Loan_Amount': [],
        'Interest_Rate': [],
        'Loan_Approval': []
    }
    
    for i in range(num_records):
        base_amount = data['Income'][i] * np.random.uniform(2, 5)
        credit_modifier = data['Credit_Score'][i] / 700
        loan_amount = int(base_amount * credit_modifier)
        data['Loan_Amount'].append(loan_amount)
        
        base_rate = 5.0
        rate_penalty = (850 - data['Credit_Score'][i]) / 100
        interest_rate = round(base_rate + rate_penalty + np.random.uniform(0, 3), 1)
        data['Interest_Rate'].append(interest_rate)
        
        debt_to_income = data['Existing_Debt'][i] / data['Income'][i]
        approval = 1 if (
            data['Credit_Score'][i] > config.MIN_CREDIT_SCORE and
            debt_to_income < config.MAX_DEBT_TO_INCOME and
            np.random.random() > 0.2
        ) else 0
        data['Loan_Approval'].append(approval)
    
    return pd.DataFrame(data)

# =============================================
# 5. FINANCIAL CALCULATIONS (WITH CACHING)
# =============================================
@lru_cache(maxsize=1000)
def calculate_emi(principal: float, rate: float, tenure: int) -> float:
    """Calculate EMI with caching for repeated inputs"""
    try:
        if rate == 0:
            return round(principal / tenure, 2)
        
        monthly_rate = rate / (12 * 100)
        emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure) / ((1 + monthly_rate) ** tenure - 1)
        return round(emi, 2)
    except Exception as e:
        logger.error(f"EMI calculation failed: {e}")
        raise ValueError("Invalid EMI calculation parameters")

def generate_amortization_schedule(principal: float, rate: float, tenure: int) -> pd.DataFrame:
    """Generate amortization schedule with validation"""
    if principal <= 0 or tenure <= 0:
        logger.warning(f"Invalid amortization params: principal={principal}, tenure={tenure}")
        return pd.DataFrame(columns=['Month', 'EMI', 'Principal', 'Interest', 'Balance'])
    
    try:
        monthly_rate = rate / (12 * 100)
        emi = calculate_emi(principal, rate, tenure)
        balance = principal
        schedule = []
        
        for month in range(1, tenure + 1):
            interest = balance * monthly_rate
            principal_paid = emi - interest
            balance -= principal_paid
            
            schedule.append({
                'Month': month,
                'EMI': emi,
                'Principal': principal_paid,
                'Interest': interest,
                'Balance': max(balance, 0)
            })
        
        return pd.DataFrame(schedule)
    except Exception as e:
        logger.error(f"Amortization failed: {e}")
        return pd.DataFrame()

# =============================================
# 6. MACHINE LEARNING MODELS
# =============================================
class LoanModel:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.regressor = RandomForestRegressor(n_estimators=200, random_state=42)
        self._load_or_train_models()

    def _load_or_train_models(self):
        """Load saved models or train new ones"""
        try:
            if config.MODEL_PATHS:
                self.classifier = joblib.load(config.MODEL_PATHS.get('classifier'))
                self.regressor = joblib.load(config.MODEL_PATHS.get('regressor'))
                logger.info("Loaded pre-trained models")
            else:
                self._train_models()
        except Exception as e:
            logger.warning(f"Model loading failed: {e}. Training new models")
            self._train_models()

    def _train_models(self):
        """Train and save models"""
        data = generate_realistic_loan_data(1000)
        
        X = data[['Credit_Score', 'Income', 'Existing_Debt', 'Loan_Amount']]
        y = data['Loan_Approval']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.classifier.fit(X_train, y_train)
        self.regressor.fit(X_train, data.loc[X_train.index, 'Interest_Rate'])
        
        Path("models").mkdir(parents=True, exist_ok=True)

        joblib.dump(self.classifier, config.MODEL_PATHS['classifier'])
        joblib.dump(self.regressor, config.MODEL_PATHS['regressor'])
        logger.info(f" Models saved to {config.MODEL_PATHS['classifier']} and {config.MODEL_PATHS['regressor']}")

        if config.MODEL_PATHS:
            joblib.dump(self.classifier, config.MODEL_PATHS['classifier'])
            joblib.dump(self.regressor, config.MODEL_PATHS['regressor'])

    @retry(stop=stop_after_attempt(3))
    def predict(self, input_data: pd.DataFrame) -> Tuple[int, float]:
        """Make predictions with retry logic"""
        try:
            approval_prob = self.classifier.predict_proba(input_data)[0][1]
            approval = 1 if approval_prob > 0.5 else 0
            rate = self.regressor.predict(input_data)[0]
            return approval, rate
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError("Model prediction error")

# Initialize models
loan_model = LoanModel()

# =============================================
# 7. USER INTERFACE & VALIDATION
# =============================================
def validate_input(credit_score: int, income: float, existing_debt: float, 
                  loan_amount: float, tenure: int) -> Tuple[bool, str]:
    """Validate user input with business rules"""
    if not (300 <= credit_score <= 850):
        return False, "Credit score must be between 300-850"
    if income <= 0:
        return False, "Income must be positive"
    if existing_debt < 0:
        return False, "Debt cannot be negative"
    if loan_amount <= 0:
        return False, "Loan amount must be positive"
    if not (6 <= tenure <= 60):
        return False, "Tenure must be 6-60 months"
    
    debt_to_income = existing_debt / income
    if debt_to_income >= config.MAX_DEBT_TO_INCOME:
        return False, f"Debt-to-income ratio exceeds {config.MAX_DEBT_TO_INCOME}"
    
    max_loan = income * config.LOAN_MULTIPLIER * 12
    if loan_amount > max_loan:
        return False, f"Loan amount exceeds maximum eligible (${max_loan:,.2f})"
    
    if credit_score < config.MIN_CREDIT_SCORE:
        return False, f"Credit score below minimum ({config.MIN_CREDIT_SCORE})"
    
    return True, ""

def get_user_input() -> Tuple[int, float, float, float, int]:
    """Get and validate user input"""
    parser = argparse.ArgumentParser(description="Loan Eligibility Calculator")
    parser.add_argument("--credit-score", type=int, required=True, help="Credit score (300-850)")
    parser.add_argument("--income", type=float, required=True, help="Monthly income ($)")
    parser.add_argument("--existing-debt", type=float, required=True, help="Existing monthly debt ($)")
    parser.add_argument("--loan-amount", type=float, required=True, help="Desired loan amount ($)")
    parser.add_argument("--tenure", type=int, required=True, help="Loan term in months (6-60)")
    
    args = parser.parse_args()
    
    is_valid, message = validate_input(
        args.credit_score, args.income,
        args.existing_debt, args.loan_amount,
        args.tenure
    )
    
    if not is_valid:
        logger.error(f"Invalid input: {message}")
        raise ValueError(message)
    
    return (args.credit_score, args.income, args.existing_debt, 
            args.loan_amount, args.tenure)

# =============================================
# 8. RESULT PROCESSING
# =============================================
def save_results(approval: int, rate: float, emi: float, 
                amortization_df: pd.DataFrame) -> None:
    """Save results with encryption and logging"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(f"results/loan_result_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save encrypted sensitive data
        with open(result_dir / "secure_data.txt", "w") as f:
            f.write(encryptor.encrypt(f"Rate: {rate}, EMI: {emi}"))
        
        # Save amortization schedule
        if not amortization_df.empty:
            amortization_df.to_csv(result_dir / "amortization.csv", index=False)
            
            # Plot EMI composition
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
            plt.savefig(result_dir / "emi_composition.png")
            plt.close()
            
        logger.info(f"Results saved to {result_dir}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def display_results(approval: int, rate: float, emi: float, 
                   amortization_df: pd.DataFrame) -> None:
    """Display results with rich formatting"""
    status = "✅ APPROVED" if approval else "❌ REJECTED"
    print(f"\nLoan Status: {status}")
    print(f"Predicted Interest Rate: {rate:.2f}%")
    
    if approval and not amortization_df.empty:
        print(f"Monthly EMI: ${emi:,.2f}")
        print("\nAmortization Schedule (First 12 Months):")
        print(tabulate(amortization_df.head(12), headers='keys', tablefmt='grid'))

# =============================================
# 9. MAIN APPLICATION FLOW
# =============================================
def main():
    """Production-ready main application flow"""
    try:
        # Start metrics server
        start_http_server(8000)
        
        # Get and validate input
        credit_score, income, existing_debt, loan_amount, tenure = get_user_input()
        
        # Prepare input data
        input_data = pd.DataFrame([[credit_score, income, existing_debt, loan_amount]],
                                columns=['Credit_Score', 'Income', 'Existing_Debt', 'Loan_Amount'])
        
        # Make predictions
        approval, rate = loan_model.predict(input_data)
        
        # Track metrics
        APPROVAL_COUNTER.labels(status='approved' if approval else 'rejected').inc()
        
        # Calculate loan details
        emi = calculate_emi(loan_amount, rate, tenure)
        amortization_df = generate_amortization_schedule(loan_amount, rate, tenure)
        
        # Output results
        display_results(approval, rate, emi, amortization_df)
        save_results(approval, rate, emi, amortization_df)
        
    except ValueError as ve:
        logger.error(f"Input validation error: {ve}")
        print(f"Error: {ve}")
    except Exception as e:
        logger.critical(f"Application error: {e}")
        print("An unexpected error occurred. Please check logs.")

if __name__ == "__main__":
    main()