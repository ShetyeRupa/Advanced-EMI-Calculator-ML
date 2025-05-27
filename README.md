
# 💼 Advanced EMI Calculator with Machine Learning (Production-Ready)

### Author: Rupali Ravindra Shetye  

---

## 📌 Overview

This project presents a production-grade **Advanced EMI (Equated Monthly Installment) Calculator** integrated with **Machine Learning-based Loan Approval Prediction**. It not only performs precise EMI computations but also predicts loan approval status and interest rate dynamically based on user financial data.

The system is engineered with a focus on:
- Modularity and code readability
- Robust validation and logging
- Secure encrypted output handling
- Real-time explainability and deployment readiness

---

## 🚀 Features

| Module                     | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| 🔐 **Configurable YAML**    | Centralized hyperparameter and path management                             |
| 🧠 **ML Model Integration** | Random Forest Classifier & Regressor for approval and interest prediction  |
| 📊 **Financial Calculation**| Accurate EMI & amortization schedule generation                            |
| 🛡️ **Data Security**        | Encryption of sensitive results using Fernet symmetric key cryptography    |
| 🔁 **Retry Logic**          | `tenacity`-powered retry for fault-tolerant predictions                    |
| 📈 **Monitoring Metrics**   | Prometheus counters for loan approval tracking                             |
| 🧪 **Testing Suite**        | Pytest-compatible test cases for validation and EMI logic                  |
| 🖥️ **CLI Interface**        | Command-line based simulation using argparse                               |
| 📂 **Result Persistence**   | Amortization CSV and graphical plot exports with secure metadata           |

---

## 🧩 Technologies Used

- **Python 3.10+**
- `scikit-learn`, `joblib`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- `cryptography`, `tenacity`, `yaml`, `argparse`, `prometheus_client`
- `pytest` (for unit testing)
- `streamlit` (optional GUI)

---

## ⚙️ Configuration (`config.yaml`)

```yaml
MIN_CREDIT_SCORE: 620
MAX_DEBT_TO_INCOME: 0.5
LOAN_MULTIPLIER: 10

MODEL_PATHS:
  classifier: models/loan_classifier.pkl
  regressor: models/loan_regressor.pkl

ENCRYPTION_KEY: "your_fernet_key_here"
````

To generate your Fernet key:

```python
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
```

---

## 📦 Directory Structure

```
AdvEMICalculatorWithML/
├── AdvEMICalculator.py
├── config.yaml
├── emi_app.py (optional Streamlit GUI)
├── models/
│   ├── loan_classifier.pkl
│   └── loan_regressor.pkl
├── results/
│   └── loan_result_<timestamp>/
├── test_emi_calculator.py
├── README.md
└── requirements.txt
```

---

## 🧪 How to Run

### 📌 CLI Execution

```bash
python AdvEMICalculator.py \
  --credit-score 750 \
  --income 6000 \
  --existing-debt 800 \
  --loan-amount 100000 \
  --tenure 36
```

### 📈 Streamlit GUI (Optional)

```bash
streamlit run emi_app.py
```

---

## ✅ Sample Output

```text
Loan Status: ✅ APPROVED
Predicted Interest Rate: 7.99%
Monthly EMI: $3,133.08

Amortization Schedule (First 12 Months):
+----+---------+---------+-------------+------------+-----------+
|    |   Month |     EMI |   Principal |   Interest |   Balance |
+====+=========+=========+=============+============+===========+
|  0 |       1 | 3133.08 |     2467.41 |    665.667 |   97532.6 |
+----+---------+---------+-------------+------------+-----------+
|  1 |       2 | 3133.08 |     2483.84 |    649.242 |   95048.7 |
+----+---------+---------+-------------+------------+-----------+
|  2 |       3 | 3133.08 |     2500.37 |    632.708 |   92548.4 |
+----+---------+---------+-------------+------------+-----------+
|  3 |       4 | 3133.08 |     2517.02 |    616.064 |   90031.4 |
+----+---------+---------+-------------+------------+-----------+
|  4 |       5 | 3133.08 |     2533.77 |    599.309 |   87497.6 |
+----+---------+---------+-------------+------------+-----------+
|  5 |       6 | 3133.08 |     2550.64 |    582.442 |   84947   |
+----+---------+---------+-------------+------------+-----------+
|  6 |       7 | 3133.08 |     2567.62 |    565.464 |   82379.3 |
+----+---------+---------+-------------+------------+-----------+
|  7 |       8 | 3133.08 |     2584.71 |    548.372 |   79794.6 |
+----+---------+---------+-------------+------------+-----------+
|  8 |       9 | 3133.08 |     2601.91 |    531.166 |   77192.7 |
+----+---------+---------+-------------+------------+-----------+
|  9 |      10 | 3133.08 |     2619.23 |    513.846 |   74573.5 |
+----+---------+---------+-------------+------------+-----------+
| 10 |      11 | 3133.08 |     2636.67 |    496.411 |   71936.8 |
+----+---------+---------+-------------+------------+-----------+
| 11 |      12 | 3133.08 |     2654.22 |    478.859 |   69282.6 |
+----+---------+---------+-------------+------------+-----------+
```

---

## 📈 Metrics Exposure

Prometheus metrics are served on `http://localhost:8000/metrics`:

```text
# HELP loan_approvals Total loan approvals/rejections
# TYPE loan_approvals counter
loan_approvals{status="approved"} 1.0
```

---

## 🔬 Unit Testing

```bash
pytest test_emi_calculator.py
```

---

## 📌 Future Enhancements

* Docker-based deployment
* GUI-first redesign (Streamlit-native)
* Advanced credit score modeling using credit history time series
* RESTful API integration using FastAPI or Flask

---

## 📜 License

This project is developed as part of academic coursework and is open for educational reuse under the MIT License.

---

## 🙋‍♀️ About the Author

**Rupali Ravindra Shetye**
Master's Student in Artificial Intelligence

Long Island University, Brooklyn

📧 [LinkedIn](https://www.linkedin.com/in/rupa-shetye/) 

---

> 🧠 *“Empowering responsible finance with AI-driven precision.”*

