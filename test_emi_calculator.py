import pytest
from AdvEMICalculator import calculate_emi, validate_input

def test_calculate_emi_basic():
    assert calculate_emi(100000, 10, 12) == pytest.approx(8791.59, 0.01)

def test_calculate_emi_zero_rate():
    assert calculate_emi(120000, 0, 24) == 5000.0

@pytest.mark.parametrize("score, income, debt, loan, tenure, valid", [
    (700, 5000, 300, 100000, 36, True),
    (250, 5000, 300, 100000, 36, False),
    (700, -1000, 300, 100000, 36, False),
    (700, 5000, 6000, 100000, 36, False),
])
def test_input_validation(score, income, debt, loan, tenure, valid):
    is_valid, _ = validate_input(score, income, debt, loan, tenure)
    assert is_valid == valid
