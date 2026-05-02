"""
Unit tests for the arithmetic tools in app/tools.py.
"""

import pytest

from app.tools import add, divide, exp, greater, multiply, percentage_change, subtract


class TestAdd:
    def test_add_two_positive_numbers(self) -> None:
        """
        GIVEN two positive floats,
        WHEN add is called,
        THEN their sum is returned.
        """
        assert add(3.0, 2.0) == pytest.approx(5.0)

    def test_add_negative_numbers(self) -> None:
        """
        GIVEN a positive and a negative float,
        WHEN add is called,
        THEN their sum is returned.
        """
        assert add(3.0, -1.0) == pytest.approx(2.0)

    def test_add_zero(self) -> None:
        """
        GIVEN a number and zero,
        WHEN add is called,
        THEN the original number is returned.
        """
        assert add(5.0, 0.0) == pytest.approx(5.0)


class TestSubtract:
    def test_subtract_smaller_from_larger(self) -> None:
        """
        GIVEN a minuend larger than the subtrahend,
        WHEN subtract is called,
        THEN the positive difference is returned.
        """
        assert subtract(10.0, 3.0) == pytest.approx(7.0)

    def test_subtract_larger_from_smaller_returns_negative(self) -> None:
        """
        GIVEN a minuend smaller than the subtrahend,
        WHEN subtract is called,
        THEN a negative result is returned.
        """
        assert subtract(3.0, 10.0) == pytest.approx(-7.0)

    def test_subtract_zero(self) -> None:
        """
        GIVEN a number and a subtrahend of zero,
        WHEN subtract is called,
        THEN the original number is returned.
        """
        assert subtract(5.0, 0.0) == pytest.approx(5.0)


class TestMultiply:
    def test_multiply_two_positive_numbers(self) -> None:
        """
        GIVEN two positive floats,
        WHEN multiply is called,
        THEN their product is returned.
        """
        assert multiply(4.0, 3.0) == pytest.approx(12.0)

    def test_multiply_by_zero(self) -> None:
        """
        GIVEN a multiplicand and a multiplier of zero,
        WHEN multiply is called,
        THEN zero is returned.
        """
        assert multiply(5.0, 0.0) == pytest.approx(0.0)

    def test_multiply_negative_numbers(self) -> None:
        """
        GIVEN two negative floats,
        WHEN multiply is called,
        THEN a positive product is returned.
        """
        assert multiply(-3.0, -4.0) == pytest.approx(12.0)


class TestDivide:
    def test_divide_two_positive_numbers(self) -> None:
        """
        GIVEN a positive numerator and a positive denominator,
        WHEN divide is called,
        THEN the quotient is returned.
        """
        assert divide(10.0, 4.0) == pytest.approx(2.5)

    def test_divide_by_zero_returns_none(self) -> None:
        """
        GIVEN a non-zero numerator and a zero denominator,
        WHEN divide is called,
        THEN None is returned.
        """
        assert divide(10.0, 0.0) is None

    def test_divide_negative_by_positive(self) -> None:
        """
        GIVEN a negative numerator and a positive denominator,
        WHEN divide is called,
        THEN a negative quotient is returned.
        """
        assert divide(-9.0, 3.0) == pytest.approx(-3.0)


class TestPercentageChange:
    def test_percentage_change_increase(self) -> None:
        """
        GIVEN a base_value smaller than the new_value,
        WHEN percentage_change is called,
        THEN a positive percentage is returned.
        """
        assert percentage_change(100.0, 150.0) == pytest.approx(50.0)

    def test_percentage_change_decrease(self) -> None:
        """
        GIVEN a base_value larger than the new_value,
        WHEN percentage_change is called,
        THEN a negative percentage is returned.
        """
        assert percentage_change(200.0, 100.0) == pytest.approx(-50.0)

    def test_percentage_change_no_change(self) -> None:
        """
        GIVEN identical base_value and new_value,
        WHEN percentage_change is called,
        THEN zero is returned.
        """
        assert percentage_change(50.0, 50.0) == pytest.approx(0.0)

    def test_percentage_change_zero_base_returns_none(self) -> None:
        """
        GIVEN a zero base_value,
        WHEN percentage_change is called,
        THEN None is returned.
        """
        assert percentage_change(0.0, 100.0) is None


class TestGreater:
    def test_greater_returns_first_when_larger(self) -> None:
        """
        GIVEN a first_value larger than second_value,
        WHEN greater is called,
        THEN first_value is returned.
        """
        assert greater(10.0, 3.0) == pytest.approx(10.0)

    def test_greater_returns_second_when_larger(self) -> None:
        """
        GIVEN a second_value larger than first_value,
        WHEN greater is called,
        THEN second_value is returned.
        """
        assert greater(3.0, 10.0) == pytest.approx(10.0)

    def test_greater_returns_value_when_equal(self) -> None:
        """
        GIVEN two equal floats,
        WHEN greater is called,
        THEN the value is returned.
        """
        assert greater(5.0, 5.0) == pytest.approx(5.0)


class TestExp:
    def test_exp_positive_base_and_exponent(self) -> None:
        """
        GIVEN a positive base and a positive integer exponent,
        WHEN exp is called,
        THEN the correct result is returned.
        """
        assert exp(2.0, 3.0) == pytest.approx(8.0)

    def test_exp_base_to_zero_exponent(self) -> None:
        """
        GIVEN any base and an exponent of zero,
        WHEN exp is called,
        THEN one is returned.
        """
        assert exp(5.0, 0.0) == pytest.approx(1.0)

    def test_exp_fractional_exponent(self) -> None:
        """
        GIVEN a base and a fractional exponent,
        WHEN exp is called,
        THEN the correct root is returned.
        """
        assert exp(9.0, 0.5) == pytest.approx(3.0)

    def test_exp_negative_base_fractional_exponent_returns_none(self) -> None:
        """
        GIVEN a negative base and a fractional exponent,
        WHEN exp is called,
        THEN None is returned as the result is not real-valued.
        """
        assert exp(-4.0, 0.5) is None

    def test_exp_zero_base_negative_exponent_returns_none(self) -> None:
        """
        GIVEN a base of zero and a negative exponent,
        WHEN exp is called,
        THEN None is returned as the result is undefined.
        """
        assert exp(0.0, -1.0) is None
