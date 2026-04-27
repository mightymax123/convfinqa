"""
Arithmetic tools for the ConvFinQA financial question-answering agent.

Each tool is a plain function registered on the Agent. All tools log their
inputs and result at DEBUG level.
"""

from loguru import logger


def add(augend: float, addend: float) -> float:
    """Add two numbers together.

    Args:
        augend: The first number.
        addend: The second number to add to the first.

    Returns:
        The sum of augend and addend.
    """
    result = augend + addend
    logger.debug(f"add({augend}, {addend}) = {result}")
    return result


def subtract(minuend: float, subtrahend: float) -> float:
    """Subtract subtrahend from minuend (minuend - subtrahend).

    Args:
        minuend: The value to subtract from.
        subtrahend: The value to subtract.

    Returns:
        The difference minuend minus subtrahend.
    """
    result = minuend - subtrahend
    logger.debug(f"subtract({minuend}, {subtrahend}) = {result}")
    return result


def multiply(multiplicand: float, multiplier: float) -> float:
    """Multiply two numbers together.

    Args:
        multiplicand: The first number.
        multiplier: The number to multiply by.

    Returns:
        The product of multiplicand and multiplier.
    """
    result = multiplicand * multiplier
    logger.debug(f"multiply({multiplicand}, {multiplier}) = {result}")
    return result


def divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator (numerator / denominator).

    Args:
        numerator: The value to be divided.
        denominator: The value to divide by.

    Returns:
        The quotient of numerator divided by denominator.

    Raises:
        ValueError: If denominator is zero.
    """
    if denominator == 0:
        raise ValueError("Cannot divide by zero.")
    result = numerator / denominator
    logger.debug(f"divide({numerator}, {denominator}) = {result}")
    return result


def percentage_change(base_value: float, new_value: float) -> float:
    """Calculate the percentage change from base_value to new_value.

    Returns a positive number for an increase and a negative number for a
    decrease. For example, a change from 100 to 150 returns 50.0.

    Args:
        base_value: The original starting value.
        new_value: The new value to compare against the base.

    Returns:
        The percentage change as a float (e.g. 50.0 for a 50% increase).

    Raises:
        ValueError: If base_value is zero.
    """
    if base_value == 0:
        raise ValueError("Cannot calculate percentage change from a zero base value.")
    result = ((new_value - base_value) / base_value) * 100
    logger.debug(f"percentage_change(base_value={base_value}, new_value={new_value}) = {result}")
    return result


def greater(first_value: float, second_value: float) -> float:
    """Return the greater of two values.

    Args:
        first_value: The first value to compare.
        second_value: The second value to compare.

    Returns:
        The larger of first_value and second_value.
    """
    result = first_value if first_value > second_value else second_value
    logger.debug(f"greater({first_value}, {second_value}) = {result}")
    return result


def exp(base: float, exponent: float) -> float:
    """Raise base to the given exponent (base ** exponent).

    Useful for compound growth calculations, e.g. exp(1.05, 3) for 3 years
    of 5% annual growth.

    Args:
        base: The base value.
        exponent: The power to raise the base to.

    Returns:
        base raised to the exponent.
    """
    result = base**exponent
    logger.debug(f"exp({base}, {exponent}) = {result}")
    return result
