"""
This is a dummy python file designed to fail the pipeline to verify intended behaviours.
"""

def add(a: int, b: str) -> int:
    """
    This is a dummy function designed to fail the pipeline to verify intended behaviours.

    Args:
        a (int): Integer to add.
        b (str): String to add.

    Returns:
        int: Sum of a and b, should always fail.
    """

    return a + b