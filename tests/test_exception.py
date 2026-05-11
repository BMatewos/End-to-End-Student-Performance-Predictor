import sys

from src.exception import CustomException


def test_custom_exception_contains_error_details():
    try:
        raise ValueError("sample error")
    except Exception as exc:
        custom_exc = CustomException(exc, sys)

    message = str(custom_exc)
    assert "sample error" in message
    assert "Error occurred in python script" in message
