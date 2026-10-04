"""Reproduction test for `sample_repo`, replayed by the `stub` backend.

Copied into the working copies as `test_issue_reproduction.py`; the file name
here deliberately does not match pytest's `test_*.py` / `*_test.py` patterns.
"""

from users import get_user, login

USERS = {"ada": {"password": "lovelace"}}


def test_issue_unknown_user_lookup_returns_none():
    assert get_user(USERS, "nobody") is None


def test_issue_login_with_unknown_user_is_rejected():
    assert login(USERS, "nobody", "secret") is False
