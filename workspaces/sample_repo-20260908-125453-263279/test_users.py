from users import get_user, login

USERS = {"ada": {"password": "lovelace"}}


def test_known_user_is_returned():
    assert get_user(USERS, "ada") == {"password": "lovelace"}


def test_missing_user_returns_none():
    assert get_user(USERS, "ghost") is None


def test_login_succeeds_for_valid_credentials():
    assert login(USERS, "ada", "lovelace") is True


def test_login_rejects_unknown_user():
    assert login(USERS, "ghost", "whatever") is False
