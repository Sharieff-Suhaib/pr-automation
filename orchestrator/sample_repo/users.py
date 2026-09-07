"""A tiny user store with a reporting bug, used to exercise the workflow."""


def get_user(users, name):
    """Return the profile stored for `name`."""
    return users[name]


def login(users, name, password):
    """Return True when `name` exists and the password matches."""
    user = get_user(users, name)
    return user["password"] == password
