"""Package-level smoke tests."""

import wildfire_sim


def test_import() -> None:
    assert wildfire_sim.__name__ == "wildfire_sim"
    assert isinstance(wildfire_sim.__version__, str)
