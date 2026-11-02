from prismadv.utils import get_current_folder


def test_get_current_folder():
    assert get_current_folder().name == "tests"
