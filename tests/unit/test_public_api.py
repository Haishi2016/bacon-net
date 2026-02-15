import bacon
import pytest


@pytest.mark.unit
def test_public_version_is_semver_like():
    version = bacon.__version__
    parts = version.split('.')

    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


@pytest.mark.unit
def test_expected_public_exports_exist():
    assert hasattr(bacon, 'baconNet')
    assert hasattr(bacon, 'binaryTreeLogicNet')
