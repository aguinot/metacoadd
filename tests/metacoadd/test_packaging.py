import metacoadd


def test_version():
    """Check to see that we can get the package version."""
    assert metacoadd.__version__ is not None
