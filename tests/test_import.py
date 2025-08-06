def test_version():
    import importlib.metadata

    print(importlib.metadata.version("pyseq"))
    assert True
