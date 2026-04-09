def test_resource_path():
    from pyseq2500.utils import RESOURCE_PATH

    assert "pyseq_core" not in str(RESOURCE_PATH)
