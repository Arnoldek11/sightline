import sightline
from sightline.paths import data_dir


def test_import():
    assert sightline.__version__


def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGHTLINE_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
