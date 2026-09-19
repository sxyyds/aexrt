import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import cli


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ([], None),
        (["--objectness", "auto"], None),
        (["--objectness", "yes"], True),
        (["--objectness", "no"], False),
    ],
)
def test_build_cli_passes_objectness_tristate(tmp_path, monkeypatch, option, expected):
    captured = {}

    def fake_save(model, output, **kwargs):
        captured.update(kwargs)
        return {
            "version": 1,
            "precision": 2,
            "value_count": 1,
            "command_count": 0,
            "constant_count": 0,
            "arena_nbytes": 0,
            "file_size": 64,
            "fusion_group_count": 0,
            "shader_cache_count": 0,
            "dxil_cache_count": 0,
            "dxbc_cache_count": 0,
            "pso_cache_count": 0,
        }

    monkeypatch.setattr(cli, "save_aexrt_engine_from_onnx", fake_save)
    output = tmp_path / "model.aexrt"

    result = cli.main([
        "build",
        "model.onnx",
        "-o",
        str(output),
        "--no-pipeline-cache",
        *option,
    ])

    assert result == 0
    assert captured["objectness"] is expected
