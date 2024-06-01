from pathlib import Path

from filters_autogen.make_file import sanitize
from generate_filters import default_imports
from generate_filters import generate_filters


def test_writing_any_file(monkeypatch, tmp_path: Path):
    """ Filename with this name should appear"""
    monkeypatch.setattr("filters_autogen.parse_ffmpeg", list) # don't actually parse anything
    target = tmp_path / "out.py"

    # create fake ffmpeg repo structure
    ffmpeg_dir = tmp_path / "FFmpeg"
    ffmpeg_dir.mkdir()
    (ffmpeg_dir / "libavutil").mkdir()
    (ffmpeg_dir / "libavutil" / "avconfig.h").touch()
    (ffmpeg_dir / "libavfilter").mkdir()
    (ffmpeg_dir / "libavfilter" / "allfilters.c").touch()

    generate_filters(ffmpeg_dir,target)
    assert target.read_text()==default_imports # shouldn't be anything more than just the imports


def test_sanitize():
    assert sanitize("as")=="_as"
    assert sanitize("with")=="_with"
    assert sanitize("123")=="_123"
    assert sanitize("0o123")=="_0o123"

    assert sanitize("seems-ok") == "seems_ok" # we don't want subtract operator

    assert sanitize("ok")=="ok"
    assert sanitize("this_is_legal")=="this_is_legal"
    assert sanitize("print")=="print" # no problem
