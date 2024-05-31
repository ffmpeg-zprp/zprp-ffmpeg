import hashlib
import pathlib

import os

import pytest

import zprp_ffmpeg as ffmpeg
import videokf as vf


def get_md5(file: str) -> str:
    with open(file, 'rb') as file_to_load:
        content = file_to_load.read()
    return hashlib.md5(content).hexdigest()


def compare_keyframes(out_file: str, keyframes_path: str, pytest_tempdir: str):
    out_keyframes = pytest_tempdir + "/keyframes"
    # extract keyframes
    vf.extract_keyframes(out_file, output_dir_keyframes=out_keyframes)
    # compare
    test_files = []
    extracted_keyframes = []
    for root, dirs, files in os.walk(keyframes_path):
        for filename in files:
            test_files.append(os.path.join(root, filename))
    for root, dirs, files in os.walk(out_keyframes):
        for filename in files:
            extracted_keyframes.append(os.path.join(root, filename))
    test_files.sort()
    extracted_keyframes.sort()
    for expected, actual in zip(test_files, extracted_keyframes):
        assert get_md5(expected) == get_md5(actual)


@pytest.mark.starts_process
def test_input(tmp_path):
    # prepare paths
    test_dir = str(pathlib.Path(__file__).parent.resolve())
    video_path = test_dir + "/assets/videos/in.mp4"
    keyframes_path = test_dir + "/assets/keyframes/test_input"
    out_file = str(tmp_path) + "/out.mp4"

    # run ffmpeg
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.hflip(stream)
    stream = ffmpeg.output(stream, out_file)
    ffmpeg.run(stream)

    compare_keyframes(out_file, keyframes_path, str(tmp_path))
