import subprocess


def probe(filename: str, format="json", **kwargs) -> str:
    """
    Calls ffprobe on given filename and returns output as string.

    :param filename: the graph to compile
    :param format: format of output stream. Available options: 'json', 'default', 'flat', 'csv', 'ini', 'xml' (default: json)
    :return: ffprobe output in given format
    """
    ffprobe_executable_path = "ffprobe"
    args = [ffprobe_executable_path, filename, "-of", format]
    args += ["-show_format", "-show_streams"]

    # parsing kwargs
    for key, value in kwargs.items():
        args.append(f"-{key}")
        if value is not None:
            args.append(str(value))

    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
    out, err = p.communicate()

    if p.returncode != 0:
        raise Exception("ffprobe", out, err)

    return out.decode("utf-8")
