import shlex
import subprocess

from .BaseConnector import BaseConnector
from .FilterGraph import FilterGraph


class ProcessConnector(BaseConnector):

    # @TODO: this isn't ideal because attacker can sideload malicious executable...
    # Maybe warn user to set this in config for safety?
    ffmpeg_executable = "ffmpeg"

    @staticmethod
    def compile(graph: FilterGraph) -> str:
        """
        Builds a command for ffmpeg from FilterGraph

        :param graph: the graph to compile
        :return: a string to pass as an argument to ffmpeg
        """
        # @TODO: implement this once FilterGraph is done
        return "-i input.mp4 --hflip -y --out output.mpt"

    @staticmethod
    def run(graph: FilterGraph) -> subprocess.Popen:
        """
        Builds a command from FilterGraph, starts ffmpeg process, and passes the command.

        :return: subprocess.Popen instance
        """

        command = ProcessConnector.compile(graph)
        return subprocess.run([ProcessConnector.ffmpeg_executable, *shlex.split(command)])  # noqa: S603
