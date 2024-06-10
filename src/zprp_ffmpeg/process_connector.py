import subprocess

import oslex

from .base_connector import BaseConnector
from .filter_graph import FilterParser
from .filter_graph import Stream


class ProcessConnector(BaseConnector):
    # @TODO: this isn't ideal because attacker can sideload malicious executable...
    # Maybe warn user to set this in config for safety?
    ffmpeg_executable_path = "ffmpeg"

    def __init__(self, ffmpeg_process) -> None:
        self.ffmpeg_process: subprocess.Popen = ffmpeg_process
        super().__init__()

    @staticmethod
    def compile(graph: Stream) -> str:
        """
        Builds a command for ffmpeg from FilterGraph

        :param graph: the graph to compile
        :return: a string to pass as an argument to ffmpeg
        """

        command = FilterParser().generate_result(graph)

        return "".join(command)

    @classmethod
    def run(cls, graph: Stream, extra_options: str = "") -> "BaseConnector":
        """
        Builds a command from FilterGraph, starts ffmpeg process, and passes the command.

        :return: subprocess.Popen instance
        """

        command = ProcessConnector.compile(graph) + extra_options
        print("Command:", oslex.split(command))
        ffmpeg_process = subprocess.Popen(
            [ProcessConnector.ffmpeg_executable_path, *oslex.split(command)], stdout=subprocess.PIPE, stderr=subprocess.PIPE  # noqa: S603
        )
        return cls(ffmpeg_process)

    def communicate(self):
        return self.ffmpeg_process.communicate()
