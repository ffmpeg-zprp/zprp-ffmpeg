import logging
import os
import pickle
import subprocess
import typing
from pathlib import Path
from typing import ClassVar

import pycparser.c_ast  # type: ignore
from pycparser import c_ast
from pycparser import parse_file  # type: ignore # ????? mypy freaking out
from pycparser.c_parser import ParseError  # type: ignore
from tqdm import tqdm

from filters_autogen import filter_classes as fc

patches_path = Path(__file__).parent / "ffmpeg_patches"
fake_libc_path = Path(__file__).parent / "pycparser" / "utils" / "fake_libc_include"
FFmpeg_source = Path(__file__).parent / "FFmpeg"
print(str(fake_libc_path))


class TypeDeclVisitor(c_ast.NodeVisitor):
    collected_filters: ClassVar = []

    def visit_TypeDecl(self, node):
        if node.declname and "ff_" in node.declname and "AVFilter" in node.type.names:
            TypeDeclVisitor.collected_filters.append(node.declname)


class StructFinder(c_ast.NodeVisitor):
    def __init__(self, struct_name: str) -> None:
        super().__init__()
        self.struct_name: str = struct_name

    def visit_Decl(self, node):
        if isinstance(node.type, pycparser.c_ast.ArrayDecl):
            if hasattr(node.type.type.type, "names") and node.type.type.type.names[0] == "AVFilterPad":
                if node.name == self.struct_name:
                    if node.type.type.declname == "ff_video_default_filterpad":
                        StructFinder.type = "AVMEDIA_TYPE_VIDEO"
                    elif node.type.type.declname == "ff_audio_default_filterpad":
                        StructFinder.type = "AVMEDIA_TYPE_AUDIO"
                    else:
                        for field in node.init.exprs[0]:
                            if field.name[0].name == "type":
                                StructFinder.type = field.expr.name
                                break


class OptionsVisitor(c_ast.NodeVisitor):
    def __init__(self, filter_name: str) -> None:
        super().__init__()
        self.options_name: str = filter_name + "_options"  # this is how ffmpeg source code does it through `AVFILTER_DEFINE_CLASS`
        logger = logging.getLogger(__name__)
        logger.debug(f"looking for {self.options_name}")
        self.options: list[fc.FilterOption] = []

    def append_to_unit(self, unit: str, constant: str, value: int):
        """Finds filter option with given unit name and gives it new constant option"""
        for option in self.options:
            if option.unit == unit:
                option.available_values[constant] = value
                option.type = "AV_OPT_TYPE_STRING"

    def visit_Decl(self, node):
        if isinstance(node.type, pycparser.c_ast.ArrayDecl):
            if node.name == self.options_name and node.type.type.type.names[0] == "AVOption":
                offsets = set()
                logger = logging.getLogger(__name__)
                for option in node.init.exprs:
                    if len(option.exprs) == 1:
                        break  # NULL terminator
                    try:
                        option_name = option.exprs[0].value[1:-1]  # those are strings with quotes
                        option_desc = option.exprs[1].value[1:-1]
                        option_type = option.exprs[3].name
                        if isinstance(option.exprs[2], pycparser.c_ast.Constant):
                            offset = option.exprs[2].value
                        elif isinstance(option.exprs[2], pycparser.c_ast.FuncCall):
                            offset = option.exprs[2].args.exprs[1].name
                        else:
                            offset = 0
                            logger.error("invalid offset")
                        if offset in offsets and option_type != "AV_OPT_TYPE_CONST":
                            continue  # ignore aliases, but CONST named params don't count
                        offsets.add(offset)
                        for expr in option.exprs:
                            if isinstance(expr, pycparser.c_ast.NamedInitializer) and expr.name[0].name == "unit":
                                unit = expr.expr.value
                                break
                        else:
                            unit = ""  # sometimes there is no unit given
                        if option_type == "AV_OPT_TYPE_CONST":
                            # append this to the parent field, and change parent field to string
                            # default_value = option.exprs[4].value  # @TODO: this could be stored for docstrings
                            self.append_to_unit(unit, option_name, 0)
                            continue  # this is not a new option
                        self.options.append(
                            fc.FilterOption(name=option_name, type=option_type, unit=unit, description=option_desc, available_values={})
                        )

                    except AttributeError:
                        logger.error("Unsupported filter option (probably name is not constant)")


class AVFilterFinder(c_ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.found_filters: list[tuple[fc.Filter, str, str]] = []  # this holds found filters and their input struct name to be found later

    def visit_Decl(self, node):
        if isinstance(node.type, c_ast.TypeDecl) and hasattr(node.type.type, "names") and "AVFilter" in node.type.type.names:
            if hasattr(node, "init") and node.init:
                filter_input_struct = ""  # sometimes this is dynamic
                filter_output_struct = ""  # here i have no idea how it's possible
                for expr in node.init.exprs:
                    if expr.name[0].name == "name":
                        filter_name = expr.expr.value[1:-1]
                    if expr.name[0].name == "description":
                        filter_desc = expr.expr.value[1:-1]
                    if expr.name[0].name == "inputs":
                        if isinstance(expr.expr, c_ast.Constant):
                            filter_input_struct = ""
                        else:
                            filter_input_struct = expr.expr.name
                    if expr.name[0].name == "outputs":
                        if isinstance(expr.expr, c_ast.Constant):
                            filter_output_struct = ""
                        else:
                            filter_output_struct = expr.expr.name

                self.found_filters.append((fc.Filter(filter_name, filter_desc, "", []), filter_input_struct, filter_output_struct))


def parse_one_file(file_path: Path) -> typing.List[fc.Filter]:
    """Parses contents of a single .c file and returns any filters it found
    :param file_path: path to file

    :return: List of filters (one file can have multiple)"""
    logger = logging.getLogger(__name__)
    filters: typing.List[fc.Filter] = []
    with file_path.open() as f:
        if "AVFilter " not in f.read():
            return []  # quick skip over files without filters
        logger.debug(f"Parsing {file_path}")
        AST = parse_file(
            str(file_path),
            use_cpp=True,
            cpp_args=[
                "-I",
                str(patches_path),
                "-I",
                str(fake_libc_path),
                "-I",
                str(file_path.parent),  # for tests, libavfilter dir
                "-I",
                str(file_path.parent.parent),  # and the fake FFmpeg fir
                "-I",
                str(FFmpeg_source),
                "-I",
                ".",
                "-D__attribute__(x)=",
                "-D__THROW=",
                "-D__END_DECLS=",
                "-D__inline=",
                "-D__extension__=",
                "-D__asm__(...)=",
            ],  # type: ignore # false positive
        )
        visitor = AVFilterFinder()
        visitor.visit(AST)
        if len(visitor.found_filters) == 0:
            logger.warning(f"Empty filter file: {file_path}")
            return []
        for filter, input_struct, output_struct in visitor.found_filters:  # one file can have multiple filters
            option_visitor = OptionsVisitor(filter.name)
            option_visitor.visit(AST)
            input_visitor = StructFinder(input_struct)
            input_visitor.visit(AST)
            input_type = input_visitor.type
            output_visitor = StructFinder(output_struct)
            output_visitor.visit(AST)
            if input_visitor.type is None and output_visitor.type == "AVMEDIA_TYPE_VIDEO":
                filter_type = "VIDEO_SOURCE"
            else:
                filter_type = input_type
            filters.append(fc.Filter(name=filter.name, description=filter.description, type=filter_type, options=option_visitor.options))
    return filters


def parse_source_code(save_pickle=False, debug=False) -> list[fc.Filter]:
    logger = logging.getLogger(__name__)
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    # TODO: clone ffmpeg --depth 1 here maybe

    script_dir = Path(__file__).parent
    os.chdir(str(script_dir) + "/FFmpeg")
    logger.debug(f"cwd: {Path.cwd()}")

    all_filters = "libavfilter/allfilters.c"

    logger.info("Configuring ffmpeg...")
    if not Path("libavutil/avconfig.h").exists():
        handle = subprocess.Popen(["./configure", "--disable-x86asm"])  # this is needed to generate some header files  # noqa: S603
        handle.wait()

    logger.info("Running parser")
    # TODO: clone pycparser repo here for fake includes
    AST = parse_file(
        all_filters,
        use_cpp=True,
        cpp_args=["-I", ".", "-I", "../pycparser/utils/fake_libc_include", "-D__attribute__(x)=", "-D__restrict="],  # type: ignore
    )

    visitor = TypeDeclVisitor()
    visitor.visit(AST)
    print(len(TypeDeclVisitor.collected_filters))

    # parse all files in `libavfilter` and look for matching names
    parsed_filters = []

    parse_errors = []

    for file in tqdm(os.listdir("libavfilter")):
        if file[-2:] == ".c":
            try:
                parsed_filters.extend(parse_one_file(Path("libavfilter") / file))
            except subprocess.CalledProcessError:
                parse_errors.append(file)
            except ParseError as e:
                print("Parse error: ", e)
                parse_errors.append(file)

    for i, filter in enumerate(parsed_filters):
        print(i, filter)

    logger.debug(f"parse errors: {parse_errors}")

    if save_pickle:
        with Path("all_filters.pickle").open("wb+") as f:
            pickle.dump(parsed_filters, f)
    return parsed_filters


if __name__ == "__main__":
    # easy to run from shell
    parse_source_code(save_pickle=True)
