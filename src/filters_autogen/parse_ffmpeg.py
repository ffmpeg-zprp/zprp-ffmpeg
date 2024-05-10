import logging
import os
import pickle
import subprocess
from pathlib import Path
from typing import ClassVar

import pycparser.c_ast  # type: ignore
from filter_classes import Filter
from filter_classes import FilterOption
from pycparser import c_ast
from pycparser import parse_file  # type: ignore # ????? mypy freaking out
from pycparser.c_parser import ParseError  # type: ignore
from tqdm import tqdm


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
        self.options: list[FilterOption] = []

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
                        if isinstance(option.exprs[2], pycparser.c_ast.Constant):
                            offset = option.exprs[2].value
                        elif isinstance(option.exprs[2], pycparser.c_ast.FuncCall):
                            offset = option.exprs[2].args.exprs[1].name
                        else:
                            offset = 0
                            logger.error("invalid offset")
                        if offset in offsets:
                            continue  # ignore aliases
                        offsets.add(offset)
                        option_type = option.exprs[3].name
                        if option_type == "AV_OPT_TYPE_CONST":
                            # flags shouldn't be separate arguments. Could add them to docstring though.
                            continue
                        self.options.append(FilterOption(option_name, option_type, option_desc))

                    except AttributeError:
                        logger.error("Unsupported filter option (probably name is not constant)")


class AVFilterFinder(c_ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.found_filters: list[tuple[Filter, str, str]] = []  # this holds found filters and their input struct name to be found later

    def visit_Decl(self, node):
        if isinstance(node.type, c_ast.TypeDecl) and hasattr(node.type.type, "names") and "AVFilter" in node.type.type.names:
            if hasattr(node, "init") and node.init:
                filter_input_struct = None  # sometimes this is dynamic
                filter_output_struct = None  # here i have no idea how it's possible
                for expr in node.init.exprs:
                    if expr.name[0].name == "name":
                        filter_name = expr.expr.value[1:-1]
                    if expr.name[0].name == "description":
                        filter_desc = expr.expr.value[1:-1]
                    if expr.name[0].name == "inputs":
                        if isinstance(expr.expr, c_ast.Constant):
                            filter_input_struct = None
                        else:
                            filter_input_struct = expr.expr.name
                    if expr.name[0].name == "outputs":
                        if isinstance(expr.expr, c_ast.Constant):
                            filter_output_struct = None
                        else:
                            filter_output_struct = expr.expr.name

                self.found_filters.append((Filter(filter_name, filter_desc, None, []), filter_input_struct, filter_output_struct))


def parse_source_code(save_pickle=False, debug=False) -> list[Filter]:
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
        cpp_args=["-I", ".", "-I", "../pycparser/utils/fake_libc_include", "-D__attribute__(x)=", "-D__restrict="],
    )

    visitor = TypeDeclVisitor()
    visitor.visit(AST)
    print(len(TypeDeclVisitor.collected_filters))

    # parse all files in `libavfilter` and look for matching names
    parsed_filters = []

    parse_errors = []

    for file in tqdm(os.listdir("libavfilter")):
        if file[-2:] == ".c" and file == "vf_scale.c":
            with Path("libavfilter/" + file).open() as f:
                if "AVFilter " not in f.read():
                    continue  # quick skip over files without filters
            logger.debug(f"Parsing {file}")
            try:
                AST = parse_file(
                    "libavfilter/" + file,
                    use_cpp=True,
                    cpp_args=[
                        "-I",
                        "../ffmpeg_patches",
                        "-I",
                        "../pycparser/utils/fake_libc_include",
                        "-I",
                        ".",
                        "-include",
                        "libavfilter/avfilter.h",
                        "-D__attribute__(x)=",
                        "-D__THROW=",
                        "-D__END_DECLS=",
                        "-D__inline=",
                        "-D__extension__=",
                        "-D__asm__(...)=",
                    ],
                )
                visitor = AVFilterFinder()
                visitor.visit(AST)
                if len(visitor.found_filters) == 0:
                    logger.warning(f"Empty filter file: {file}")
                    continue
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

                    parsed_filters.append(
                        Filter(name=filter.name, description=filter.description, type=filter_type, options=option_visitor.options)
                    )
            except subprocess.CalledProcessError:
                parse_errors.append(file)
            except ParseError as e:
                print("Parse error: ", e)
                parse_errors.append(file)
                # exit()

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
