import logging
import os
import pickle
import subprocess
from pathlib import Path
from typing import ClassVar

import pycparser
import pycparser.c_ast
from filter_classes import Filter
from filter_classes import FilterOption
from pycparser import c_ast
from pycparser.c_parser import ParseError


class TypeDeclVisitor(c_ast.NodeVisitor):
    collected_filters: ClassVar = []

    def visit_TypeDecl(self, node):
        if node.declname and "ff_" in node.declname and "AVFilter" in node.type.names:
            TypeDeclVisitor.collected_filters.append(node.declname)


class OptionsVisitor(c_ast.NodeVisitor):
    def __init__(self, filter_name: str) -> None:
        super().__init__()
        self.options_name: str = filter_name + "_options"  # this is how ffmpeg source code does it through `AVFILTER_DEFINE_CLASS`
        print("looking for", self.options_name)
        self.options: list[FilterOption] = []

    def visit_Decl(self, node):
        if isinstance(node.type, pycparser.c_ast.ArrayDecl):
            if node.name == self.options_name and node.type.type.type.names[0] == "AVOption":  # sanity check
                offsets = set()
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
        self.found_filters: list[Filter] = []  # this holds found filter names

    def visit_Decl(self, node):
        if isinstance(node.type, c_ast.TypeDecl) and hasattr(node.type.type, "names") and "AVFilter" in node.type.type.names:
            if hasattr(node, "init") and node.init:
                for expr in node.init.exprs:
                    if expr.name[0].name == "name":
                        filter_name = expr.expr.value[1:-1]
                    if expr.name[0].name == "description":
                        filter_desc = expr.expr.value[1:-1]
                self.found_filters.append(Filter(filter_name, filter_desc, []))


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# TODO: clone ffmpeg --depth 1 here maybe

script_dir = Path.parent(os.path.realpath(__file__))
os.chdir(script_dir + "/FFmpeg")
logger.debug(f"cwd: {Path.cwd()}")

all_filters = "libavfilter/allfilters.c"

logger.info("Configuring ffmpeg...")
if not Path.exists("libavutil/avconfig.h"):
    handle = subprocess.Popen(["./configure", "--disable-x86asm"])  # this is needed to generate some header files  # noqa: S603
    handle.wait()

logger.info("Running parser")
# TODO: clone pycparser repo here for fake includes
AST = pycparser.parse_file(
    all_filters, use_cpp=True, cpp_args=["-I", ".", "-I", "../pycparser/utils/fake_libc_include", "-D__attribute__(x)="]
)

visitor = TypeDeclVisitor()
visitor.visit(AST)
print(len(TypeDeclVisitor.collected_filters))


# parse all files in `libavfilter` and look for matching names
all_filters = []

parse_errors = []

for file in os.listdir("libavfilter"):
    if file[-2:] == ".c":
        with Path("libavfilter/" + file).open() as f:
            if "AVFilter " not in f.read():
                continue  # quick skip over files without filters
        print("Parsing", file)
        try:
            AST = pycparser.parse_file(
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
                ],
            )
            visitor = AVFilterFinder()
            visitor.visit(AST)
            if len(visitor.found_filters) == 0:
                logger.warning(f"Empty filter file: {file}")
                continue
            for filter in visitor.found_filters:  # one file can have multiple filters
                visitor2 = OptionsVisitor(filter.name)
                visitor2.visit(AST)
                all_filters.append(Filter(name=filter.name, description=filter.description, options=visitor2.options))
        except subprocess.CalledProcessError:
            parse_errors.append(file)
        except ParseError as e:
            print("Parse error: ", e)
            parse_errors.append(file)
            # exit()

for i, filter in enumerate(all_filters):
    print(i, filter)

print("parse errors:", parse_errors)

with Path("all_filters.pickle").open("wb+") as f:
    pickle.dump(all_filters, f)
