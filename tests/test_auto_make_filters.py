from pathlib import Path

from filters_autogen import make_file
from filters_autogen.filter_classes import Filter
from filters_autogen.filter_classes import FilterOption


def test_fill_template_no_options():
    f = Filter(name="template_test",
               description="Here's a description",
               type="filter_type",
               options=[])

    code_string = make_file.fill_template(make_file.filter_template, f)

    target = (Path(__file__).parent / "assets/template1.res").read_text()

    assert target==code_string

def test_fill_template():
    f = Filter(name="template_test_2",
               description="Here's a description",
               type="filter_type",
               options=[FilterOption("option1","AV_OPT_TYPE_INT", "",{},"this option is an int"),
                        FilterOption("option2","AV_OPT_TYPE_STRING", "",{},"this option is a string")])

    code_string = make_file.fill_template(make_file.filter_template, f)

    target = (Path(__file__).parent / "assets/template2.res").read_text()

    assert target==code_string
