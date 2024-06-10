
def template_test_2(graph: Stream, option1: Optional[int] = None, option2: Optional[str] = None):
    """Here's a description
    :param int option1: this option is an int
    :param str option2: this option is a string"""
    graph.append(Filter(command="template_test_2",filter_type="filter_type",params=[FilterOption(name="option1",value=option1), FilterOption(name="option2",value=option2)]))
    return graph
