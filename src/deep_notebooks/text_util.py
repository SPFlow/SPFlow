import os.path
from IPython.display import display, Markdown

from deep_notebooks.nalgene.generate import fix_sentence, generate_from_file


def printmd(string=''):
    '''
    a display wrapper for Jupyter notebooks
    TODO: move to ba_plot
    :param string: a markdown string
    :return: a jupyter display object
    '''
    display(Markdown(str(string)))


def get_nlg_phrase(base_dir, file_name):
    phrase = fix_sentence(generate_from_file(base_dir, file_name)[1].raw_str)
    return phrase


def deep_join(deep_list, string):
    is_string = [isinstance(x, str) for x in deep_list]
    if all(is_string):
        return string.join(deep_list)

    result = [x if isinstance(x, str) else deep_join(x, string) for x in
              deep_list]
    return string.join(result)


def colored_string(string, color):
    return '<span style="color:' + color + '">' + string + '</span>'


def strip_dataset_name(string):
    return os.path.splitext(os.path.basename(string))[0]
