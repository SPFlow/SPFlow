# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
import sphinx_gallery

# -- Project information -----------------------------------------------------
project = "SPFlow"
copyright = "2020, Alejandro Molina, Antonio Vergari, Karl Stelzner, Robert Peharz, Nicola Di Mauro, Kristian Kersting"
author = "Alejandro Molina, Antonio Vergari, Karl Stelzner, Robert Peharz, Nicola Di Mauro, Kristian Kersting"

# Get __version__ from _meta
from spn._meta import __version__

version = __version__
release = __version__

extensions = [
    "sphinx.ext.linkcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "env"]

pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
html_logo = "../../Documentation/logo/spflow_logoSquare.png"

# -- Extension configuration -------------------------------------------------

autosummary_generate = True
autodoc_default_options = {"undoc-members": None}

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# -- Options for todo extension ----------------------------------------------
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Linkcode extension
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return "https://github.com/SPFlow/SPFlow/blob/master/src/%s.py" % filename

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# sphinx_gallery.gen_gallery settings
sphinx_gallery_conf = {
    "doc_module": "spn",
    "backreferences_dir": os.path.join("generated"),
    "reference_url": {"spn": None},
    "remove_config_comments": True,
}
