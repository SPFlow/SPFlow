# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add the parent directory to the path so Sphinx can find the spflow package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SPFlow"
copyright = "2025, SPFlow Contributors"
author = "SPFlow Contributors"
release = "1.0.0"

# The full version, including alpha/beta/rc tags
version = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon configuration (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_method = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_keyword_only_arg_name = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary configuration
# autosummary_generate = True  # Disabled in favor of manual automodule directives
# autosummary_context = {}

# Type hints configuration
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}

# Intersphinx configuration - Link to external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# nbsphinx configuration for Jupyter notebooks
nbsphinx_execute = "always"
nbsphinx_timeout = 300
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"

# Myst configuration for Markdown support
myst_enable_extensions = ["dollarmath", "amsmath", "html_image"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {version}"
html_logo = None  # Add logo path if available
html_favicon = None  # Add favicon path if available

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#0066cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4da6ff",
        "color-brand-content": "#4da6ff",
    },
}

# Furo specific options
html_show_sphinx = True
html_show_copyright = True

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "11pt",
}

# Configure source file format
source_suffix = {
    ".rst": None,
    ".md": "myst-nb",
}

EXCLUDED_MEMBERS = {"extra_repr"}


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Globally exclude 'extra_repr' from autodoc."""
    if name in EXCLUDED_MEMBERS:
        return True  # Return True to skip this member

    return None  # Return None to let Sphinx decide


def setup(app):
    """Connect the skip_member function to the event."""
    app.connect("autodoc-skip-member", autodoc_skip_member)
