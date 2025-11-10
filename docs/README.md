# SPFlow Documentation

This directory contains the Sphinx documentation for SPFlow.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e .[dev]
# or if using uv:
uv sync --extra dev
```

This will install Sphinx, Furo theme, and all necessary extensions.

### Build HTML Documentation

Using Make:

```bash
cd docs
make html
```

Or using sphinx-build directly:

```bash
cd docs
python3 -m sphinx -b html . _build/html
```

The generated HTML documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Clean Build Directory

```bash
cd docs
make clean
```

### Live Reload (Development)

For live reloading during documentation development:

```bash
pip install sphinx-autobuild
cd docs
make livehtml
```

This will start a local server and automatically rebuild the docs when files change.

## Documentation Structure

```
docs/
├── index.rst              # Main documentation page
├── installation.rst       # Installation guide
├── quickstart.rst         # Quick start guide
├── architecture.rst       # Architecture guide
├── api/                   # API reference
│   ├── index.rst
│   ├── modules.rst        # Inner nodes (Sum, Product)
│   ├── leaf.rst           # Leaf distributions
│   ├── distributions.rst  # Distribution classes
│   ├── learning.rst       # Learning algorithms
│   ├── meta.rst           # Metadata system
│   ├── utils.rst          # Utilities
│   └── exceptions.rst     # Exceptions
├── tutorials/             # Jupyter notebook tutorials
│   ├── index.rst
│   ├── user_guide.ipynb
│   └── user_guide2.ipynb
├── examples/              # Code examples
│   └── index.rst
├── _static/               # Static files (CSS, images)
└── _build/                # Build output (generated)
```

## Updating Documentation

### Adding New Pages

1. Create a new `.rst` file in the appropriate directory
2. Add it to the relevant `toctree` directive in `index.rst` or section index
3. Rebuild the documentation

### API Documentation

API documentation is automatically generated using Sphinx autodoc. To add documentation for a new module:

1. Add the module path to the appropriate file in `docs/api/`
2. Use the `.. autoclass::` or `.. autofunction::` directives
3. Ensure docstrings follow Google-style format

### Jupyter Notebooks

Notebooks in `docs/tutorials/` are rendered using nbsphinx:

- Notebooks are executed during build (currently disabled in conf.py)
- Add new notebooks to `docs/tutorials/index.rst` toctree
- Ensure notebooks have a title in the first cell (Markdown heading)

## Configuration

The main configuration file is `conf.py`, which contains:

- Sphinx extensions (autodoc, napoleon, nbsphinx, etc.)
- Theme configuration (Furo)
- Project metadata
- Build options

## Troubleshooting

### Import Errors

If you see import errors during build, ensure:
- SPFlow is installed in your environment
- All dependencies are installed
- You're running from the correct directory

### Missing Modules

Some warnings about missing modules are expected for placeholder files that haven't been implemented yet.

### Notebook Rendering Issues

If notebooks don't render correctly:
- Check that `nbsphinx` is installed
- Verify the notebook has a valid title
- Check `conf.py` for `nbsphinx_execute` settings
