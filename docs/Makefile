# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PAPER         ?=
BUILDDIR      ?= build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source $(BUILDDIR)
# the i18n builder cannot share the output directory and doctrees with the other builders
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help Makefile

# Put it first so that "make help" prints help
help:
	@echo "Documentation build targets:"
	@echo "  make html       - Build HTML documentation"
	@echo "  make clean      - Clean build artifacts"
	@echo ""
	@echo "To view docs: open build/index.html"

html:
	@$(SPHINXBUILD) -b html source build

clean:
	@rm -rf $(BUILDDIR)/*
	@echo "Cleaned build artifacts"

.PHONY: html clean help
