# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'ml-examplemodel-package'
copyright = '2025, Manuela Detomaso'
author = 'Manuela Detomaso'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",                # Only if you're using Markdown
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc settings
autoapi_type = 'python'
autoapi_dirs = ['../../src/diabete_prediction']
autoapi_keep_files = True  # helpful for debugging
autoapi_add_toctree_entry = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
#html_static_path = ['_static']
# Show type hints inline
autodoc_typehints = "description"

