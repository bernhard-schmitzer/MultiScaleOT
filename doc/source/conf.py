# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#sys.path.insert(0, os.path.abspath('../../python/'))

# -- Project information -----------------------------------------------------

project = 'MultiScaleOT'
copyright = '2020, Bernhard Schmitzer'
author = 'Bernhard Schmitzer'

# The full version, including alpha/beta/rc tags
release = '0.3.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.autodoc','sphinx.ext.napoleon','sphinx_gallery.gen_gallery'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

#html_context = {
#    'css_files': [
#        '_static/theme_overrides.css'
#    ]
#}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# examples
sphinx_gallery_conf = {
     'examples_dirs': '../python-examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}

# suppress matplotlib warning in galleries
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.")

