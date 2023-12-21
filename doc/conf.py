# encoding: utf8
#
# mne-rsa documentation build configuration file, created by
# sphinx-quickstart on Fri Nov 24 14:06:52 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

from numpydoc import numpydoc, docscrape  # noqa
import mne
import mne_rsa

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

# generate autosummary even if no references
autosummary_generate = True
autosummary_imported_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8-sig"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "mne-rsa"
copyright = "2024, Aalto University"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = mne_rsa.__version__
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/wmvanvliet/mne-rsa",
            icon="fa-brands fa-square-github",
        ),
    ],
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "article_header_start": [],  # disable breadcrumbs
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
}

html_context = {
    "default_mode": "auto",
    # next 3 are for the "edit this page" button
    "github_user": "wmvanvliet",
    "github_repo": "mne-mne",
    "github_version": "main",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# Output file base name for HTML help builder.
htmlhelp_basename = "mnersadoc"

# -- Options for LaTeX output ---------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "mne-rsa.tex", "MNE-RSA Documentation", "Marijn van Vliet", "manual"),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "mne_rsa", "MNE-RSA Documentation", ["Marijn van Vliet"], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "mne_rsa",
        "MNE-RSA Documentation",
        "Marijn van Vliet",
        "mne_rsa",
        "Representational Similarity Analysis",
        "Miscellaneous",
    ),
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "mne": ("https://mne.tools/stable/", None),
    "mne_rsa": ("https://users.aalto.fi/~vanvlm1/mne-rsa/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "backreferences_dir": "generated",
    "image_scrapers": ["matplotlib", mne.viz._brain._BrainScraper()],
    "reference_url": dict(mne=None),
}
