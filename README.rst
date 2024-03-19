========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |github-actions| |codecov|
    * - package
      - |version| |wheel| |supported-versions| |supported-implementations| |commits-since|
.. |docs| image:: https://readthedocs.org/projects/zprp-ffmpeg/badge/?style=flat
    :target: https://readthedocs.org/projects/zprp-ffmpeg/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/Madghostek/zprp-ffmpeg/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/Madghostek/zprp-ffmpeg/actions

.. |codecov| image:: https://codecov.io/gh/Madghostek/zprp-ffmpeg/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/Madghostek/zprp-ffmpeg

.. |version| image:: https://img.shields.io/pypi/v/zprp-ffmpeg.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/zprp-ffmpeg

.. |wheel| image:: https://img.shields.io/pypi/wheel/zprp-ffmpeg.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/zprp-ffmpeg

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/zprp-ffmpeg.svg
    :alt: Supported versions
    :target: https://pypi.org/project/zprp-ffmpeg

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/zprp-ffmpeg.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/zprp-ffmpeg

.. |commits-since| image:: https://img.shields.io/github/commits-since/Madghostek/zprp-ffmpeg/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/Madghostek/zprp-ffmpeg/compare/v0.0.0...main



.. end-badges

Implementation of the successor to the ffmpeg-python library

* Free software: MIT license

Installation
============

::

    pip install zprp-ffmpeg

You can also install the in-development version with::

    pip install https://github.com/Madghostek/zprp-ffmpeg/archive/main.zip


Documentation
=============


https://zprp-ffmpeg.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
