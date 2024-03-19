Dokumentacja pisana wraz z kodem jako docstringi

1.    18.03-24.03
  * Struktura projektu (cookiecutter)
2.    25.03-31.03
  * Podstawowa interakcja z procesem FFmpeg jako subproces
  * metody `input`, `output`
  * testy potwierdzające działanie
3.    01.04-07.04
  * metoda `filter`, `run`, `compile`
4.    08.04-14.04
  * implementacja części logiki grafu filtrów
    * jakiś podzbiór funkcjonalności np. `concat` tylko
5.    15.04-21.04
  * dalsza część logiki grafów
6.    22.04-28.04
  * poboczne metody: `probe`, `view`, `run_async`
7.    29.04-05.05
  * dodanie osobnych metod na popularne filtry np. `hfilp`.
  * ? automatyczna generacja metod z kodu źródłowego ffmpeg
8.    06.05-12.05
9.    13.05-19.05
10.    20.05-26.05
11.    27.05-02.06
12.    03.06-09.06
13.    10.06-16.06 

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
