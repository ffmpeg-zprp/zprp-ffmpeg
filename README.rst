===============
Design proposal
===============
Biblioteka ma za zadanie zastąpić i rozszerzyć istniejącą bibliotekę (https://github.com/kkroening/ffmpeg-python)

Minimalną funkcjonalność, którą chcemy zaimplementować, to ta oferowana przez powyższą bibliotekę, czyli korzystanie z grafów filtrów poprzez prosty interfejs. Poza tym, zależy nam na lepszej integracji z IDE (opisy filtrów w docstringach, typy argumentów), żeby ograniczyć potrzebę krążenia po dokumentacji FFmpeg.

====================
Stack technologiczny
====================
* szablon cookiecutter: https://github.com/ionelmc/cookiecutter-pylibrary
* dokumentacja mkdocs
* linter ruff
* Poetry do budowania paczki
* tox do automatycznych testów
* CI/CD przy użyciu github actions (uruchomienie testów, budowanie paczki, automatyczny release do PyPI)

=======================
Planowany rozkład jazdy
=======================
#. 18.03-24.03
    * Struktura projektu (cookiecutter)
    * Chcemy mieć możliwość uruchomienia testów, zbudowania dokumentacji (narazie pustej), zbudowania paczki.
#. 25.03-31.03
    * Podstawowa interakcja z procesem FFmpeg jako subproces
    * metody `input`, `output`
    * testy potwierdzające działanie
#. 01.04-07.04
    * metoda `filter`, `run`, `compile`
#. 08.04-14.04
    * implementacja części logiki grafu filtrów
    * jakiś podzbiór funkcjonalności np. `concat` tylko
#. 15.04-21.04
    * dalsza część logiki grafów
#. 22.04-28.04
    * poboczne metody: `probe`, `view`, `run_async`
#. 29.04-05.05
    * dodanie osobnych metod na popularne filtry np. `hfilp`.
    * ? automatyczna generacja metod z kodu źródłowego ffmpeg

Tygodnie 8-13 przewidziane na potencjalne przesunięcia w planie.

Dokumentacja będzie pisana regularnie wraz z kodem jako docstringi


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

.. |github-actions| image:: https://github.com/ffmpeg-zprp/zprp-ffmpeg/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/ffmpeg-zprp/zprp-ffmpeg/actions

.. |codecov| image:: https://codecov.io/gh/ffmpeg-zprp/zprp-ffmpeg/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/gh/ffmpeg-zprp/zprp-ffmpeg

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

.. |commits-since| image:: https://img.shields.io/github/commits-since/ffmpeg-zprp/zprp-ffmpeg/v2.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ffmpeg-zprp/zprp-ffmpeg/compare/v2.1.0...main



.. end-badges

Implementation of the successor to the ffmpeg-python library

* Free software: MIT license

Installation
============

::

    pip install zprp-ffmpeg

You can also install the in-development version with::

    pip install https://github.com/ffpmeg-zprp/zprp-ffmpeg/archive/main.zip


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
