### Changelog

All notable changes to this project will be documented in this file. Dates are displayed in UTC.

#### [v1.1.0](https://github.com/Madghostek/zprp-ffmpeg/compare/v1.0.0...v1.1.0)

> 25 May 2024

- feat: Add ffprobe function [`#4`](https://github.com/Madghostek/zprp-ffmpeg/pull/4)
- add ffmpeg headers for tests [`881d6ea`](https://github.com/Madghostek/zprp-ffmpeg/commit/881d6ea0bdcfccdaed53ef87b019ce1982911cf6)
- fix: properly parse flag-type options [`c9384d0`](https://github.com/Madghostek/zprp-ffmpeg/commit/c9384d061f5c2f60c89798fb0700c81f61f38185)
- fix: move `generate_filters.py` out of package, so that it works both with mypy and normal run [`74cb5f6`](https://github.com/Madghostek/zprp-ffmpeg/commit/74cb5f6590214bfe7d13447ea9c489e16cfd8c55)

### [v1.0.0](https://github.com/Madghostek/zprp-ffmpeg/compare/v0.1.0...v1.0.0)

> 10 May 2024

- feature: extract filter type (video, audio) from source code. Make all filter options optional [`c77af88`](https://github.com/Madghostek/zprp-ffmpeg/commit/c77af8807ed7dc650d80781682ad98249bab3faa)
- refactor: code is more readable, split into files, changed prints to logger with debug level [`2e42ae8`](https://github.com/Madghostek/zprp-ffmpeg/commit/2e42ae8a3a6d5785adfe3aef596ada3d5e584074)
- fix: take care of typing in autogen code [`fef9dab`](https://github.com/Madghostek/zprp-ffmpeg/commit/fef9dabb56efacf058fbd08744bc412f765a95d9)

#### [v0.1.0](https://github.com/Madghostek/zprp-ffmpeg/compare/v0.0.0...v0.1.0)

> 29 April 2024

- Mypy [`#2`](https://github.com/Madghostek/zprp-ffmpeg/pull/2)
- Ffmpeg connector and initial stream class logic [`#1`](https://github.com/Madghostek/zprp-ffmpeg/pull/1)
- feature: add very basic graph structure and crucial api parts [`cb6c4fd`](https://github.com/Madghostek/zprp-ffmpeg/commit/cb6c4fd2473b66f968131dfd806e82902395f78b)
- feature: crucial base classes for the package [`e133438`](https://github.com/Madghostek/zprp-ffmpeg/commit/e133438f08fbf248f28e7d67b4c40640ed9f3717)
- fix: remove not needed class, fix mypy type errors [`91aa8cf`](https://github.com/Madghostek/zprp-ffmpeg/commit/91aa8cf23ad051d4126083c57f6749bd49d4d517)

#### v0.0.0

> 26 March 2024

- Change authors [`2892f0f`](https://github.com/Madghostek/zprp-ffmpeg/commit/2892f0fac9b13743e06969e8e8a46ee8792541dd)
- Restore design proposal [`09e47f5`](https://github.com/Madghostek/zprp-ffmpeg/commit/09e47f5279fc933980b10e220292e400f2635b4e)
- Try to revert merge. [`6c4fda6`](https://github.com/Madghostek/zprp-ffmpeg/commit/6c4fda6d834687cc2a3e4e9cca4df722df1356aa)
