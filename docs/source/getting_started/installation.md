# Installation

[![PyPI](https://img.shields.io/pypi/v/HipoMap?style=flat&colorB=0679BA)](https://pypi.org/project/HipoMap/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/HipoMap?label=pypi%20downloads)](https://pypi.org/project/HipoMap/)

HipoMap can be installed using ``pip``.

## Before Installation

### OpenSlide
Before installing HipoMap, Should be installed OpenSlide first.

OpenSlide is a C library that provides a simple interface to read whole-slide images (also known as virtual slides). The
current version is 3.4.1, released 2015-04-20.

#### Linux (Fedora)
For Linux (Fedora), you can install latest version of OpenSlide by running following commands from terminal:

```
$ dnf install openslide
```

#### Linux (Debian, Ubuntu)
For Linux (Debian, Ubuntu), you can install latest version of OpenSlide by running following commands from terminal:

```
$ apt-get install openslide-tools
```

#### Linux (RHEL, CentOS)
For Linux (RHEL, CentOS), you can install latest version of OpenSlide by running following commands from terminal:

```
$ yum install epel-release
$ yum install openslide
```

#### Mac OSX
For MacOSX, you can install latest version of OpenSlide by running following commands from terminal:

```
$ brew install openslide
```

#### Windows
For Window, you can install latest version of OpenSlide:

```
https://openslide.org/download/#windows-binaries
```

## pip

If you use ``pip``, you can install it with:
```
pip install HipoMap
```
