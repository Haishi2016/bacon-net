# Developer Guide

## Rebuild Docs

The `docs/source` folder contains the source code of the documentation site. To rebuild:
```bash
# under docs/source folder
sphinx-build -b html . ../build/html
# to build without cache, add "-E" switch
sphinx-build -E -b html . ../build/html
```
