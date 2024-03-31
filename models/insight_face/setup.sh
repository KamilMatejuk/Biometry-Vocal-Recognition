#!/bin/bash

src=$(dirname "$(realpath "$0")")/src

# sed -i 's/from \([a-zA-Z0-9_]*\) import/from .\1 import/g' $src/models/__init__.py
