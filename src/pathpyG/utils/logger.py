#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : logger.py -- Module to log comments
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Thu 2024-11-14 12:09 juergen>
#
# Copyright (c) 2016-2019 Pathpy Developers
# =============================================================================
import os
import sys
import logging.config
from pathlib import Path


# path to the module
path = Path(sys.modules[__name__].__file__).resolve().parents[1]

# default logging config file name
loggingfile_name = "logging.toml"

# check if local logging config file is defined
if os.path.exists(os.path.join(os.getcwd(), loggingfile_name)):
    # get location of local config file
    loggingfile_path = os.path.join(os.getcwd(), loggingfile_name)
else:
    # get location of default config file
    loggingfile_path = os.path.join(path, loggingfile_name)

# update logging confing
logging.config.fileConfig(loggingfile_path)

# create logger
logger = logging.getLogger("pathpyg")

# Status message of the logger
logger.debug("Logger successful initialized.")
