"""Module for pathpy processes."""

# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : __init__.py -- initialisation of processes
# Author    : Ingo Scholtes <scholtes@uni-wuppertal.de>
# Time-stamp: <Mon 2020-04-20 10:28 ingo>
#
# Copyright (c) 2016-2020 Pathpy Developers
# =============================================================================
# flake8: noqa
# pylint: disable=unused-import

from pathpyG.processes.random_walk import RandomWalk

# from pathpyG.processes.epidemic_spreading import EpidemicSIR
from pathpyG.processes.sampling import VoseAliasSampling
from pathpyG.processes.random_walk import HigherOrderRandomWalk
