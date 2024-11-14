"""Config reader."""

import os
import sys
import tomllib
from pathlib import Path


# path to the module
path = Path(sys.modules[__name__].__file__).resolve().parents[1]

# default config file name
configfile_name = "pathpyG.toml"

# load default config
configfile_path = os.path.join(path, configfile_name)

# load config file
with open(configfile_path, "rb") as f:
    config = tomllib.load(f)

# check if local config file is defined
if os.path.exists(os.path.join(os.getcwd(), configfile_name)):
    # get location of local config file
    configfile_path = os.path.join(os.getcwd(), configfile_name)

    # load local config file
    with open(configfile_path, "rb") as f:
        _config = tomllib.load(f)

    # update default config file
    config.update(_config)
