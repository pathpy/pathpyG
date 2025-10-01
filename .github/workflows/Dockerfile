FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /workspaces/pathpyG
RUN apt-get update
RUN apt-get -y install git
RUN apt install gnupg2 -y

# Install dependencies for manim
RUN apt-get install -y build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg

# Install torch
RUN pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install torch_geometric and dependencies
RUN pip install torch_geometric>=2.5.0
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# install pathpy
RUN pip install -e '.[dev,test,doc,vis]' && git config --global --add safe.directory /workspaces/pathpyG