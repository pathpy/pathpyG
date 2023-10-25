FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /workspaces/pathpyG
RUN apt-get update
RUN apt-get -y install git

# Install torch
RUN pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
# pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu # CPU only

# Install torch_geometric and dependencies
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
# RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html # CPU only
