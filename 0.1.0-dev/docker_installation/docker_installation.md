# Docker Installation {#docker_installation}

:pytorch_logo: PyTorch provides a :docker_logo: [Docker image](https://hub.docker.com/r/pytorch/pytorch) with PyTorch preinstalled. Using this image, the Dockerfile below creates a Docker image with PathpyG installed.

=== "GPU"
    ```dockerfile
    FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    WORKDIR /workspaces/pathpyG
    RUN apt-get update
    RUN apt-get -y install git

    RUN pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

    RUN pip install torch_geometric>=2.4.0
    RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    RUN pip install git+https://github.com/pathpy/pathpyG.git
    ```
=== "CPU"
    ```dockerfile
    FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    WORKDIR /workspaces/pathpyG
    RUN apt-get update
    RUN apt-get -y install git

    RUN pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu # CPU only

    RUN pip install torch_geometric>=2.4.0
    RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html # CPU only
    RUN pip install git+https://github.com/pathpy/pathpyG.git
    ```
