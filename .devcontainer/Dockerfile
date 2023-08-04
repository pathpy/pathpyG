FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
RUN apt-get update
RUN apt-get -y install git
RUN pip install pathpy2
RUN pip install zstandard
RUN pip install matplotlib
RUN pip install numpy
RUN pip install scipy
RUN pip install singledispatchmethod
RUN pip install seaborn
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install jedi
RUN pip install torch_geometric
RUN pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
RUN /opt/conda/bin/conda install jupyter -y --quiet