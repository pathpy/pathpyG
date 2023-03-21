FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
RUN pip install zstandard
RUN pip install matplotlib
RUN pip install numpy
RUN pip install scipy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install jedi==0.17.2
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN /opt/conda/bin/conda install jupyter -y --quiet
RUN mkdir /opt/notebooks
EXPOSE 8888
CMD cd /notebooks && /opt/conda/bin/jupyter notebook --ip='*' --port=8888 --no-browser --allow-root