#FROM git.ni.dfki.de:5050/ml_infrastructure/airflow-pbr-dags:ml-pipeline_dev_yolov50.1
FROM apache/airflow:2.2.3

USER root

RUN apt-get update && \
    apt-get install -y software-properties-common

RUN apt-get update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
    libffi-dev uuid-dev tk-dev wget && \
    apt-get clean

RUN wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz && \
    tar xvf Python-3.8.10.tgz && \
    cd Python-3.8.10 && \
    ./configure --enable-optimizations && \
    make install && \
    cd .. && \
    rm -rf Python-3.8.10*

RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN python3 -m pip install --upgrade pip setuptools

RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


COPY . /home/airflow/ai_test_field/
WORKDIR /home/airflow/ai_test_field/
RUN python3 -m pip install -r requirements.txt
RUN echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" >> ~/.bashrc

USER airflow
