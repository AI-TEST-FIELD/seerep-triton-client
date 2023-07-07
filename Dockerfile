FROM python:3.8 

RUN apt-get update && \
    apt-get install -y software-properties-common

RUN apt-get update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
    libffi-dev uuid-dev tk-dev wget

RUN python3 -m pip install --upgrade pip setuptools

COPY . /home/airflow/ai_test_field/
WORKDIR /home/airflow/ai_test_field/
RUN python3 -m pip install opencv-python scipy scikit-image numpy matplotlib pandas bagpy geopy

RUN python3 -m pip install -r requirements.txt
