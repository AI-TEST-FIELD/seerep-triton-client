FROM nvcr.io/nvidia/pytorch:22.12-py3

# Add your customizations here
COPY . /opt/client 
RUN pip install -r /opt/client/requirements.txt
WORKDIR /opt/client