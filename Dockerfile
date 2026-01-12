FROM nvcr.io/nvidia/pytorch:25.11-py3

COPY requirements_server.txt /workspace/requirements_server.txt
RUN pip install -r /workspace/requirements_server.txt

COPY vla-scripts/deploy.py /workspace/deploy.py

ENTRYPOINT ["python", "/workspace/deploy.py"]