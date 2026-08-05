FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt pyproject.toml LICENSE ./
COPY code ./code
RUN pip install --no-cache-dir .
ENTRYPOINT ["glioma-atlas"]
