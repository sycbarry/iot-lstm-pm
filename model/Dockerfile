FROM continuumio/miniconda3

WORKDIR /app

# Be explicit about what’s happening
COPY environment.yml ./environment.yml

# Print Conda version + verbose install
RUN conda --version && \
    echo "Creating Conda environment..." && \
    conda env create -f environment.yml && \
    echo "Conda environment created."

# Use bash shell for CMDs in the environment
SHELL ["conda", "run", "-n", "iot-lstm-pm", "/bin/bash", "-c"]

# Copy everything else into the container
COPY . .

RUN apt-get update && apt-get install -y curl

# Disable stdout buffering for real-time logs
ENV PYTHONUNBUFFERED=1

# Optionally confirm env is activated
RUN echo "Python path: $(which python)" && \
    python --version && \
    pip list

# Run preprocessing + server with clear logs
CMD echo "Starting preprocessing..." && \
    python -u notebook.py && \
    echo "Starting Flask server..." && \
    python -u server.py
