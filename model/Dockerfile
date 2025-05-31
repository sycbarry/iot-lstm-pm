FROM continuumio/miniconda3

WORKDIR /app

# Copy the environment.yml file
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Set shell to use the environment by default
SHELL ["conda", "run", "-n", "iot-lstm-pm", "/bin/bash", "-c"]

# Copy the rest of the application code
COPY . .

ENV PYTHONUNBUFFERED=1

# Run notebook.py, then start Flask server
CMD ["conda", "run", "-n", "iot-lstm-pm", "bash", "-c", "python notebook.py && python server.py"]