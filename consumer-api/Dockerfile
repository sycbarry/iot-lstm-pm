FROM continuumio/miniconda3

# Set work directory
WORKDIR /app

# Copy environment file and install dependencies
COPY environment.yml .
RUN conda env create -f environment.yml

# Make sure conda env is activated when container runs
SHELL ["conda", "run", "-n", "iot-lstm-pm", "/bin/bash", "-c"]

# Copy rest of app
COPY . .

# Expose the Flask port
EXPOSE 5002

# Run app in the conda environment
CMD ["conda", "run", "-n", "iot-lstm-pm", "python", "server.py"]
