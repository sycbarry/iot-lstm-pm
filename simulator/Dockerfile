FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file first (for layer caching)
COPY environment.yml .

# Create Conda environment (named "" in this case)
RUN conda env create -f environment.yml

# Use the conda environment for the rest of the commands
SHELL ["conda", "run", "-n", "iot-lstm-pm", "/bin/bash", "-c"]

# Copy the rest of your app
COPY . .

# Set default command to run the simulator
CMD ["conda", "run", "-n", "iot-lstm-pm", "python", "simulator.py"]
