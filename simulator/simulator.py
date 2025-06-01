from random import randint, choice
import json 
from time import sleep
import uuid
import numpy as np
import requests
import os
import time


DEBUG = False
EXPORT_CSV = False

endpoint = os.getenv("ENDPOINT", "https://webhook.site/b425fbac-0ca2-4ebf-a205-209f40193be9")
health_endpoint = os.getenv("HEALTHENDPOINT", "http://localhost:5001/health")

class Sensor:
    def __init__(self, *args, noise_level=1):
        self.id = uuid.uuid4()
        self.period = 1000
        self.amplitude = 5
        self.noise_level = noise_level
        self.time = np.arange(4 * 1000 + 1)

    def get_id(self):
        return self.id

    def seasonal_pattern(self, season_time):
        base = np.sin(2 * np.pi * season_time) * 0.5
        spikes = np.exp(-40 * (season_time - 0.1) ** 2)
        burst = 0.2 * np.sin(8 * np.pi * season_time)
        return base + spikes + burst

    def seasonality(self, time, period, amplitude=1, phase=0):
        season_time = ((time + phase) % period) / period
        return amplitude * self.seasonal_pattern(season_time)

    def noise(self, time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        noise = rnd.randn(len(time)) * (noise_level / 2)
        return noise

    def generate_signal(self, anomaly=False):
        base_series = self.seasonality(
            self.time, period=self.period, amplitude=self.amplitude
        )
    
        if not anomaly:
            noise_signal = self.noise(self.time, noise_level=self.noise_level, seed=None)
            return base_series + noise_signal
    
        else:
            # Start with base signal
            series = base_series.copy()
        
            # Inject different kinds of anomalies
            num_anomalies = randint(2, 6)
            for _ in range(num_anomalies):
                anomaly_type = choice(['spike', 'drop', 'drift', 'shift', 'noise_burst'])
                idx = randint(100, len(series) - 100)

                if anomaly_type == 'spike':
                    series[idx:idx+10] += np.random.uniform(5, 15)

                elif anomaly_type == 'drop':
                    series[idx:idx+10] -= np.random.uniform(5, 15)

                elif anomaly_type == 'drift':
                    drift = np.linspace(0, np.random.uniform(3, 10), 50)
                    series[idx:idx+50] += drift

                elif anomaly_type == 'shift':
                    shift = np.random.uniform(-5, 5)
                    series[idx:] += shift

                elif anomaly_type == 'noise_burst':
                    burst = np.random.normal(0, 5, 30)
                    series[idx:idx+30] += burst

            # Add general noise
            noise_signal = self.noise(self.time, noise_level=self.noise_level, seed=None)
            return series + noise_signal

    def generate_signal_2(self, anomaly=False):
        if not anomaly:
            series = self.seasonality(
                self.time, period=self.period, amplitude=self.amplitude
            )
            # noise_signal = self.noise(self.time, noise_level=1, seed=42)
            return series
        elif anomaly == True:
            anomaly_level = randint(self.amplitude - 3, self.amplitude + 5)
            anomaly_period = randint(self.period - 500, self.period + 500)
            print(f"anomaly set / value => {anomaly_level}")
            series = self.seasonality(
                self.time,
                period=anomaly_period,
                amplitude=anomaly_level,
            )
            noise_signal = self.noise(self.time, noise_level=0.5, seed=42)
            return series + noise_signal


def post_request(data: np.ndarray) -> None: 
    if endpoint is None: 
        print("no endpoint has been specified")
        return None
    payload = {"reading": list(data)}
    print(f"sending payload to prediction api: {endpoint}")
    response = requests.post(endpoint, data=json.dumps(payload))
    return response.status_code


def wait_for_model(url, retries=30, delay=10):
    for i in range(retries):
        try:
            res = requests.get(url)
            if res.status_code == 200:
                return
        except Exception as e:
            print(f"Waiting for model to be ready... ({i+1}/{retries})")
        time.sleep(delay)
    raise RuntimeError("Model did not become ready in time.")


def main():
    sensor = Sensor(noise_level=2)
    counter = 0
    readings = []
    while True:
        """Generate sensor readings"""

        create_anomaly = randint(0, 1)
        if DEBUG == True and EXPORT_CSV == True:
            create_anomaly = 0
        # generate a normal value
        if create_anomaly == 0:
            sensor_reading = sensor.generate_signal(anomaly=False)  # if we have some sort of anomaly, set this to True
        # otherwise create a sensor reading with a form of anomaly
        else:
            sensor_reading = sensor.generate_signal(anomaly=True)  # if we have some sort of anomaly, set this to True

        print(f"emitting sensor value => {sensor_reading}")

        if not DEBUG: 
            wait_for_model(health_endpoint)
            post_request(sensor_reading)

        """Dump the readings to a file if debug mode is on"""
        if DEBUG == True and EXPORT_CSV == True:
            readings.append(
                {
                    "reading_type": "normal" if create_anomaly == 0 else "anomaly",
                    "reading": [float(x) for x in sensor_reading],
                }
            )
            if counter == 1:
                with open("readings.csv", "w", newline="") as csvfile:
                    import csv

                    fieldnames = ["reading_type", "reading"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(readings)
                break
            counter += 1

        sleep(5)


if __name__ == "__main__":
    main()
