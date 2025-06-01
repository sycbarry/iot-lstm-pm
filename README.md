# iot-lstm-pm

# TODO 
- [ ] finish docker composing stuff 

A demo that shows how an LSTM model can be used to monitor sensor data over a period of time.

![system overview](./assets/image_1.png)

### Self-hosted instructions

Make sure you have docker installed with docker compose.

1. Clone this repo
2. Run: `./build-docker.sh` from the root directory. (IF you're on a Windows Machine, just run the docker command explicitly with: `docker compose up --build`)
3. To exit kill the program, just close the terminal or hit Control+C

#### Alternative

You can just use mprocs on your local machine to start up the services. Just run `mprocs` from the root of this directory (make adjustments to the `mprocs.yaml` file if needed)

> This was not generated with AI. Cheers.
