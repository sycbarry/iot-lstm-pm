import { Fragment, useEffect, useState } from 'react';
import './App.css';
import { LineChart, Line, Legend, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

function App() {
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    let cancel = false;

    const poll = async () => {
      if (paused) return;

      try {
        const res = await fetch("http://127.0.0.1:5002/consume");
        const json = await res.json();
        if (!cancel) setCurrentPrediction(json);
      } catch (e) {
        console.log(e);
      }

      if (!cancel) setTimeout(poll, 5000);
    };

    poll();
    return () => {
      cancel = true;
    };
  }, [paused]); // Only rerun when paused toggles

  return (
    <div className="App">
      <div>
        <div className='section-meta'>
          <h2>Overview</h2>
          <p>The original values represent raw sensor data.</p>
          <p>The predicted values are what the model expects based on that input.</p>
          <p>The model is treated as the reference — sensor data may include anomalies.</p>
          <p>If anomalies are rare, it means the model sees the input as close to an ideal time-series.</p>
          <button onClick={() => setPaused(!paused)}>
            {paused ? "Resume Reading" : "Pause Reading from API"}
          </button>
        </div>
        {currentPrediction ? <PredictionResult prediction={currentPrediction} /> : "loading..."}
      </div>
    </div>
  );
}

const TimeSeriesChart = ({ predicted, original }) => {
  const chartData = predicted.map((val, idx) => ({
    time: idx,
    predictedsignal: val,
    sensordata: original[idx],
  }));

  return (
    <ResponsiveContainer width="100%" height={700}>
      <LineChart data={chartData}>
        <CartesianGrid stroke="#ccc" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />
        <Legend />
        {/* <Line type="monotone" dataKey="predictedsignal" stroke="#8884d8" />
        <Line type="monotone" dataKey="sensordata" stroke="#82ca9d" /> */}
        <Line type="monotone" dataKey="predictedsignal" stroke="green" dot={false} />
        <Line type="monotone" dataKey="sensordata" stroke="blue" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
};

function PredictionResult({ prediction }) {
  return (
    <Fragment>
      <div className='section-meta'>
        <h3>Anomalies</h3>
        <div>{prediction["message"]["anomalies"]}</div>
        <br />
        <h3>Anomaly %</h3>
        <div>{prediction["message"]["anomaly_percentage"]}</div>
      </div>
      <div className='section-graph'>
        <TimeSeriesChart
          predicted={prediction["message"]["predicted_waveform"]}
          original={prediction["message"]["sensor_waveform"]}
        />
      </div>
    </Fragment>
  );
}

export default App;
