# Develop

Build out the notebook, export it to a .py file that we execute as part of a pre-processing step prior to starting our server. This always guarantees a fresh copy of a model that the server loads prior to serving it for predictions.

# API Contracts

#### POST /predict

Body: 
```
{ 'reading': [1, 2, 3, ...] }
```

Response: 
```
{status: ""}
```


#### GET|POST /health
pretty self-explanatory
