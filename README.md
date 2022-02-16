# Kohonen Network Challenge

### Walkthrough
See the *kohonen_demo.ipynb* notebook for the walkthrough of how I developed a Self-Organized Map (SOM) model in Python/NumPy.  

### Productionise
The steps below outline how to run the model from a Jupyter notebook in a docker container on your own device. 

**Steps:**
Git clone this repo and navigate to the *deployment* folder. 

Build the docker image:
```docker build -t kohonen:v0.0.1 .```

Run the containerised image:
```docker run --rm -it -p 8888:8888 kohonen:v0.0.1```

Navigate to http://localhost:8888/ in your browser, and open the *main.ipynb* notebook to run the model with your chosen parameters. 

