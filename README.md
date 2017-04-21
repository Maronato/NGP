# NGP
Harvard's CS50 Final Project - Natural Grade Processing

Using machine learning to predict students' grades!

NGP uses various machine learning algorithms and techniques to predict students' grades based on their peers' and their own. This readme is a little shallow, so do check out the [website/blog](https://gradeprocessing.herokuapp.com) dedicated to this project if you want to see live examples, tutorials and data.

# The website

The project's main selling point is the website that offers in-depth information about the algorithms used and easy-to-use simulations and explanations.

## About the website

It is made with Django and Python and is hosted on Heroku.
It has essentially 4 pages:

* Home
* W and H
* Design and Math
* Statistics

The home page explains the motivations behind the project and contain a simulation of the algorithm that can generate real-time predictions.

W and H contain another example table but focus on explaining what is the meaning behind the `H` and `W` tables used by the algorithm

Design and Math is a in-depth explanation of the design choices and the math that powers NMFs. It is also filled with links to the literature used by me when coding the algorithm.

Statistcs is an overview of the dataset collected by me and an analysis of the results collected with the algorithm.

## Usage
Is pretty simple. Just a regular website.

## Deploying

Want to deploy a version of yours?

To make it work porperly I use an Anaconda buildpack.
The one used is this: https://github.com/icoxfog417/conda-buildpack.git

To deploy, simply click the "Deploy to Heroku" button

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

## Running locally

### Requirements
The main requirement is Python 3
The rest can be found within requirements.txt

### Running
To run the website locally using Django's server, do the following from the project's root:

I also recommend downloading [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) but you can skip the first 2 lines if you don't want to use it.
```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 manage.py runserver
```

and then open the website on the created port.

# The Command Line Implementation

## Requirements
* Python 3.x
* Numpy
* Scipy
* Pandas
* Cython

## Data
The data was collected from various universities in Brazil.
The subjects are, in order:

Calculus 1,Calculus 2,Calculus 3,Calculus 4,Physics 1,Physics 2,Physics 3,Physics 4,Programming 1,Programming 2,Analytic Geometry,Linear Algebra

(exact names may differ in other universities)

To load the database:
```
from data import unicamp
data = unicamp.load()
```
`data` has 3 attributes:

`data.data`: 2D array with the whole dataset

`data.H`: the preloaded H-matrix of the data (see NMF)

`data.W`: the preloaded W-matrix of the data (see NMF)

# Prediction algorithms

## Non-Negative Matrix Factorization
### What is
Is very good, but takes some time to fit. Best used with preloaded `H` and `W` matrices

The basic idea is that there is a lot of implicit relations and information in a set of data. By understanding and extracting these implicit relations we can extrapolate them and generate new data(predictions).

For this, we assume that in a given dataset there are `R` hidden 'features', so that, given a matrix `X` with `n` rows and `m` columns, one can factorize `X` into `X ≃ WH` where `W` and `H` are other matrices such that `W` is `n x R` and `H` is `R x m`.

Then, in our case, `W`'s values represent the weights between the students and the `R` features, and `H`'s values represent the relationship between the features and the classes that the students took.

By generating `W` and `H` using only the collected grades, not using the ones missing, we can find `V = WH`, where `V`' values for collected values are pretty close to `X`'s, but `V` now also contains values where `X` had nothing. These new values are the predictions generated from the 'hidden' features that we found on `H` and `W`.

More details and the maths behind it all can be found in the project's website

### Usage
The algorithm is written in python but actually compiled to C using Cython so that they run faster.

If you are using Windows you may need to compile a version of the algorithm so that it runs on your OS. The easiest way to do that is the following:

I recommend using [virtualenv](https://virtualenv.pypa.io/en/stable/installation/). If you don't want to use it, skip the first 2 lines.

Then follow these commands from the root of the project:
```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
cd learning
python3 setup.py build_ext --inplace
```

You can ignore the warnings.

A new `.so` file should be located at
`/learning/NGP/learning`
Just copy it and paste at
`/learning`

Done!
Now you can load the classes and predict away.

#### Examples

To use it with the preloaded data:
```
from data import unicamp
from learning.nmf import NMF
u = unicamp.load()
model = NMF(u.W, u.H)
test = [10., 5., 0., 0., 9., 8., 0., 0., 0., 0., 8., 0.]
model.predict(test)
```
Can also be used with new data(might take a while to fit):
```
from data import unicamp
from learning.nmf import NMF
u = unicamp.load()
model = NMF()
model.fit(u.data, 8)
test = [10., 5., 0., 0., 9., 8., 0., 0., 0., 0., 8., 0.]
model.predict(test)
```
the generated `W` and `H` can be accessed with:
```
model.W
model.H
```
To extract the whole predicted matrix `V`:
```
model.get_V()
```
