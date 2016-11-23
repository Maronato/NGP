# NGP
Harvard's CS50 Final Project - Natural Grade Processing - Predicts students' grades like a pro(or does it?)

# Requirements
* Python 3.x
* Numpy
* Scipy
* Pandas

# Data
The data was collected from various universities in Brazil.
The subjects are, in order:

Calculus 1,Calculus 2,Calculus 3,Calculus 4,Physics 1,Physics 2,Physics 3,Physics 4,Programming 1,Programming 2,Analytic Geometry,Linear Algebra

(exact names may differ in certain universities)

To load the database:
```
from NGP.Data import Unicamp
data = Unicamp.load()
```
`data` has 3 attributes:

`data.dataset`: 2D array with the whole dataset

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

More details and the maths behind it all can be found within the code

### Usage

Can be used with preloaded data:
```
from Data import Unicamp
from Fun.ML import NMF
u = Unicamp.load()
model = NMF(u.W, u.H)
test = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
model.predict(test)
```
Can also be used with new data(might take a while to fit):
```
from Data import Unicamp
u = Unicamp.load()
model = NMF()
model.fit(u.dataset, 8)
test = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
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
## Neighborhood-Based Collaborative Filtering

### What is
Not so good, but very fast.
Can be used to predict single grades of a given student.

* Calculates the Pearson Correlation between a given user(the `active` user) and all the other users.
* Selects the `n` closest users(`neighbors`)
* Calculates the weighted average of the `neighbors`'s grades within a selected class using the correlation index calculated above.
* The result is the prediction.

As you can see, it is pretty simple and the results aren't very good, but is really fast and may be useful if used in a hybrid system.

### Usage

Usage:
```
from Data import Unicamp
from Fun.ML import NB_CF
u = Unicamp.load()
model = NB_CF()
active = [10, 5, 0, 0, 9, 8, 0, 0, 0, 0, 8, 0]
model.fit(active, u.dataset)
model.predict(0)
```

## More to come
