# README

## Introduction

random_variate is a Python library that generates nonuniform (pseudo-) random variates, from several distributions, specifically the following:
* Bernoulli
* Geometric
* Erlang
* Exponential
* Normal
* Triangular
* Uniform
* Weibull


Generally, random_variate implements generators for most distributions using the inverse transform method. However, the Normal distribution is implemented using the polar method.

## Setup

A `requirements.txt` file is provided. 
The required packages can be installed via `pip install -r requirements.txt`.

A [virtual environment](https://docs.python.org/3/library/venv.html) is recommended for installation but not strictly required. 

## Use

random_variate should be imported like a normal library, i.e. `import random_variate`

Each generator is implemented as a class. Each class requires one or more parameters upon instantiation; the parameters describe the distribution being modeled. For instance, the `Bernoulli` class requires one parameter, p, a float between 0 and 1. 

E.g. to instantiate a new `Bernoulli` class: `MyBernoulli = random_variate.Bernoulli(0.4)`

Each class has three methods:
* `generate_random()`
* `expected_value()`
* `variance()`

All of the methods do not take any additional parameters.

`generate_random()` generates a random variate from the distribution being modeled.
`expected_value()` returns the expected value of the distribution.
`variance()` returns the variance of the distribution.

## Advanced Use

To generate multiple random variates from a single distribution, a list comprehension is recommended, e.g.
`[MyBernoulli.generate_random() for _ in range(1000)]`

To set a seed in order to ensure reproducibility, please call `random.seed()`, as all the random variate generation algorithms use the `random.random()` method to generate one or more random uniform variates. Don't forget to `import random` if this is the case.

Docstrings are available for each class to provide more detail on the modeled distributions.