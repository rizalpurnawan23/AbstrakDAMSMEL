<p align="center">
  <img src="img/AbstrakDAMSMEL_logo_v1.svg" alt="AbstrakDAMSMEL Logo">
</p>

# **AbstrakDAMSMEL**

Welcome to **AbstrakDAMSMEL**, and get ready for continuous optimisation.

## **Overview**

**AbstrakDAMSMEL** (Abstrak-Directional Adaptive Metric Sampling Minimal Expected Loss), is a Python package for mathematical optimisation, specifically, continuous optimisations.

**DAMSMEL** (Directional Adaptive Metric Sampling Minimal Expected Loss) is an optimisation method for continuous optimisation, currently for unconstraint problems.

**DAMSMEL** uses adaptive metric sampling with exponentially decaying distance of adjacent points and by exploiting the minimal expected loss in the samples.
In our paper, we have shown DAMSMEL's convergence for convex optimisation landscapes.
We have also performed empirical tests on DAMSMEL in several problems and have shown that DAMSMEL tends to provide better accuracies and reliability compared to gradient-based optimisations.
We have also demonstrated DAMSMEL capacity as a machine learning model for medium scale regression problems in [this notebook](damsmel_tests/damsmel_test_concrete.ipynb).
All of our current tests on DAMSMEL can be observerd [here](damsmel_tests).

## **Contributors**

**DAMSMEL** is developed by
[Rizal Purnawan](https://orcid.org/0000-0001-8858-4036)
and [Dieky Adzkiya](https://orcid.org/0000-0002-4718-2871)
(Department of Mathematics, Institut Teknologi Sepuluh Nopember) as in independent research project.

While **AbstrakDAMSMEL** is an implementation of DAMSMEL into a Python package, developed by
[Rizal Purnawan](https://orcid.org/0000-0001-8858-4036)
and [Dieky Adzkiya](https://orcid.org/0000-0002-4718-2871)
(Department of Mathematics, Institut Teknologi Sepuluh Nopember) under project Abstrak.


## **Importing AbstrakDAMSMEL**

Follow the command below to import the modules into your notebook or local machine:

1. First, install the package using the following command:
```
!pip install git+https://github.com/rizalpurnawan23/AbstrakDAMSMEL.git
```
2. Then import `damsmel` module using the following command:
```
from damsmel import damsmel
```
There are two main Python classes in `damsmel` module, namely `DAMSMELRegressor` for machine learning regression implementations and `DAMSMEL` for general continuous optimisations,
which can also be imported as follows:
```
from damsmel.damsmel import DAMSMELRegressor
```
```
from damsmel.damsmel import DAMSMEL
```

## **License**

**AbstrakDAMSMEL** is under MIT License. So, feel free to use it. We hope it could be helpful for everybody.
