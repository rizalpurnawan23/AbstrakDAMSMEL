<p align="center">
  <img src="img/AbstrakDAMSMEL_logo.svg" alt="AbstrakDAMSMEL Logo">
</p>

# **AbstrakDAMSMEL**

Welcome to **AbstrakDAMSMEL**, and get ready for continuous optimisation.

## **Overview**

**AbstrakDAMSMEL** (Abstrak-Directional Adaptive Metric Sampling Minimal Expected Loss), is a Python package for mathematical optimisation developed by Rizal Purnawan
in collaboration with Dr. Dieky Adzkiya (Department of Mathematics, Institut Teknologi Sepuluh Nopember) under an indpendent project Abstrak.

**DAMSMEL** (Directional Adaptive Metric Sampling Minimal Expected Loss) is an optimisation method for continuous optimisation, currently for unconstraint problems,
developped by Rizal Purnawan and Dr. Dieky Adzkiya (Department of Mathematics, Institut Teknologi Sepuluh Nopember) on an independent research project.

**DAMSMEL** uses adaptive metric sampling with exponentially decaying distance of adjacent point samples and by exploiting the minimal expected loss in the samples.
In our paper, we have shown DAMSMEL's convergence for convex optimisation landscapes.
We have also performed empirical tests on DAMSMEL in several problems and have shown that DAMSMEL tends to provide better accuracies and reliability compared to gradient-based optimisations.
We have also demonstrated DAMSMEL capacity as a machine learning model for medium scale regression problems in [this notebook](damsmel_tests/damsmel_test_concrete.ipynb).
All of our current tests on DAMSMEL can be observerd [here](damsmel_tests).

## **Contributors**

**DAMSMEL** is developed by Rizal Purnawan and Dr. Dieky Adzkiya (Department of Mathematics, Institut Teknologi Sepuluh Nopember).
While the code implementation (including AbstrakDAMSMEL) and testing are conducted by Rizal Purnawan.

## **Importing AbstrakTS**

Follow the command below to import the modules into your notebook or local machine:

1. First, install the package using the following command:
```
!pip install git+https://github.com/rizalpurnawan23/AbstrakDAMSMEL.git
```
2. Then import `damsmel` module using the following command:
```
from damsmel import damsmel
```
There are two main Python classes in `damsmel` module, namely `DAMSMELRegressor` for machine learning regression implementations and `DAMSMEL` for general continuous optimisations.

## **License**

**AbstrakDAMSMEL** is under MIT License. So, feel free to use it. We hope it could be helpful for everybody.
