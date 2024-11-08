<p align="center">
  <img src="img/AbstrakDAMSMEL_logo_v1.svg" alt="AbstrakDAMSMEL Logo">
</p>

[![DOI](https://zenodo.org/badge/883910405.svg)](https://doi.org/10.5281/zenodo.14042916)

# **AbstrakDAMSMEL**

Welcome to **AbstrakDAMSMEL**, and get ready for continuous optimisation.

## **Overview**

**AbstrakDAMSMEL** (Abstrak-Directional Adaptive Metric Sampling Minimal Expected Loss),
is a Python package for mathematical optimisation, specifically, continuous optimisations.

**DAMSMEL** (Directional Adaptive Metric Sampling Minimal Expected Loss) is an optimisation method for continuous optimisation, currently for unconstrained problems.

**DAMSMEL** uses adaptive metric sampling with exponentially decaying distance of adjacent points and by exploiting the minimal expected loss in the samples.
In our paper [(Purnawan and Adzkiya, 2024)](https://doi.org/10.21203/rs.3.rs-5402563/v1), we have shown DAMSMEL's convergence for convex optimisation landscapes.
We have also performed empirical tests on DAMSMEL in several problems and have shown that DAMSMEL tends to provide better
accuracies and reliability compared to gradient-based optimisations. We have also demonstrated DAMSMEL capacity as a machine learning model
for medium scale regression problems in [this notebook](damsmel_tests/damsmel_test_concrete.ipynb).
All of our current tests on DAMSMEL can be observerd [here](damsmel_tests).

**Reference**\
Purnawan, R.; Adzkiya, D. (2024). *Directional Adaptive Metric Sampling Minimal Expected Loss: A Continuous Optimisation Method*.
PREPRINT (Version 1). Research Square. DOI: [10.21203/rs.3.rs-5402563/v1](https://doi.org/10.21203/rs.3.rs-5402563/v1)

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
There are two main Python classes in `damsmel` module, namely `DAMSMELRegressor` for machine learning regression
implementations and `DAMSMEL` for general continuous optimisations,
which can also be imported as follows:
```
from damsmel.damsmel import DAMSMELRegressor
```
```
from damsmel.damsmel import DAMSMEL
```
For ease of use, watch [our tutorials](https://youtu.be/ZjmQ48pCego?si=W34WWYDduDqu11YB) on [our Youtube channel](https://www.youtube.com/@abstrak-math).

## **Contributors**

**DAMSMEL** is developed by [Rizal Purnawan](https://github.com/rizalpurnawan23)<sup>1</sup> and [Dieky Adzkiya](https://github.com/diekyadzkiya)<sup>2</sup>
as in independent research project.

While **AbstrakDAMSMEL** is an implementation of [DAMSMEL](https://doi.org/10.21203/rs.3.rs-5402563/v1) into a Python package,
developed by [Rizal Purnawan](https://github.com/rizalpurnawan23)<sup>1</sup> and [Dieky Adzkiya](https://github.com/diekyadzkiya)<sup>2</sup>.

<sup>1</sup>ORCID: [0000-0001-8858-4036](https://orcid.org/0000-0001-8858-4036)\
<sup>2</sup>ORCID: [0000-0002-4718-2871](https://orcid.org/0000-0002-4718-2871)\
<sup>2</sup>Department of Mathematics, Institut Teknologi Sepuluh Nopember

## **License**

**AbstrakDAMSMEL** is under [MIT License](LICENSE). So, feel free to use it. We hope it could be helpful for everybody.

## **Get in Touch with Us**

[<img align="left" alt="Linkedin" height="22px" src="img/linkedin-original.svg" style="padding-right:10px;"/>](https://www.linkedin.com/company/abstrak-math/)
[<img align="left" alt="Linkedin" height="22px" src="img/YouTube.svg" style="padding-right:10px;"/>]([https://www.linkedin.com/company/abstrak-math/](https://www.youtube.com/@abstrak-math))
