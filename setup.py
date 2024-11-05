from setuptools import setup, find_packages

setup(
    name='DAMSMEL',
    version='0.0.1',
    packages=find_packages(),
    description="A continuous optimisation algorithm",
    author="Rizal Purnawan",
    install_requires=[
        'numpy>=1.26.0',
        'pandas>=2.1.1'
        ],
    url='https://github.com/rizalpurnawan23/AbstrakDAMSMEL'
)