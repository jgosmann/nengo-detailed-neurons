#!/usr/bin/env python

try:
    from setuptools import setup
    from setuptools.command.test import test as TestCommand
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception as e:
        print("Forget setuptools, trying distutils...")
        from distutils.core import setup

description = ("Enable Nengo to use neuron models simulated in NEURON.")

setup(
    name="nengo_detailed_neurons",
    version="0.1",
    author="Jan Gosmann",
    author_email="jgosmann@uwaterloo.ca",
    packages=['nengo_detailed_neurons'],
    scripts=[],
    description=description,
    requires=[
        "nengo",
    ]
)
