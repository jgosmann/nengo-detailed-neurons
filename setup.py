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

description = ("Add capabilities to Nengo for running Neuron neuron models.")

setup(
    name="nrnengo",
    version="0.1",
    author="Jan Gosmann",
    author_email="jgosmann@uwaterloo.ca",
    packages=['nrnengo'],
    scripts=[],
    description=description,
    requires=[
        "nengo",
    ]
)
