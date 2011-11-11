#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: SETUP.PY
Date: Thursday, November 10 2011
Description: Setup and install gridworld package.
"""

from distutils.core import setup

setup(name='gridworld',
      version='0.01',
      description='Gridworld simulations using Python',
      author="Jeremy Stober",
      author_email="stober@gmail.com",
      package_dir={"gridworld" : "src"},
      packages=["gridworld"]
      )
      

