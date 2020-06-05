#!/usr/bin/env python3

from setuptools import setup

setup(name='Willump-Simple',
      version='0.2',
      description='Willump Is a Low-Latency Useful ML Platform',
      author='Peter Kraft',
      author_email='kraftp@cs.stanford.edu',
      url='https://github.com/stanford-futuredata/Willump-Simple',
      packages=['willump', 'willump.evaluation', 'willump.graph'],
      test_suite='tests',
     )