from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
  name = 'pykaHFM',
  packages = ['pykaHFM'],
  version = '0.2.1',
  description = 'An implementation of kaHFM algorithm',
  author = 'Jiří Filip',
  author_email = 'j.f.ilip@seznam.cz',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url = 'https://github.com/jirifilip/pykaHFM',
  keywords = 'pykaHFM',
  classifiers = [],
  install_requires=['pandas', 'numpy']
)