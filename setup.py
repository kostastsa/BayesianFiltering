from setuptools import setup, find_packages

setup(
    name='gaussfiltax',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'JAX',
        'numpy'
    ],
    author='Kostas Tsampourakis',
    author_email='kostas.tsampourakis@gmail.com',
    description='A simple Gaussian filter implementation',
)
