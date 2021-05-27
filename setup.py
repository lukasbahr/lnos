from setuptools import setup

setup(
    name='lnos',
    url='https://github.com/lukasbahr/lnos',
    author='Lukas Bahr',
    packages=['lnos.net', 'lnos.experiments', 'lnos.datasets', 'lnos.observer'],
    install_requires=['numpy', 'torch', 'scipy', 'matplotlib', 'torchdiffeq'],
    version='0.01',
    license='MIT',
    description='Implemantation of lnos.',
)
