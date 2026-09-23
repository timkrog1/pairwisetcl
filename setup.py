from setuptools import setup

setup(
    name='pairwisetcl',
    version='1.0.0',
    description='A Python package for calculating electron spin dephasing due to pairwise nuclear spin flip-flops using the Time-convolutionless master equation',
    url='https://github.com/timkrog1/pytcl',
    author='Timothy J. Krogmeier',
    author_email='krogm033@umn.edu',
    packages=['pairwisetcl'],
    install_requires=[
        'numpy>=1.24.0,<2.0.0',
        'pyscf>=2.0.0'
        ],
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: MIT License',
    ],
)
