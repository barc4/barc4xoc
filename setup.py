from setuptools import find_packages, setup

setup(
    name='barc4xoc',
    version='2024.07.04',
    author='Rafael Celestre',
    author_email='rafael.celestre@synchrotron-soleil.fr',
    description='A Python package for X-ray Optical Calculations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/barc4/barc4xoc',
    license='GPL-3.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
