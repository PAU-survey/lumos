import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# This requires PyTorch, which cannot be automatically  installed
# using pip.

setup(
    name = "lumos",
    version = "1.0.0",
    author = "Laura Cabayol",
    author_email = "lcabayol@ifae.es",
    description = ("Photometry on PAUCam images with neural networks."),
    keywords = "astronomy",
    url = "https://github.com/lauracabayol/Lumos",
    license="GPLv3",
    packages=['lumos'],
    install_requires=['numpy', 'pandas', 'torch', 'astropy', 'scipy', 'scikit-image'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
