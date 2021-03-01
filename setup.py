import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# This requires PyTorch, which cannot be automatically  installed
# using pip.

setup(
    name = "Lumos",
    version = "0.1.0",
    author = "Laura Cabayol",
    author_email = "lcabayol@ifae.es",
    description = ("Photometry on PAUCam images with neural networks."),
    keywords = "astronomy",
    url = "https://github.com/lauracabayol/Lumos",
    license="GPLv3",
    packages=['Lumos'],
    install_requires=['numpy', 'pandas'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
