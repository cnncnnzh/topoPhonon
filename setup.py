import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="topoPhonon",
    version="1.0.3",
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.5.2",
        "numpy>=1.23.1",
        "tqdm>=4.64.1",
        "scipy>=1.6.2"
    ],
    author="He (Alan) Zhu",
    author_email="zhu00336@umn.edu",
    description=" topoPhonon package is a python package that allows users to" 
    "phononic calculate topological properties, by building phonon tight-binding"
    "model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cnncnnzh/topoPhonon",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
