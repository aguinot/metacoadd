from setuptools import find_packages, setup

with open("README.md") as rm:
    long_description = rm.read()

setup(
    name="metacoadd",
    version="0.0.1",
    author="Axel Guinot",
    author_email="axel.guinot.astro@gmail.com",
    description="Shear image and coadd them to run metadetection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aguinot/metacoadd",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
