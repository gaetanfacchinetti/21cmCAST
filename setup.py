import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py21cmcast",
    version="1.0.0",
    author="<Gaetan Facchinetti>",
    author_email="<gaetanfacc@gmail.com>",
    description="<Fisher forecasts with 21cm experiments>",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<https://github.com/gaetanfacchinetti/21cmCAST>",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU LICENCE :: GPL3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)