import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnboost",
    version="0.0.1",
    author="Igor Svystun",
    author_email="i.svystun.dev@gmail.com",
    packages=["nnboost"],
    description="A small gradient boosting library that uses neural network based learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isvystun/nnboost",
    license='MIT',
    python_requires='>=3.7',
    install_requires=[
         "tensorflow>=2.4",
         "numpy>=1.20"
    ]
)