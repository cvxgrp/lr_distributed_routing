from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="lrrouting",
    version="0.0.1",
    packages=["lrrouting"],
    license="GPLv3",
    description="Low Rank Distributed Approximate Routing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # install_requires=[
    #     "numpy >= 1.22.2",
    #     "scipy >= 1.8.0",
    #     "cvxpy >= 1.2.0",
    #     "matplotlib >= 3.5.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
 
# conda create -n hlr python=3.9
# conda activate hlr
# conda install numpy scipy matplotlib seaborn pandas numba networkx dask
# pip install torch==2.0.0 ipython
# python setup.py install
# conda config --prepend channels conda-forge
# pip install osmnx
# conda install scipy matplotlib seaborn pandas numba networkx dask
# conda install profilehooks memory_profiler
# pip install pypardiso==0.4.3
# pip install -U pymde
# pip install cvxpy==1.4.2 numpy==1.21.6 numba==0.55.0 osmnx==1.2.1 osqp
# pip install objgraph