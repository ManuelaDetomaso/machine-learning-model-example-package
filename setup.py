# setup is the backend used to build this distribution package
# It creates the wheel file to upload / install when it's required.
# to build a .whl file into the dist/ folder, run this command from terminal:
#     python setup.py bdist_wheel
# However the setup.cfg/setup.py files are now replaced by the pyproject.toml file, 
# as the standard way of specifying project metadata,
# and so also the previous command is now deprecated and replaced by: 
# python -m build

# If this commands throws an error like: "your current network has https://repo.anaconda.com blocked":
# check if ssl_verify is True with the command:
#     conda config --show
# then do:
#     conda config --set ssl_verify False
# If the error persists, check if anaconda access if blocked by a firewall, 
# in this case use a connection (like a vpn) which removes the firewall block.


from setuptools import setup, find_packages

setup(
    name="diabete_prediction",
    version="0.1.0",
    description="A library to create a simple machine learning model for diabete prediction",
    packages=find_packages(),
    istall_requires=["pandas", "pyspark"]
)
