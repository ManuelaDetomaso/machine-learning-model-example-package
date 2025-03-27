# ML Model Example Package

This is a simple example package for creating a simple Machine Learning model to predict patients with diabete.

This package has the following structure:

ml-examplemodel-package/
├───📁 src/
│   ├───📁 diabete_prediction/
│   ├───📄 prepare_training_data.py
│   └───📄 __init__.py
├───📁 tests/
├───📄 LICENSE.md
├───📄 pyproject.toml
├───📄 README.md
├───📄 requirements.txt
└───📄 setup.py

To update this repo tree: install tree-extended VS Code extension
Right click in the sapce below ml-examplemodel-package and select "Get tree representation" 
The select "No" when prompted on "choose custom configuration".

## Manage package dependencies

Generate the file of this package dependencies with:
```
conda list -e > requirements.txt
```
For those getting odd path references in requirements.txt, use:
```
pip list --format=freeze > requirements.txt
```
This file may be used to create an environment using:
```
conda create --name <env> --file <this file>
```
Copy/paste requirements.txt libraries into the pyproject.toml dependencies.

## Build the package
Activate the environment when you want to use the package, for instance:
```
python -m venv pkg-test-env
```
Activate the environment (in Windows):
```
pkg-test-env\Scripts\activate
```
In the terminal:
Make sure you have the latest version of PyPA’s build installed:
```
py -m pip install --upgrade build
```
Now run this command from the same directory where pyproject.toml is located:
```
python -m build
```
This command should output a lot of text and once completed should generate two files in the dist directory:
dist/
├── diabete_prediction-0.0.1-py3-none-any.whl
└── diabete_prediction-0.0.1.tar.gz

- diabete_prediction-0.0.1-py3-none-any.whl is the built distribution 
- diabete_prediction-0.0.1.tar.gz is the source distribution

## Install the package

In the terminal:
```
pip install dist/diabete_prediction-0.0.1-py3-none-any.whl

# or in case of issues with some libaries' wheels
pip install --only-binary :all: -r requirements.txt
pip install --no-deps dist/your_wheel.whl
```
In a python file or notebook check that the following print
```
print(diabete_prediction.__file__) 
```
shows it's loading from site-packages, not your source directory.

## Checks after installation
Unzip the wheel and check if the config.ini file is there:
```
cd dist
mkdir wheel_contents
cd wheel_contents
tar -xf ../diabete_prediction-0.0.1-py3-none-any.whl
dir diabete_prediction
```
Check if diabete_prediction/config.ini is there.

##  Clean old builds (in Windows)

In Windows, open Powershell terminal and lanuch the following commands:
```
Remove-Item -Recurse -Force dist, build
Remove-Item -Recurse -Force *.egg-info
```

## Modidfy the package in editable mode
In the pkg-test-env:
```
pip uninstall diabete-prediction
```
Then install your project in editable mode:
```
pip install -e .
```
