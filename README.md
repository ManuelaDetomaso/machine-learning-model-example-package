# ML Model Example Package

This is a simple example package for creating a simple Machine Learning model to predict patients with diabete.

This package has the following structure:

```
ml-examplemodel-package/
├───📁 src/
│   ├───📁 diabete_prediction/
│   ├───📄 config_loader.py
│   ├───📄 config.ini
│   ├───📄 prepare_data.py
│   ├───📄 score_data.py
│   ├───📄 train_model.py
│   ├───📄 utils.py
│   └───📄 __init__.py
├───📁 tests/
│   ├───📄 test_data_prepator.py
│   ├───📄 test_model_scorer.py
│   ├───📄 test_model_trainer.py
│   └───📄 test_utils.py
├───📄 LICENSE.md
├───📄 pyproject.toml
├───📄 README.md
├───📄 requirements.txt
└───📄 test_pkg.ipynb
```

To update this repo tree: install tree-extended VS Code extension
Right click in the sapce below ml-examplemodel-package and select "Get tree representation" 
The select "No" when prompted on "choose custom configuration".

## Manage package dependencies

### Install existing dependencies
Install requirements-dev.txt or requirements.txt in pkg-test-env:
```
pip install -r requirements-dev.txt
```
Or this file may be used to create a new environment using:
```
conda create --name <env> --file <this file> # this file: requirements.txt or requirements-dev.txt
```

### Create/ Update the file of dependencies

Generate the file of this package dependencies with:
```
conda list -e > requirements.txt
```
For those getting odd path references in requirements.txt, use:
```
pip list --format=freeze > requirements.txt
```
If needed, copy/paste requirements.txt libraries into the pyproject.toml dependencies.

## Build the package
Activate the environment when you want to use the package, for instance:
Python 3.11.8 is required for this package, check it in the terminal: python --version > python 3.11.8
Or in the terminal:
```
python --version
# if version is different from 3.11, then do:
pyenv install 3.11.8
```
Create the environment:
```
python -m venv pkg-test-env
```
Activate the environment (in Windows):
```
pkg-test-env\Scripts\activate
```
Check the environment is activated in this package folder path:
```
where pip
where python 
```
Then, in the terminal:
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
In a python file or notebook (use test_pkg.ipynb) check that the following print
```
print(diabete_prediction.__file__) 
```
shows it's loading from site-packages, not your source directory.

## Checks after installation
Unzip the wheel and check if all the files are there:
```
cd dist
mkdir wheel_contents
cd wheel_contents
tar -xf ../diabete_prediction-0.0.1-py3-none-any.whl
dir diabete_prediction
```
In particular, check if diabete_prediction/config.ini is there.

##  Clean old builds (in Windows)

In Windows, open Powershell terminal and lanuch the following commands:
```
Remove-Item -Recurse -Force dist, build
Remove-Item -Recurse -Force *.egg-info
```

## Modify and build the package in editable mode
This is meant for 
In the pkg-test-env:
```
pip uninstall diabete-prediction -y
pip uninstall diabete_prediction -y
```
Then install your project in editable mode:
```
python.exe -m pip install --upgrade pip #run the first time the environment is created
pip install -e .
```

## Install the wheel package in MS Fabric

This Medium artical provides a good example: https://robkerr.ai/fabric-custom-libraries/

## Generating Docs

In the environment do:
```
pip install sphinx myst-parser sphinx-autoapi sphinx-autodoc-typehints furo
```
At the root of your project:
```
sphinx-quickstart docs
```
Answer the prompts like this:

- Separate source and build dirs? → yes
- Project name? → ml-examplemodel-package
- Author? → Your name or org
- Project version → 0.1.0
- Others → default

It creates: 
```
docs/
├── build/
├── source/
│   ├── conf.py
│   ├── index.rst
```
Edit docs/source/conf.py:
```
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))  # So Sphinx can find diabete_prediction
```
Enable useful extensions:
```
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",                # Only if you're using Markdown
]
```
Set theme:
```
html_theme = "furo"
```
Run this in your project root:
```
sphinx-apidoc -o docs/source src/diabete_prediction
```
It will generate .rst files like:
```
docs/source/diabete_prediction.prepare_data.rst
```
Build the docs:
```
cd docs
make html 
```
To clean and regenerate:
```
make clean
```
or delete manually docs/build

For markdown docs add the following to conf.py:
```
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc settings
autoapi_type = 'python'
autoapi_dirs = ['../../src/diabete_prediction']
autoapi_keep_files = True  # helpful for debugging
autoapi_add_toctree_entry = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
#html_static_path = ['_static']
# Show type hints inline
autodoc_typehints = "description"
```