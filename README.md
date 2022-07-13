# machine-learning-projects

## Local Setup

1. Generate a virtual environment.
    - `python3 -m venv .venv`
2. Activate the virtual environment from the root.
    - `source .venv/bin/activate`
3. Install requirements.
    - `pip3 install -r requirements.txt`

## Updating requirements

1. From project root, with venv activated, run code below.
    - `pip3 freeze > requirements.txt`