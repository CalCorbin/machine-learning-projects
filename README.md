# machine-learning-projects

## Local Setup

1. Generate a virtual environment.
    - `python3 -m venv .venv`
2. Activate the virtual environment from the root.
    - `source .venv/bin/activate`
3. Install requirements.
    - `pip3 install -r requirements.txt`
4. Run migrations.
    - `python3 manage.py migrate`
5. Start the server.
    - `python3 manage.py runserver`

## Updating requirements

1. From project root, with venv activated, run code below.
    - `pip3 freeze > requirements.txt`

## Training Digits Model

1. Active venv.
2. Run the following management command.
    - `python3 manage.py makedigitsmodel`