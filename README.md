# control_element_classification

## Background
This application was created for a master thesis to improve the automatisation of train validation tests.
It is an AI-based classification of textual test steps to atomic hardware control element identifiers. The application contains a trained SVM and it's internal functions are controllable via configuration keywords.

## Interface
The application is communicating via external parameters and can be integrated into other python programs. The interface contains 3 inputs (test step, cabin location, configuration keyword) and 1 output (classification result)

## Setup
The application can be integrated via a virtual environment

1. Create a venv: `python -m venv venv`
2. Activate venv: `venv\Scripts\activate.bat`
3. Install required packages `pip install -r requirements.txt`

## Run application
- Use run.sh file to perform a test run with a self-written test step text
- Integrate the application into another project:
```
project_path = Path("/pfad/zu/project_b")
    python = project_path / "venv/bin/python"
    entrypoint = project_path / "run.py"
    cmd = [
        str(python),
        str(entrypoint),
        "--cab", cab, 
        "--test_step", test_step,
        "--config", config
        ]
    result = subprocess.run(cmd)
    element_name = result.stdout.strip().splitlines()[-1] if result else None
```
