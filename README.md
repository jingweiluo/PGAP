# PAGP
This is the official implementation of our paper "[Exploring EEG Decoding with LLMs: A Purity-Guided Active Prompting Framework]".

## Environment Set Up
Install required packages:
```bash
conda create -n pgap python=3.11.9
conda activate pgap
pip install -r requirements.txt
```

## Run Experiments
### Add API Keys
First, add the API keys for each model platform to the .env file; otherwise, the APIs cannot be called.
### Run Command.py
```bash
python command.py
```