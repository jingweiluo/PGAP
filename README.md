# PGAP
This is the official implementation of our paper "PGAP: Purity-Guided Active Prompting for EEG Decoding With LLMs".

## Paper
- IEEE Xplore: https://ieeexplore.ieee.org/document/11609540
- DOI: https://doi.org/10.1109/MCI.2026.3665159

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

## Citation
If you find this work useful, please cite:

```bibtex
@article{luo2026pgap,
  title={PGAP: Purity-Guided Active Prompting for EEG Decoding With LLMs},
  author={Luo, Jingwei and Wang, Ziwei and Liu, Dingkun and Wu, Dongrui},
  journal={IEEE Computational Intelligence Magazine},
  volume={21},
  number={3},
  pages={49--64},
  year={2026},
  month={Aug.},
  doi={10.1109/MCI.2026.3665159},
  publisher={IEEE}
}
```
