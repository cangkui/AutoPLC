# Supplementary Materials Overview

This repository contains supplementary materials for our research on AutoPLC, a flexible framework for structured text (ST) generation. The materials are organized into the following main directories:

## Directory Structure

### 1. **Benchmarks**
This folder includes three datasets used in our experiments:
- **oscat**: Contains tasks and data relevant to the OSCAT library.
- **lgf**: Includes tasks related to LGF functions.
- **competition**: Comprises datasets derived from the industrial code generation competition.

### 2. **Experiment_Results**
This directory holds the experimental results obtained from our study. The results demonstrate the performance of different baselines and configurations of AutoPLC on the provided benchmarks.

### 3. **Scripts**
This folder contains the code used for our experiments and implementations. It is further organized into the following subdirectories:

- **experiment**: Contains the baseline implementations used for comparison in our study.
- **rag_data**: Includes resources for retrieval-augmented generation (RAG), such as the Rq2ST benchmark and the case library.
- **scl_team_coder**: Represents the implementation of AutoPLC applied to Siemens SCL.
- **st_team_coder**: Represents the implementation of AutoPLC applied to CODESYS ST.

---

## Usage Guide

### Installation
1. Install the necessary dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Bash Environment Setup
Prior to executing the scripts, please ensure that the following environment variables are properly configured. This step is exclusively for testing purposes and does not involve any leakage of personal information. 

   ```bash
   export GPT_API_KEY_LIST=sk-Poyu4IIJG4VXpQjj6a8cAdE6126e4e26BfA07f62Cb8cDd07,sk-le8JA4P1jeeltzcm6756F97d39A343339405F5Ca09Ef2957
   export GPT_API_KEY=sk-heJMQX0Z4FEBm1ve7a5320F48f034fE688D104E53c2fBe45
   export API_KEY_KNOWLEDGE=664db87328b7f19ffcd10b9a8a4d9147.Ak1fANcqRx4CJPC0
   export DEEPSEEK=sk-089e91649ae948f1b99665c1a8ef0c57
   ```

### Running Baselines
1. Modify the configuration in `experiment/<corresponding_baseline>/run.py` to suit your requirements.
2. Update the experiment entry point in `experiment/__main__.py`.
3. Run the baseline experiment from the root directory:
   ```bash
   python -m experiment
   ```

### Running AutoPLC
1. Modify the `dataset` and `prompt_file` parameter in `st_team_coder/__main__.py` (only oscat_en) or `scl_team_coder/__main__.py`(lgf_en or competition_en) to the one you want to experiment with.
2. Start the experiment from the root directory:
   ```bash
   python -m st_team_coder
   ```
3. Similarly, for Siemens SCL, update `scl_team_coder/__main__.py` and run:
   ```bash
   python -m scl_team_coder
   ```

case from github:
https://github.com/panasewicz/PLC_SCL_Motion_Control_Example/blob/main/Picker_Station.scl
https://github.com/lopez-dev/Siemens-SCL-Source-Files/blob/main/Meldungen/Meldungen.scl
https://github.com/LCC-Automation/OpenPID-TIA-SCL/blob/main/LLCCA_TimeLag_1_0_1.scl

---

Thank you!