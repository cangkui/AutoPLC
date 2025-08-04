# AutoPLC: Generating Vendor-Aware Structured Text for Programmable Logic Controllers

[👉中文版本](README_zh.md)

This repository contains supplementary materials for our research on AutoPLC, a flexible framework for structured text (ST) generation. The materials are organized into the following main directories:

## Directory Structure

### 1. Benchmarks
The folder `data/benchmarks` includes 4 datasets used in our experiments:
- **oscat**: Contains tasks and data relevant to the OSCAT library.
- **lgf**: Includes tasks related to LGF functions.
- **competition**: Comprises datasets derived from the industrial code generation competition.
- **agents4plc benchmark**: The benchmark proposed by the Agents4PLC Team.

<!-- ### 2. **Experiment_Results**
This folder `exps` holds the experimental results obtained from our study. The results demonstrate the performance of different baselines and configurations of AutoPLC on the provided benchmarks. -->

### 3. Source Code
This folder contains the code used for our experiments and implementations. It is further organized into the following subdirectories:

<!-- - **baselines**: Contains the baseline implementations used for comparison in our study. -->
- **rag_data**: Includes resources for RAG, such as the Rq2ST benchmark and the case library.
- **autoplc_scl**: Represents the implementation of AutoPLC applied to Siemens SCL.
- **autoplc_st**: Represents the implementation of AutoPLC applied to CODESYS ST.

---

## Usage Guide

### Installation

We recommend to run AutoPLC on Windows 10 Professional, which is compatible with both TIA Portal V19 and CODESYS V3.5 SP20.

1. Install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

2. Bash Environment Setup  
Prior to executing the scripts, please prepare the environment for AutoPLC, copy the `.env.example` file to `.env` and fill in the necessary information. This step is exclusively for testing purposes and does not involve any leakage of personal information. 

3. Compiler Tools Setup  
- For Siemens SCL: we used the openness api provided by Siemens, which requires 1) C# .Net Framework runtime; 2) TIA Portal V19 on Windows. See [TIACompileService](https://github.com/cangkui/TIACompileService). The repository provides source code, and a tested binary package is provided in Release for download.
- For CODESYS ST: we used the codesys script engine, which requires CODESYS V3.5 SP20 on Windows. See [CODESYSCompileService](https://github.com/cangkui/CODESYSCompileService). A tested CODESYS installer is also provided in Release.

> Openness compilation tools are developed by ourselves. For CODESYS compilation tool, we have referenced the implementation of this repository: [codesys-api](https://github.com/johannesPettersson80/codesys-api). Thanks very much for the contribution of the author of this repository.  
> We encapsulate each tool as a standard http service for AutoPLC's calling. In order to reduce service load, we deploy multiple services based on multiple machines in the LAN, which can be configured in the configuration file.
> The compilation service address can be modified in the yaml configuration file.

4. Knowledge Base Setup  
We use the online knowledge base provided by the Zhipu Team and integrated their glm-airx model for reranking. See [https://open.bigmodel.cn/dev/howuse/retrieval](https://open.bigmodel.cn/dev/howuse/retrieval).

### Running AutoPLC

To run AutoPLC, execute the following command:
```
python src/autoplc/main.py --benchmark [benchmark name] --config [config name, same to config file name]
```

Replace `[benchmark name]` with the name of the benchmark you want to use (e.g., `oscat`, `lgf`, `competition`, `agents4plc`) and `[config name]` with the name of the configuration you want to use. These configurations need to be created in the `src/config` directory.

For ST generation, replace `main.py` with `main_st.py` in the command.

---

2025 AutoPLC Team.
