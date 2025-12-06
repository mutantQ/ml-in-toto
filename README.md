# Secure ML Workflows with in-toto

This repository demonstrates a project utilizing in-toto to enhance the security of machine learning (ML) workflows. The goal is to ensure the integrity and authenticity of datasets and models used in ML pipelines by leveraging the capabilities of in-toto.

## Project Overview

Supply chain attacks have become a significant threat in the software and ML development landscape. High-profile incidents like the SolarWinds attack and the Log4j vulnerability have shown the devastating impact that compromised supply chains can have on organizations worldwide. These attacks exploit weaknesses in the development process, injecting malicious code or altering critical components that then propagate through the entire supply chain.

To mitigate such risks, companies like SolarWinds have implemented in-toto in their systems. in-toto provides a framework to ensure the integrity and security of every step in a software or ML supply chain. By verifying that each step in the workflow is performed as expected and by authorized parties, in-toto helps prevent unauthorized alterations and ensures the end product is trustworthy.

This project leverages in-toto to secure ML workflows, providing a reliable and verifiable process from dataset preparation to model distribution.

### Workflow Summary

The `run_all.sh` script demonstrates a typical ML pipeline where:
- **Alice** prepares a dataset.
- **Bob** trains a model using the dataset.
- **Carl** tests the model.
- **Diana** packages the model for distribution.

The process is tracked using in-toto, and if any step is compromised, the verification will fail - indicating a problem in the pipeline. Try `bash run_all.sh --corrupt --dry-run` to see what happens if Alice unknowingly introduces a corrupted dataset.

For more details on how in-toto works and real-world use cases, you can refer to the official [in-toto friends GitHub page](https://github.com/in-toto/friends), which includes SolarWinds among other users.

## Getting Started

### Prerequisites

- **Python 3.8** or later (tested on Ubuntu 20.04 LTS).
- [Git](https://git-scm.com/) for cloning the repository.
- [Conda](https://conda.io/projects/conda/en/latest/index.html) for replicating environment
- Basic understanding of Python and Bash scripting.

### Installation

1. **Clone the Repository, initialize submodules**:
   ```bash
   git clone https://github.com/mutantQ/ml-in-toto.git
   cd ml-in-toto
   git submodule update --init
   ```

2. **Setup conda environment**:
   
   For Ubuntu:
   ```bash
   conda env create -f environment.yml
   conda activate ml-in-toto
   ```
   For MacOS:
   ```bash
   conda env create -f environment-mac.yml
   conda activate ml-in-toto
   ```

### Running the Demo

To see a demo of the integration, follow these steps:

1. **Run the demo**:
   ```bash
   bash run_all.sh [--corrupt] [--dry-run] [epochs N]
   ```

   This script will execute the entire ML workflow, starting from dataset preparation to model distribution, with in-toto recording each step for verification. `--epochs 1` is enough to give >95% accuracy during test.

   **arguments:**\
   --corrupt : Alice/data is overwritten with corrupt data. \
   --dry-run : Skips training by breaking for loop \
   --epochs N : Run N epochs

2. **Clean up**:
   After running the demo, you can delete all generated files by executing:
   ```bash
   bash delete.sh
   ```

## Limitations and Future Work

- **C2PA Integration**: Future work includes integrating C2PA to enhance the provenance and verification of datasets. C2PA will help verify the authenticity of datasets from the point of capture, providing a high-quality data pipeline for ML workflows.

- **Error Handling**: The current demo does not include robust error handling or rollback mechanisms. This is an area for future enhancement.

- **Security Expertise Needed**: Although this project aims to enhance security, I am not a security expert. Contributions from security professionals are greatly appreciated to strengthen the integrity and robustness of this project.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
