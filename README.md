
# FIAP Tech Challenge - Phase 03

## Overview

This repository contains the complete workflow for preparing data and fine-tuning a language model, as part of the FIAP Tech Challenge (Phase 03). The process is divided into two main stages:

1. **Data Preparation** (`data_preparation.ipynb`)
2. **Fine-Tuning** (`fine_tuning.ipynb`)

Each stage is implemented in a dedicated Jupyter Notebook, with clear, reproducible steps and code. This README provides a comprehensive guide to the workflow, requirements, and usage.

---

## 1. Data Preparation (`data_preparation.ipynb`)

### Purpose
Prepare and clean the raw dataset for use in language model fine-tuning. This includes filtering, cleaning, and formatting the data into prompt/completion pairs.

### Steps

1. **Setup and Imports**
	- Import required libraries: `pandas`, `numpy`, `os`, `json`, and `ijson` (for efficient JSON processing).
	- Mount Google Drive (if running in Colab) to access data files.

2. **File Paths**
	- Define paths for the raw JSON data and output files (filtered and processed datasets).

3. **Install Dependencies**
	- Install the `ijson` library for efficient, chunked JSON reading (especially useful for large files).

4. **Filter and Save Data in Chunks**
	- Read the raw JSON data in chunks to handle large files efficiently.
	- Select only the necessary columns (`title`, `content`).
	- Save the filtered data in JSONL format, appending each chunk to a single file.

5. **Load and Clean Filtered Data**
	- Load the filtered data.
	- Remove rows where `title` or `content` is empty or null.
	- Save the cleaned data for further processing.

6. **Data Inspection**
	- Display the head of the processed DataFrame.
	- Show the shape of the data and count null/empty values for quality assurance.

7. **Create Prompt/Completion Pairs**
	- Construct prompt/completion pairs in English:
		- Prompt: `Question: <title>\nAnswer:`
		- Completion: `<content>`
	- Save the final dataset in JSONL format, ready for fine-tuning.

### Output
- `trn_finetune.jsonl`: Cleaned and formatted dataset with prompt/completion pairs for model training.

---

## 2. Fine-Tuning (`fine_tuning.ipynb`)

### Purpose
Use the prepared dataset to fine-tune a language model for question-answering or similar NLP tasks.

### Steps

1. **Setup and Imports**
	- Import necessary libraries for model training (e.g., Hugging Face Transformers, PyTorch, or other frameworks as required).

2. **Load Prepared Data**
	- Load the `trn_finetune.jsonl` file generated in the data preparation step.

3. **Model Selection and Configuration**
	- Choose a pre-trained language model suitable for fine-tuning (e.g., GPT, BERT, etc.).
	- Configure training parameters (batch size, learning rate, epochs, etc.).

4. **Fine-Tuning Process**
	- Tokenize the prompt/completion pairs.
	- Train the model on the dataset.
	- Monitor training metrics and adjust parameters as needed.

5. **Evaluation and Saving**
	- Evaluate the fine-tuned model on validation data (if available).
	- Save the trained model and any relevant artifacts.

### Output
- Fine-tuned model weights and configuration files.
- Training logs and evaluation metrics.

---

## Requirements

- Python 3.7+
- Jupyter Notebook
- pandas, numpy
- ijson
- (For fine-tuning) Hugging Face Transformers, PyTorch or TensorFlow (as required)

Install dependencies with:

```bash
pip install pandas numpy ijson transformers torch
```

---

## Usage

1. **Clone the repository:**
	```bash
	git clone https://github.com/renato-penna/fiap-tech-challenge-fase03.git
	cd fiap-tech-challenge-fase03
	```

2. **Open and run `data_preparation.ipynb`:**
	- Follow the notebook cells to prepare and clean your dataset.
	- Ensure the output file `trn_finetune.jsonl` is generated.

3. **Open and run `fine_tuning.ipynb`:**
	- Use the prepared dataset to fine-tune your chosen language model.
	- Save the resulting model and artifacts.

---

## Project Structure

```
├── data_preparation.ipynb   # Data cleaning and preparation notebook
├── fine_tuning.ipynb        # Model fine-tuning notebook
├── README.md                # Project documentation
```

---

## License

This project is for educational purposes as part of the FIAP Tech Challenge. Please refer to the repository or institution for licensing details.
