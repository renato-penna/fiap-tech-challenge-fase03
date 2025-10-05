
# FIAP Tech Challenge - Phase 03

## Overview

This repository contains the complete workflow for preparing data and fine-tuning a language model, as part of the FIAP Tech Challenge (Phase 03). The process is divided into two main stages: **Data Preparation** and **Fine-Tuning**.

To facilitate execution, the project includes both a **complete notebook** that runs the entire fine-tuning workflow (training, inference, and comparison with base model) and **separate notebooks** for individual tasks, allowing you to execute each step independently by simply clicking "Run All" in Google Colab.

### Available Notebooks:

#### Data Preparation:
1. **`data_preparation.ipynb`** - Data cleaning and preparation

#### Fine-Tuning - Complete Workflow:
2. **`complete fine_tuning.ipynb`** - Complete fine-tuning process including:
   - Training
   - Inference with the fine-tuned model
   - Inference with model saved in Google Drive
   - Inference comparing with the base model

#### Fine-Tuning - Separated Tasks (for individual execution):
3. **`fine_tuning.ipynb`** - Training only
4. **`fine-tunig saved model.ipynb`** - Inference with model saved in Google Drive
5. **`base model.ipynb`** - Inference with base model for comparison

Each notebook is designed to run independently in Google Colab, with "Open in Colab" badges for easy access.

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

## 2. Fine-Tuning

### Complete Workflow (`complete fine_tuning.ipynb`)

#### Purpose
Execute the entire fine-tuning workflow in a single notebook, from training to inference and comparison with the base model.

#### Steps

1. **Mount Google Drive**
	- Mount Google Drive to access datasets and save models.

2. **Format Dataset**
	- Load and format the `trn_finetune.jsonl` file generated in the data preparation step.
	- Transform data into the format required for training.

3. **Install Dependencies**
	- Install required libraries (unsloth, transformers, datasets, etc.).

4. **Setup and Model Configuration**
	- Load a pre-trained language model (e.g., Llama, GPT-based models).
	- Configure LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning.
	- Set up training parameters (batch size, learning rate, epochs, etc.).
	**Parameters for each atempt** (files available in https://drive.google.com/drive/folders/1RzkWy3StDKgjLPwe17r2_JQJYX9LB5aY?usp=drive_link)
		**file: lora_model** (wandb: exalted-surf-1)
			## LoRA
				- r = 16
				- lora_alpha = 16
				- lora_dropout = 0

			## SFTTrainer
				- per_device_train_batch_size = 2
				- gradient_accumulation_steps = 4
				- max_steps = 60
				- learning_rate = 2e-4
				
		**file: lora_model2** (wandb: misunderstood-night-2)
			## LoRA
				- r = 32
				- lora_alpha = 32
				- lora_dropout = 0.05

			## SFTTrainer
				- per_device_train_batch_size = 3
				- gradient_accumulation_steps = 6
				- max_steps = 120
				- learning_rate = 1e-4
				
		**file: lora_model3** (wandb: desert-waterfall-3)
			## LoRA
				- r = 32
				- lora_alpha = 32
				- lora_dropout = 0.05

			## SFTTrainer
				- per_device_train_batch_size = 4
				- gradient_accumulation_steps = 8
				- max_steps = 180
				- learning_rate = 1e-5

5. **Fine-Tuning Process**
	- Tokenize the prompt/completion pairs.
	- Train the model on the dataset using the configured parameters.
	- Monitor training metrics (loss, learning rate, etc.).

6. **Save Model**
	- Save the fine-tuned model to Google Drive for future use.

7. **Run Inference**
	- Test the fine-tuned model with sample prompts.
	- Generate responses and evaluate quality.

8. **Compare with Base Model**
	- Load the base (non-fine-tuned) model.
	- Run the same prompts on the base model.
	- Compare outputs between fine-tuned and base models.

#### Output
- Fine-tuned model weights saved to Google Drive
- Training logs and metrics
- Inference results from both fine-tuned and base models
- Performance comparison

---

### Separated Tasks (for independent execution)

For more flexibility, the fine-tuning process is also available as separate notebooks:

#### 2a. Training Only (`fine_tuning.ipynb`)

**Purpose:** Focus only on training the model without inference.

**Key Steps:**
- Mount Google Drive
- Format dataset
- Install dependencies
- Configure and train the model
- Save model to Google Drive

**Output:**
- Fine-tuned model weights and configuration files
- Training logs and metrics

#### 2b. Inference with Saved Model (`fine-tunig saved model.ipynb`)

**Purpose:** Run inference using a model previously saved to Google Drive.

**Download fine-tuned model**
- https://drive.google.com/drive/folders/1RzkWy3StDKgjLPwe17r2_JQJYX9LB5aY?usp=drive_link

**Key Steps:**
- Mount Google Drive
- Install dependencies
- Load the saved fine-tuned model from Google Drive
- Run inference with sample prompts
- Generate and display responses

**Output:**
- Inference results from the fine-tuned model

#### 2c. Base Model Inference (`base model.ipynb`)

**Purpose:** Run inference using the base (non-fine-tuned) model for comparison purposes.

**Key Steps:**
- Mount Google Drive
- Install dependencies
- Load the base model
- Run inference with sample prompts
- Generate and display responses

**Output:**
- Inference results from the base model
- Baseline for comparing with fine-tuned model performance

---

## Requirements

- Python 3.7+
- Google Colab (recommended) or Jupyter Notebook
- Google Drive account (for storing datasets and models)
- pandas, numpy
- ijson (for data preparation)
- unsloth, transformers, datasets, torch (for fine-tuning)

**Note:** All dependencies are installed automatically within the notebooks. If running locally, install with:

```bash
pip install pandas numpy ijson transformers torch datasets
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Usage

### Quick Start (Recommended)

The easiest way to run the project is using Google Colab:

1. **Clone the repository (optional for reference):**
	```bash
	git clone https://github.com/renato-penna/fiap-tech-challenge-fase03.git
	```

2. **Open notebooks directly in Google Colab:**
	- Each notebook has an "Open in Colab" badge at the top
	- Click the badge to open the notebook in Google Colab
	- Click "Runtime" → "Run all" to execute the entire notebook
	

### Workflow Options

#### Option 1: Complete Workflow (All-in-One)

Best for running the entire pipeline from start to finish:

1. **Run `data_preparation.ipynb`:**
   - Opens in Colab → Click "Run all"
   - Prepares and cleans the dataset
   - Generates `trn_finetune.jsonl`

2. **Run `complete fine_tuning.ipynb`:**
   - Opens in Colab → Click "Run all"
   - Performs training, inference, and comparison with base model
   - Saves model to Google Drive
   - Displays comparison results

#### Option 2: Separated Tasks (Step-by-Step)

Best for understanding each step individually or running specific tasks:

1. **Run `data_preparation.ipynb`:**
   - Opens in Colab → Click "Run all"
   - Generates `trn_finetune.jsonl`

2. **Run `fine_tuning.ipynb`:**
   - Opens in Colab → Click "Run all"
   - Trains the model and saves to Google Drive

3. **Run `fine-tunig saved model.ipynb`:**
   - Opens in Colab → Click "Run all"
   - Loads saved model and runs inference

4. **Run `base model.ipynb`:**
   - Opens in Colab → Click "Run all"
   - Runs inference with base model for comparison

### Tips

- **Google Drive:** Make sure to mount your Google Drive when prompted in the notebooks
- **Runtime:** Select GPU runtime in Colab (Runtime → Change runtime type → GPU) for faster training
- **Dataset:** Ensure your dataset is uploaded to the correct path in Google Drive (`/content/drive/MyDrive/Fiap/trn.json`)
- **Run All:** All notebooks are designed to work with "Run all" - no manual cell execution needed

---

## Project Structure

```
├── data_preparation.ipynb          # Data cleaning and preparation
├── complete fine_tuning.ipynb      # Complete workflow: training + inference + comparison
├── fine_tuning.ipynb               # Training only (separated task)
├── fine-tunig saved model.ipynb    # Inference with saved model (separated task)
├── base model.ipynb                # Inference with base model (separated task)
└── README.md                       # Project documentation
```

### Notebook Descriptions

- **`data_preparation.ipynb`**: Prepares the raw dataset, filters and cleans data, and creates prompt/completion pairs for training.

- **`complete fine_tuning.ipynb`**: All-in-one notebook that includes data formatting, model training, saving to Google Drive, inference with the fine-tuned model, and comparison with the base model.

- **`fine_tuning.ipynb`**: Focuses solely on training the model. Use this if you want to train the model separately without running inference.

- **`fine-tunig saved model.ipynb`**: Loads a previously trained model from Google Drive and runs inference. Useful for testing a saved model without retraining.

- **`base model.ipynb`**: Runs inference using the base (non-fine-tuned) model. Useful for establishing a baseline and comparing with fine-tuned results.

---

## License