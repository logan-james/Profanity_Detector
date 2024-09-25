# Profanity Detection Comment Classifier

This project is a machine learning-based profanity classifier that detects if a comment is profane or non-profane. It uses a logistic regression model to classify text and displays results through a simple web interface built with Streamlit.

## Features

* Classifies comments as Profane or Non-Profane.
* Provides accuracy metrics and visualizations such as confusion matrices, pie charts, and heatmaps for precision, recall, and F1 scores.
* Allows user input via a text box and displays results dynamically.

## Prerequisites

Before running the project, you need the following installed on your system:

* **Python 3.9 or later** (if using Conda, it will handle this)
* **Anaconda** or **Miniconda** (for managing Python environments)
* Internet connection (to download required packages)

## Setup Instructions (Conda Environment)

### Step 1: Install Anaconda (or Miniconda)

Download and install Anaconda or Miniconda [here](https://www.anaconda.com/download/), from the official Anaconda or Miniconda websites.

### Step 2: Download the project

Download the project files (or the compressed .zip) and extract them to your desired location.

### Step 3: Open the Anaconda Prompt

On Windows, search for **Anaconda Prompt** in the Start Menu and open it.

### Step 4: Navigate to the project folder

Use the following command to move to the directory where you extracted the project files:

```bash
cd path_to_your_project_folder
````
For example:
```bash
cd C:\Users\YourUser\Downloads\Profanity_Detector
```
### Step 5: Create and activate the Conda environment

Create a new Conda environment with Python 3.9:

```bash
conda create --name profanity_env python=3.9
```

Activate the newly created environment:

```bash
conda activate profanity_env
```

### Step 6: Install the required packages

Install the required dependencies from the requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 7: Run application

After the dependencies are installed, run the application.

```bash
streamlit run main.py
```

This will automatically open the application in your default web browser. If the browser does not open, navigate to the URL printed in the terminal (usually http://localhost:8501).

### Step 8: Use the Profanity Classifier

1. Enter any comment in the text box provided and press Enter or click Submit.
2. The system will classify the comment as Profane or Non-Profane, displaying the prediction below the submission form.
3. You can explore various metrics and visualizations using the sidebar.

### Step 9: Deactivate the environment

When you're done running the program, deactivate the Conda environment with:

```bash
conda deactivate
```

### Step 10: Closing the Application

To stop the application, return to the terminal where you ran the streamlit run command and press Ctrl + C.
Files in the Project

+ `main.py`: Main script for running the Streamlit interface.
+ `train_model.py`: Script used for training the machine learning model (you don't need to run this).
+ `profanity_logistic_model.pkl`: The pre-trained logistic regression model for classification.
+ `vectorizer.pkl`: The TF-IDF vectorizer for text processing.
+ `test_data.pkl`: Preprocessed test data used for evaluation purposes.
+ `requirements.txt`: List of all dependencies required for the project.
+ `train.csv`: The dataset used to train the classifier.

### Troubleshooting

#### Issue: Streamlit Not Recognized

If the terminal does not recognize the `streamlit` command, ensure the environment is activated:

```bash
conda activate profanity_env
```

Then try running the application again:

```bash
streamlit run main.py
```

#### Issue: Matplotlib or Seaborn Errors

If you encounter errors related to `matplotlib` or `seaborn`, follow these steps:

1. Ensure that all dependencies have been installed correctly:

```bash
pip install -r requirements.txt
```

2. If errors persist, ensure that the latest version of Anaconda is installed. You can uninstall and reinstall Anaconda if necessary.2. If errors persist, ensure that the latest version of Anaconda is installed. You can uninstall and reinstall Anaconda if necessary.

#### Issue: Conda Command Not Found

If `conda` is not recognized as a command, ensure that Anaconda is installed and added to your system's PATH. If you left ***"Add Anaconda to my PATH environment variable"*** unchecked during installation, launch Anaconda from the start menu and run commands inside the Anaconda Prompt.





