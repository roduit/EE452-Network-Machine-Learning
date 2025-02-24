# Anaconda, Python, and Essential Libraries: Onboarding Guide

## Introduction
This guide is designed to help you download Anaconda, set up Python, and install essential libraries such as NumPy and Pandas. It also quickly introduces how to create virtual environments using Conda to guarantee reproducibility and **avoid package conflicts**.

---

## 1. Installing Anaconda
Anaconda is a popular tools that allow you to manage the different python packages and deploy environment efficiently.

### **Step 1: Download Anaconda**
- Visit the official Anaconda download page: [Anaconda Download](https://www.anaconda.com/products/distribution#download-section)
- Choose the installer based on your OS:
  - **Windows**: `.exe` installer
  - **Mac**: `.pkg` installer
  - **Linux**: `.sh` installer
- Follow the installation instructions for your operating system.

### **Step 2: Verify Installation**
After installation, open a terminal or Anaconda Prompt and run:
```bash
conda --version
python --version
```
If these commands return version numbers, the installation was successful.

---

## 2. Understanding Virtual Environments
### **What is a Virtual Environment?**
A virtual environment is an isolated workspace that allows you to manage Python packages **independently** from the system-wide installation. It prevents conflicts between different projects requiring different package versions.

Conda enables the creation of virtual environments effortlessly. Check this [website](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for a detailed guide on how to manage Anaconda environments.

---

## 3. Creating and Managing Conda Environments

### **Step 1: Create a New Conda Environment**
To create an environment named `myenv` with Python 3.10, open a terminal and write:
```bash
conda create --name myenv python=3.10
```

### **Step 2: Activate the Environment**
Before installing packages, activate the environment.:
```bash
conda activate myenv
```
You will now notice that the name in parenthesis has changed from `(base)` (the system wide installation) to `(myenv)` (the desired environment).

### **Step 3: Install Essential Libraries**
Once inside the environment, install commonly used libraries using `pip`:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn networkx

```
Alternatively, you can use `conda` to install your libraries:
```bash
conda install numpy pandas matplotlib scikit-learn seaborn networkx
```

### **Step 4: List Installed Packages**
To check which packages are installed in the current environment:
```bash
pip list
```

### **Step 5: Deactivate the Environment**
To leave the environment and return to the base system:
```bash
conda deactivate
```

### **Step 6: Remove an Environment (if needed)**
To delete an environment and free up space:
```bash
conda remove --name myenv --all
```

---

## 6. Jupyter Notebook & IDE Setup
Jupyter Notebook is useful for interactive Python coding.
- Install Jupyter:
  ```bash
  conda install jupyter
  ```
- Start Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
For IDEs like VS Code, install the Conda extension and link your Conda environment.

---

## 6. Additional Resources
- Official Conda Documentation: [https://docs.conda.io](https://docs.conda.io)
- Python Packages Index: [https://pypi.org](https://pypi.org)
- Jupyter Notebook Guide: [https://jupyter.org](https://jupyter.org)

---

