Below is your complete README in one block. You can copy and paste this directly into VS Code:

---

# GenAI for Software Development (Ngram)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run N-gram](#23-run-n-gram)  
* [3 Report](#3-report)  

---

# **1. Introduction**  
This project explores **code completion in Java**, leveraging **N-gram language modeling**. The N-gram model predicts the next token in a sequence by learning the probability distributions of token occurrences in training data. The model selects the most probable token based on learned patterns, making it a fundamental technique in natural language processing and software engineering automation.  

---

# **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/Mohameddeidd/N-Gram-Project.git
```

(2) Navigate into the repository:
```shell
~ $ cd N-Gram-Project
~/N-Gram-Project $
```

(3) Set up a virtual environment and activate it:

For macOS/Linux:
```shell
~/N-Gram-Project $ python -m venv ./venv/
~/N-Gram-Project $ source venv/bin/activate
(venv) ~/N-Gram-Project $
```

To deactivate the virtual environment, use the command:
```shell
(venv) $ deactivate
```

## **2.2 Install Packages**

Install the required dependencies:
```shell
(venv) ~/N-Gram-Project $ pip install -r requirements.txt
```

## **2.3 Run N-gram**

(1) Run N-gram Demo

The script takes a corpus of Java methods as input and automatically identifies the best-performing model based on a specific N-value. It then evaluates the selected model on the test set extracted according to the assignment specifications. Since the training corpus differs from both the instructor-provided dataset and our own dataset, we store the results in files named `results_student_model.json` and `results_teacher_model.json` to distinguish them accordingly.

Run the demo with:
```shell
(venv) ~/N-Gram-Project $ python NGram_Model.py training.txt
```

*(Ensure that `training.txt` is present in the project folder. If you also wish to train using the instructor-provided data, include `instructor_data.txt` as required.)*

---

# **3. Report**

The assignment report is available in the file `Assignment_Report.pdf`.

---