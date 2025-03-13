# N-Gram-Project

## 1. Introduction
This project implements an N-gram language model to assist code completion for Java methods. The model predicts the next token in a sequence using statistical probabilities derived from a training corpus.

## 2. Dataset Creation
The training data was gathered in a text file (`training.txt`), where each line contains a single Java method. We preprocess this data using a custom regex-based tokenizer that preserves both code words and punctuation. The dataset is then split into training (70%), evaluation (15%), and test (15%) sets with a fixed random seed for reproducibility.

## 3. Model Training Methodology
Our implementation includes:
- **Tokenization:** A custom function tokenizes each Java method while retaining essential punctuation.
- **N-gram Generation:** For each method, overlapping N-grams are generated and counts for each context and subsequent token are maintained.
- **Smoothing:** Laplace smoothing is applied to handle unseen tokens.
- **Evaluation:** We train models for various n-values (3, 4, 5, and 7) and select the best model based on the lowest perplexity computed on the evaluation set.
- **Token Generation:** Instead of a purely greedy approach, we use temperature-based sampling to generate diverse code completions.

A parallel teacher model is trained using instructor-provided data (`instructor_data.txt`), and its evaluation results are stored in `results_teacher_model.json`.

## 4. Evaluation Results
- **Student Model:** The best model (n=3) achieved a perplexity of approximately 403, and code completions were generated for 100 test methods. Results are saved in `results_student_model.json`.
- **Teacher Model:** (If available, similar evaluation is performed and results are saved in `results_teacher_model.json`.)

## 5. Repository and Execution
The complete source code (including `NGram_Model.py`), datasets, results, and this report are hosted on GitHub:
[GitHub Repository Link]([https://github.com/Mohameddeidd/N-Gram-Project])

To run the project:
1. Ensure `training.txt` (and optionally `instructor_data.txt`) are in the project folder.
2. Install the necessary Python packages (`nltk`, `scikit-learn`, and `matplotlib`).
3. Execute the main script with:
