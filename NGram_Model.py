import nltk
import json
import random
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [word_tokenize(line.strip()) for line in f.readlines()]

def split_dataset(corpus, train_ratio=0.7, eval_ratio=0.15):
    random.shuffle(corpus)
    train_size = int(len(corpus) * train_ratio)
    eval_size = int(len(corpus) * eval_ratio)
    return corpus[:train_size], corpus[train_size:train_size + eval_size], corpus[train_size + eval_size:]

def train_ngram_model(corpus, n):
    ngram_counts = defaultdict(Counter)
    context_counts = Counter()
    
    for sentence in corpus:
        for ngram in ngrams(sentence, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"):
            context, token = tuple(ngram[:-1]), ngram[-1]
            ngram_counts[context][token] += 1
            context_counts[context] += 1
            
    return ngram_counts, context_counts

def compute_perplexity(ngram_counts, context_counts, corpus, n):
    total_log_prob = 0
    num_tokens = 0

    for sentence in corpus:
        for ngram in ngrams(sentence, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"):
            context, token = tuple(ngram[:-1]), ngram[-1]
            count_context = context_counts[context] if context in context_counts else 1
            count_ngram = ngram_counts[context][token] if token in ngram_counts[context] else 1
            prob = count_ngram / count_context
            total_log_prob += -1 * (prob if prob > 0 else 1e-10)
            num_tokens += 1

    return 2 ** (total_log_prob / num_tokens)

def generate_predictions(ngram_counts, context_counts, test_data, n):
    predictions = {}

    for i, sentence in enumerate(test_data[:100]):
        context = tuple(sentence[:n-1])
        predicted_tokens = []

        while len(predicted_tokens) < 10: 
            if context in ngram_counts:
                sorted_predictions = sorted(
                    ngram_counts[context].items(), key=lambda x: x[1], reverse=True
                )
                token, count = sorted_predictions[0]
                prob = count / context_counts[context]
                predicted_tokens.append((token, f"{prob:.3f}"))
                context = tuple(list(context[1:]) + [token])
            else:
                break

        if not predicted_tokens:  
            predicted_tokens.append(("UNKNOWN", "0.000"))

        predictions[i] = predicted_tokens

    return predictions


if __name__ == "__main__":
    train_file = "training.txt"

    corpus = load_dataset(train_file)
    train_set, eval_set, test_set = split_dataset(corpus)

    n_values = [3, 4, 5, 7]
    best_n = min(n_values, key=lambda n: compute_perplexity(*train_ngram_model(train_set, n), eval_set, n))

    print(f"Best N-gram model found: N={best_n}")

    student_ngram_counts, student_context_counts = train_ngram_model(train_set, best_n)
    student_predictions = generate_predictions(student_ngram_counts, student_context_counts, test_set, best_n)

    with open("results_student_model.json", "w") as f:
        json.dump(student_predictions, f, indent=2)

    print("Saved results_student_model.json")

    teacher_ngram_counts, teacher_context_counts = train_ngram_model(train_set, best_n)
    teacher_predictions = generate_predictions(teacher_ngram_counts, teacher_context_counts, test_set, best_n)

    with open("results_teacher_model.json", "w") as f:
        json.dump(teacher_predictions, f, indent=2)

    print("Saved results_teacher_model.json")
