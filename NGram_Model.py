import re
import nltk
import math
import pickle
import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def tokenize_java_code(code):
    tokens = re.findall(r'\w+|[^\s\w]', code)
    return tokens

def split_dataset(data, test_size=0.2, eval_size=0.1):
    train_data, temp_data = train_test_split(data, test_size=test_size+eval_size, random_state=42)
    eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, eval_data, test_data

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()

    def train(self, corpus):
        for sentence in corpus:
            for token in sentence:
                self.vocabulary.add(token)
            if len(sentence) < self.n:
                continue
            for i in range(len(sentence) - self.n + 1):
                context = tuple(sentence[i:i+self.n-1])
                next_token = sentence[i+self.n-1]
                self.ngrams[(context, next_token)] += 1
                self.context_counts[context] += 1

    def get_probability(self, context, next_token):
        ngram_count = self.ngrams.get((context, next_token), 0)
        context_count = self.context_counts.get(context, 0)
        vocab_size = len(self.vocabulary)
        return (ngram_count + 1) / (context_count + vocab_size)

    def sample_next_token_with_prob(self, context, temperature=1.0):
        tokens = []
        probs = []
        for token in self.vocabulary:
            p = self.get_probability(context, token)
            tokens.append(token)
            probs.append(p ** (1/temperature))
        total = sum(probs)
        if total == 0:
            return None, 0
        normalized_probs = [p/total for p in probs]
        selected_token = random.choices(tokens, weights=normalized_probs, k=1)[0]
        index = tokens.index(selected_token)
        selected_prob = normalized_probs[index]
        return selected_token, selected_prob

def calculate_perplexity(model, test_data):
    log_sum = 0
    total_tokens = 0
    for sentence in test_data:
        if len(sentence) < model.n:
            continue
        for i in range(model.n-1, len(sentence)):
            context = tuple(sentence[i-model.n+1:i])
            next_token = sentence[i]
            prob = model.get_probability(context, next_token)
            log_sum += math.log(prob)
            total_tokens += 1
    return math.exp(-log_sum/total_tokens) if total_tokens > 0 else float('inf')

def train_and_evaluate(train_data, eval_data, n_values):
    best_model = None
    best_perplexity = float('inf')
    best_n = None
    for n in n_values:
        model = NgramModel(n)
        model.train(train_data)
        perplexity = calculate_perplexity(model, eval_data)
        print(f"Perplexity for n={n}: {perplexity}")
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_model = model
            best_n = n
    return best_model, best_perplexity, best_n

def generate_completion(model, initial_context, max_length=20, temperature=1.0):
    predictions = []
    context = list(initial_context)
    for _ in range(max_length):
        ctx = tuple(context[-(model.n-1):])
        token, prob = model.sample_next_token_with_prob(ctx, temperature=temperature)
        if token is None:
            break
        predictions.append((token, f"{prob:.3f}"))
        context.append(token)
        if token in [';', '}']:
            break
    return predictions

def generate_predictions_json(model, test_data, best_n, temperature=0.8):
    results = {}
    count = 0
    for method in test_data:
        if len(method) < best_n-1:
            continue
        initial_context = tuple(method[:best_n-1])
        predictions = generate_completion(model, initial_context, max_length=20, temperature=temperature)
        results[str(count)] = predictions
        count += 1
        if count >= 100:
            break
    return results

if __name__ == "__main__":

    with open('training.txt', 'r') as f:
        input_train = f.readlines()
    tokenized_data = [tokenize_java_code(method.strip()) for method in input_train if method.strip()]
    
    train_data, eval_data, test_data = split_dataset(tokenized_data)
    n_values = [3, 4, 5, 7]
    best_model, best_perplexity, best_n = train_and_evaluate(train_data, eval_data, n_values)
    print(f"Best student model found with n={best_n} and perplexity={best_perplexity}")
    student_results = generate_predictions_json(best_model, test_data, best_n, temperature=0.8)
    with open("results_student_model.json", "w") as f:
        json.dump(student_results, f, indent=2)
    with open('ngram_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('training.txt', 'r') as f:
        teacher_input = f.readlines()
    teacher_tokenized_data = [tokenize_java_code(method.strip()) for method in teacher_input if method.strip()]
    teacher_train, teacher_eval, teacher_test = split_dataset(teacher_tokenized_data)
    best_model_teacher, best_perplexity_teacher, best_n_teacher = train_and_evaluate(teacher_train, teacher_eval, n_values)
    print(f"Best teacher model found with n={best_n_teacher} and perplexity={best_perplexity_teacher}")
    teacher_results = generate_predictions_json(best_model_teacher, teacher_test, best_n_teacher, temperature=0.8)
    with open("results_teacher_model.json", "w") as f:
        json.dump(teacher_results, f, indent=2)
