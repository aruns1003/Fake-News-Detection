This project focuses on fake news detection using two different approaches:

A classical Machine Learning baseline with TF-IDF vectorization + Logistic Regression.

A state-of-the-art Deep Learning model using DistilBERT fine-tuned for sequence classification.

By comparing both approaches, the project highlights the trade-off between speed, simplicity, and interpretability (Logistic Regression) vs. accuracy and semantic understanding (DistilBERT).

📂 Dataset

Source: Kaggle – Fake and Real News Dataset

Two CSV files:

Fake.csv → Fake news articles

True.csv → Real news articles

Labels added:

0 → Fake

1 → True

⚙️ Methods
1️⃣ Logistic Regression (Baseline)

Preprocessing:

Convert text into TF-IDF features.

Limit features with stop_words='english' and max_df=0.7.

Classifier: Logistic Regression.

Pros:

Extremely fast, interpretable.

Works surprisingly well with textual data.

Cons:

Ignores word context/semantics.

Struggles with subtle differences in language.

2️⃣ DistilBERT (Deep Learning)

Preprocessing:

Tokenization with Hugging Face DistilBertTokenizerFast.

Sequences padded/truncated to fixed length.

Model: DistilBERT fine-tuned with classification head (num_labels=2).

Training: AdamW optimizer, cross-entropy loss, trained for 2 epochs.

Pros:

Captures semantic meaning & context.

Achieves state-of-the-art accuracy.

Cons:

Computationally expensive (hours on CPU, faster on GPU).

Black-box nature makes it less interpretable.

📊 Results
Model	Precision	Recall	F1-score	Accuracy	Training Time
Logistic Regression (TF-IDF)	~0.98	~0.98	~0.98	98%	⚡ Seconds
DistilBERT	1.00	1.00	1.00	100%	🐢 ~7h (CPU)

Key Insight:

Logistic Regression is fast, lightweight, and already very accurate.

DistilBERT achieves perfect results but at a heavy computational cost.

Depending on resources and needs, Logistic Regression may be preferable for quick deployment, while DistilBERT is best for maximum accuracy.
