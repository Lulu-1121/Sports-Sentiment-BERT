import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset

class ModelTraining:
    def __init__(self, label2id, id2label, seed=42):
        """
        Initialize the ModelTraining class.

        Args:
            label2id (dict): Mapping from emotion labels to IDs.
            id2label (dict): Mapping from IDs to emotion labels.
            seed (int): Random seed for reproducibility.
        """
        self.emotion_cols = list(label2id.keys())  # List of emotion labels
        self.label2id = label2id  # Label to ID mapping
        self.id2label = id2label  # ID to label mapping
        self.seed = seed  # Random seed for reproducibility

        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Initialize variables for traditional models
        self.vectorizer = None
        self.best_logreg = None
        self.best_rf = None

        # Initialize variables for BERT model
        self.bert_trainer = None
        self.bert_model = None

    def train_traditional_models(self, train_df, val_df):
        """
        Train traditional machine learning models: Logistic Regression and Random Forest.

        Args:
            train_df (pd.DataFrame): Training dataset.
            val_df (pd.DataFrame): Validation dataset.

        Returns:
            dict: Performance metrics for Logistic Regression and Random Forest.
        """
        # Vectorize text data using TF-IDF
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        X_train = self.vectorizer.fit_transform(train_df['text_clean'])
        X_val = self.vectorizer.transform(val_df['text_clean'])

        y_train = train_df['label']
        y_val = val_df['label']

        # Train Logistic Regression with hyperparameter tuning
        logreg = LogisticRegression(max_iter=1000, random_state=self.seed)
        param_grid_logreg = {'C': [0.01, 0.1, 1, 10]}
        grid_logreg = GridSearchCV(logreg, param_grid_logreg, cv=3, scoring='f1_macro', verbose=0)
        grid_logreg.fit(X_train, y_train)
        self.best_logreg = grid_logreg.best_estimator_

        # Train Random Forest with hyperparameter tuning
        rf = RandomForestClassifier(random_state=self.seed)
        param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='f1_macro', verbose=0)
        grid_rf.fit(X_train, y_train)
        self.best_rf = grid_rf.best_estimator_

        # Evaluate models on validation data
        y_pred_logreg = self.best_logreg.predict(X_val)
        y_pred_rf = self.best_rf.predict(X_val)

        results = {
            'logreg': {
                'report': classification_report(y_val, y_pred_logreg),
                'f1': f1_score(y_val, y_pred_logreg, average='macro')
            },
            'rf': {
                'report': classification_report(y_val, y_pred_rf),
                'f1': f1_score(y_val, y_pred_rf, average='macro')
            }
        }

        return results

    def train_bert_model(self, train_df, val_df):
        """
        Train a BERT model for sequence classification.

        Args:
            train_df (pd.DataFrame): Training dataset.
            val_df (pd.DataFrame): Validation dataset.

        Returns:
            dict: Performance metrics for the BERT model.
        """
        model_name = "bert-base-uncased"  # Pretrained BERT model name
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

        # Tokenize training and validation data
        train_encodings = tokenizer(train_df['text_clean'].tolist(), truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_df['text_clean'].tolist(), truncation=True, padding=True, max_length=128)

        # Map labels to IDs
        train_labels = train_df['label'].map(self.label2id).values
        val_labels = val_df['label'].map(self.label2id).values

        # Prepare HuggingFace datasets
        hf_train_dataset = HFDataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })

        hf_val_dataset = HFDataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })

        # Load pretrained BERT model
        self.bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(self.emotion_cols))

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        # Define metrics for evaluation
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

        # Train the BERT model
        self.bert_trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        self.bert_trainer.train()
        # metrics = self.bert_trainer.evaluate()

        preds = self.bert_trainer.predict(hf_val_dataset).predictions
        pred_labels = np.argmax(preds, axis=1)

        report = classification_report(val_labels, pred_labels, target_names=self.emotion_cols)
        f1 = f1_score(val_labels, pred_labels, average='macro')

        return {'report': report, 'f1': f1}

    def compare_models(self, results_traditional, results_bert):
        """
        Compare the performance of Logistic Regression, Random Forest, and BERT.

        Args:
            results_traditional (dict): Metrics from traditional models.
            results_bert (dict): Metrics from the BERT model.

        Displays:
            A bar chart comparing the Macro-F1 scores.
        """
        # Create a DataFrame for comparison
        comparison = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'BERT'],
            'F1_score': [results_traditional['logreg']['f1'], results_traditional['rf']['f1'], results_bert['f1']]
        })
        print(comparison)

        # Visualize comparison
        sns.barplot(data=comparison, x='Model', y='F1_score')
        plt.title("Model Comparison (Macro-F1)")
        plt.ylim(0, 1)
        plt.show()
