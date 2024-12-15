import pandas as pd
import re
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, filepath, test_ratio=1/20, random_state=42):
        """
        Initialize the DataPreprocessing class.

        Args:
            filepath (str): Path to the dataset file.
            test_ratio (float): Ratio of data to be used for validation.
            random_state (int): Seed for reproducibility in train-test splitting.
        """
        self.filepath = filepath  # Path to the dataset
        self.emotion_cols = []  # List of emotion columns
        self.test_ratio = test_ratio  # Test data split ratio
        self.random_state = random_state  # Random seed for reproducibility
        self.train_df = None  # DataFrame for training data
        self.val_df = None  # DataFrame for validation data
        self.label2id = {}  # Mapping from labels to IDs
        self.id2label = {}  # Mapping from IDs to labels

    def load_data(self):
        """
        Load data from the file, dropping rows with missing text values.

        Returns:
            pd.DataFrame: Cleaned DataFrame with valid text entries.
        """
        df = pd.read_csv(self.filepath, encoding="latin-1")  # Load the dataset
        df = df.dropna(subset=['text'])  # Drop rows where 'text' column is NaN
        return df

    def clean_text(self, text):
        """
        Clean and preprocess text by removing URLs, special characters, and extra spaces.

        Args:
            text (str): Raw text data.

        Returns:
            str: Cleaned text.
        """
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        return text

    def preprocess_data(self, df):
        """
        Preprocess the data by filtering and preparing it for modeling.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with clean text and labels.
        """
        self.emotion_cols = list(df.columns[2:])  # Identify emotion columns
        df = df[df['example_very_unclear'] == "FALSE"]  # Filter unclear examples
        df = df[df[self.emotion_cols].eq(1).any(axis=1)]  # Keep rows with any emotion
        df = df[['text'] + self.emotion_cols]  # Keep only relevant columns
        df[self.emotion_cols] = df[self.emotion_cols].astype(int)  # Ensure integer types
        df['text_clean'] = df['text'].apply(self.clean_text)  # Clean text data
        
        # Assign a single label based on the emotion with the highest value
        df['label'] = df[self.emotion_cols].idxmax(axis=1)

        # Create mappings between labels and IDs
        unique_labels = df['label'].unique().tolist()
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        return df

    def split_data(self, df):
        """
        Split the dataset into training and validation sets.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Sets:
            self.train_df: Training DataFrame.
            self.val_df: Validation DataFrame.
        """
        train_df, val_df = train_test_split(
            df, 
            test_size=self.test_ratio, 
            random_state=self.random_state, 
            stratify=df['label']  # Ensure balanced label distribution
        )
        self.train_df = train_df
        self.val_df = val_df

    def get_data(self):
        """
        Retrieve the processed training and validation data, along with label mappings.

        Returns:
            tuple: (train_df, val_df, label2id, id2label)
        """
        return self.train_df, self.val_df, self.label2id, self.id2label