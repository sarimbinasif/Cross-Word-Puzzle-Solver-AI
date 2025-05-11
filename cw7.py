import os
import nltk
import spacy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import wordnet
import joblib
from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class CrosswordMLModel:
    def __init__(self, data_dir="./data"):
        """Initialize the ML model for crossword solving."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Load spaCy model
        logger.info("Loading NLP models...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load or create crossword dictionary
        self.dictionary_path = os.path.join(data_dir, "crossword_dictionary.txt")
        self.crossword_dictionary = self.load_crossword_dictionary()
        logger.info(f"Loaded {len(self.crossword_dictionary)} words in crossword dictionary")
        
        # Load dataset
        self.dataset_path = os.path.join(data_dir, "crossword_dataset.csv")
        self.clue_answer_pairs = self.load_dataset()
        logger.info(f"Loaded {len(self.clue_answer_pairs)} clue-answer pairs")
        
        # Prepare TF-IDF vectorizer
        self.vectorizer_path = os.path.join(data_dir, "tfidf_vectorizer.joblib")
        self.vectorizer = self.load_or_train_vectorizer()
        
        # Train or load the classifier model
        self.classifier_path = os.path.join(data_dir, 'crossword_rf_model.joblib')
        self.ranking_model_path = os.path.join(data_dir, 'crossword_ranking_model.joblib')
        
        # Load or train models
        if os.path.exists(self.classifier_path) and os.path.exists(self.ranking_model_path):
            logger.info("Loading existing ML models...")
            self.classifier = joblib.load(self.classifier_path)
            self.ranking_model = joblib.load(self.ranking_model_path)
        else:
            logger.info("Training new ML models...")
            self.classifier, self.ranking_model = self.train_models()
            
        # Track feedback for model improvement
        self.feedback_path = os.path.join(data_dir, 'user_feedback.csv')
        if os.path.exists(self.feedback_path):
            self.feedback_data = pd.read_csv(self.feedback_path).to_dict('records')
        else:
            self.feedback_data = []
    
    def load_crossword_dictionary(self):
        """Load or create a dictionary of common crossword answers."""
        if os.path.exists(self.dictionary_path):
            with open(self.dictionary_path, 'r') as f:
                return set(line.strip().upper() for line in f)
        else:
            # Create a basic dictionary
            logger.info("Creating basic crossword dictionary...")
            dictionary = self.create_basic_dictionary()
            
            # Save the dictionary
            with open(self.dictionary_path, 'w') as f:
                for word in sorted(dictionary):
                    f.write(f"{word}\n")
            
            return dictionary
    
    def create_basic_dictionary(self):
        """Create a basic dictionary of words commonly found in crosswords."""
        # Start with some basic words
        dictionary = set()
        
        # Add words from WordNet
        for synset in list(wordnet.all_synsets())[:10000]:  # Limit to prevent memory issues
            word = synset.name().split('.')[0].upper()
            if 2 <= len(word) <= 15 and word.isalpha():  # Common crossword length constraints
                dictionary.add(word)
        
        # Add common abbreviations and short words
        common_short_words = [
            "AM", "PM", "ET", "PT", "EST", "PST", "CEO", "CFO", "CTO", "VP",
            "NYC", "LA", "SF", "DC", "UK", "EU", "UN", "WHO", "NBA", "NFL",
            "MLB", "NHL", "CNN", "BBC", "NBC", "CBS", "ABC", "FOX", "A", "AN",
            "THE", "AND", "BUT", "OR", "NOR", "FOR", "YET", "SO", "AT", "BY",
            "IN", "OF", "ON", "TO", "UP", "ACE", "ADO", "AID", "AIL", "AIM"
        ]
        dictionary.update(common_short_words)
        
        return dictionary
    
    def load_dataset(self):
        """Load crossword dataset from CSV or create a sample dataset."""
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
        else:
            # Create a sample dataset if none exists
            logger.info("Creating sample dataset...")
            df = self.create_sample_dataset()
            df.to_csv(self.dataset_path, index=False)
        
        # Clean the dataset: drop rows with missing or invalid answers
        df = df.dropna(subset=['answer'])  # Drop rows where 'answer' is NaN
        df['answer'] = df['answer'].astype(str).str.upper()  # Ensure all answers are uppercase strings
        df['clue'] = df['clue'].astype(str)  # Ensure all clues are strings
        
        # Filter out non-alphabetic answers or those with special characters
        df = df[df['answer'].str.match(r'^[A-Z]+$')]
        
        # Try to expand the dataset if it's too small for training
        if len(df) < 1000:
            logger.info("Dataset is small, attempting to augment with additional data...")
            df = self.augment_dataset(df)
            df.to_csv(self.dataset_path, index=False)
        
        sample_size = min(len(df), 5000)
        df = df.sample(n=sample_size, random_state=42)  # Use only 5,000 rows for testing
        return list(zip(df['clue'], df['answer']))
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration."""
        return pd.DataFrame({
            'clue': [
                "Capital of France", "Man's best friend", "Feline pet",
                "Cereal grain", "Computer network", "Opposite of north",
                "Flying machine", "Large African mammal with trunk",
                "Timekeeping device", "Frozen water", "Apple's mobile OS",
                "Ocean surrounding Hawaii", "Boxing great Muhammad",
                "Color of the sky", "Planet nearest to the sun",
                "Detective Sherlock", "Playwright Shakespeare",
                "Soft drinks", "Type of tree that produces acorns",
                "Largest planet in our solar system", "Opposite of west",
                "Month after February", "Thanksgiving bird", "Symbol of peace",
                "Fictional monster made from body parts", "Santa's helper",
                "Currency of Japan", "Highest mountain on Earth",
                "Final resting place", "Capital of Japan", "Movie awards",
                "Nocturnal flying mammal", "Large body of saltwater",
                "King of the jungle", "Seven deadly ___", "Garden tool",
                "Ancient Egyptian king", "Winged horse of mythology",
                "Capital of Italy", "Large flightless bird"
            ],
            'answer': [
                "PARIS", "DOG", "CAT", "WHEAT", "INTERNET", "SOUTH", 
                "AIRPLANE", "ELEPHANT", "CLOCK", "ICE", "IOS", "PACIFIC", 
                "ALI", "BLUE", "MERCURY", "HOLMES", "WILLIAM", "SODA",
                "OAK", "JUPITER", "EAST", "MARCH", "TURKEY", "DOVE",
                "FRANKENSTEIN", "ELF", "YEN", "EVEREST", "GRAVE",
                "TOKYO", "OSCAR", "BAT", "OCEAN", "LION", "SINS",
                "HOE", "PHARAOH", "PEGASUS", "ROME", "EMU"
            ]
        })
    
    def augment_dataset(self, df):
        """Attempt to augment the dataset with additional clue-answer pairs."""
        # In a real implementation, this would fetch data from online sources
        # For this example, we'll add some more pairs manually
        additional_data = pd.DataFrame({
            'clue': [
                "Comedian Jerry", "Precious metal", "Unit of sound intensity",
                "Physicist Einstein", "Popular search engine", "Largest US state",
                "Tiny particle", "Volcanic rock", "Bird of prey", "Famous vampire",
                "Disney deer", "Tree house builder", "Precious stone", "Computer company",
                "US currency", "Car part", "Type of dance", "Coffee shop",
                "Superhero with spider powers", "Capital of England"
            ],
            'answer': [
                "SEINFELD", "GOLD", "DECIBEL", "ALBERT", "GOOGLE", "ALASKA",
                "ATOM", "LAVA", "EAGLE", "DRACULA", "BAMBI", "TREEHOUSE",
                "DIAMOND", "APPLE", "DOLLAR", "ENGINE", "TANGO", "STARBUCKS",
                "SPIDERMAN", "LONDON"
            ]
        })
        
        # Combine with original data
        return pd.concat([df, additional_data], ignore_index=True)
    
    def load_or_train_vectorizer(self):
        """Load existing TF-IDF vectorizer or train a new one."""
        if os.path.exists(self.vectorizer_path):
            return joblib.load(self.vectorizer_path)
        else:
            # Extract all clues and answers for training the vectorizer
            clues = [clue for clue, _ in self.clue_answer_pairs]
            answers = [answer for _, answer in self.clue_answer_pairs]
            
            # Combine with wordnet definitions for richer vocabulary
            texts = clues + answers
            for answer in answers:
                for synset in wordnet.synsets(answer.lower()):
                    if synset.definition():
                        texts.append(synset.definition())
            
            # Create and train vectorizer
            vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                max_features=5000,
                lowercase=True,
                min_df=2
            )
            vectorizer.fit(texts)
            
            # Save the vectorizer
            joblib.dump(vectorizer, self.vectorizer_path)
            
            return vectorizer
    
    def extract_features(self, clue, answer=None):
        """Extract enhanced features from a clue-answer pair for ML model."""
        try:
            doc = self.nlp(clue)
            features = {}
            
            # Basic NLP features
            features['num_tokens'] = len(doc)
            features['num_entities'] = len(doc.ents)
            features['has_number'] = int(any(token.like_num for token in doc))
            features['clue_length'] = len(clue)
            
            # Enhanced POS features
            pos_counts = {pos: 0 for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']}
            for token in doc:
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1
            features.update({f'num_{pos.lower()}': count for pos, count in pos_counts.items()})
            
            # Semantic features
            # WordNet-based features
            clue_nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN']
            try:
                features['num_synonyms'] = sum(len(wordnet.synsets(noun)) for noun in clue_nouns)
            except:
                features['num_synonyms'] = 0
            
            # Named entity features
            entity_types = {'PERSON': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0, 'DATE': 0}
            for ent in doc.ents:
                if ent.label_ in entity_types:
                    entity_types[ent.label_] += 1
            features.update({f'ent_{ent_type.lower()}': count for ent_type, count in entity_types.items()})
            
            # Crossword-specific pattern detection
            crossword_patterns = {
                'anagram': ['scrambled', 'mixed', 'confused', 'jumbled', 'arranged', 'disorganized'],
                'hidden': ['hidden', 'within', 'in', 'inside', 'concealed', 'tucked'],
                'homophone': ['sounds', 'heard', 'say', 'audibly', 'spoken', 'aloud', 'listening'],
                'reversal': ['back', 'return', 'reversed', 'backwards', 'reflecting'],
                'deletion': ['without', 'missing', 'losing', 'dropped', 'removed', 'cut'],
                'initial': ['initially', 'first', 'leader', 'head', 'start'],
                'final': ['finally', 'last', 'end', 'ultimate', 'tail'],
            }
            
            for pattern_type, indicators in crossword_patterns.items():
                features[f'has_{pattern_type}'] = int(any(ind in clue.lower() for ind in indicators))
            
            # Question features
            features['is_question'] = int('?' in clue)
            features['is_fill_blank'] = int('___' in clue or '_' in clue or 'blank' in clue.lower())
            
            # Answer features if available
            if answer and isinstance(answer, str):
                answer = answer.upper()
                features['answer_length'] = len(answer)
                features['first_letter'] = ord(answer[0]) - ord('A') if answer else -1
                features['last_letter'] = ord(answer[-1]) - ord('A') if answer else -1
                features['vowel_count'] = sum(1 for c in answer if c in 'AEIOU')
                features['consonant_count'] = len(answer) - features['vowel_count']
                features['has_repeated_letter'] = int(any(answer.count(c) > 1 for c in answer))
                
                # Scrabble score as a feature
                scrabble_scores = {
                    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
                    'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
                    'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
                }
                features['scrabble_score'] = sum(scrabble_scores.get(c, 0) for c in answer)
            else:
                # Default values when no answer is provided
                features['answer_length'] = -1
                features['first_letter'] = -1
                features['last_letter'] = -1
                features['vowel_count'] = -1
                features['consonant_count'] = -1
                features['has_repeated_letter'] = -1
                features['scrabble_score'] = -1
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return basic features as fallback
            basic_features = {
                'num_tokens': len(clue.split()),
                'clue_length': len(clue),
                'has_number': int(any(c.isdigit() for c in clue)),
                'answer_length': len(answer) if answer else -1,
                'first_letter': ord(answer[0]) - ord('A') if answer and answer else -1
            }
            return basic_features
    
    def extract_answer_features(self, answer):
        """Extract features specific to an answer word."""
        answer = answer.upper()
        features = {
            'answer_length': len(answer),
            'first_letter': ord(answer[0]) - ord('A'),
            'last_letter': ord(answer[-1]) - ord('A'),
            'vowel_count': sum(1 for c in answer if c in 'AEIOU'),
            'consonant_count': len(answer) - sum(1 for c in answer if c in 'AEIOU'),
            'has_repeated_letter': int(any(answer.count(c) > 1 for c in answer)),
            'common_crossword_word': int(answer in self.crossword_dictionary)
        }
        
        # Scrabble score as a feature
        scrabble_scores = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
            'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
            'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
        }
        features['scrabble_score'] = sum(scrabble_scores.get(c, 0) for c in answer)
        
        return features
    
    def compute_clue_answer_similarity(self, clue, answer):
        """Compute semantic similarity between clue and answer."""
        try:
            # Ensure the vectorizer has been fitted
            if not hasattr(self.vectorizer, 'vocabulary_'):
                logger.warning("TF-IDF vectorizer not fitted. Skipping similarity calculation.")
                return {'clue_answer_similarity': 0.0}
            
            # Vector representations
            clue_vector = self.vectorizer.transform([clue])
            
            # Get answer-related words (the answer itself and its synonyms)
            answer_words = [answer.lower()]
            for synset in wordnet.synsets(answer.lower()):
                answer_words.extend([lemma.name() for lemma in synset.lemmas()])
                answer_words.append(synset.definition())
            
            # Combine answer words into a single string for vectorization
            answer_text = " ".join(set(answer_words))
            answer_vector = self.vectorizer.transform([answer_text])
            
            # Compute cosine similarity
            similarity = cosine_similarity(clue_vector, answer_vector)[0][0]
            
            return {'clue_answer_similarity': similarity}
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return {'clue_answer_similarity': 0.0}
    
    def prepare_training_data(self):
        """Prepare training data for the ML models."""
        # Extract features for each clue-answer pair
        X_data = []
        y_data = []
        
        logger.info("Preparing training data...")
        for clue, answer in self.clue_answer_pairs:
            try:
                features = self.extract_features(clue, answer)
                X_data.append(features)
                y_data.append(answer)
            except Exception as e:
                logger.error(f"Error processing training pair: {e}")
        
        # Convert to DataFrame for easier handling
        X_df = pd.DataFrame(X_data)
        
        return X_df, y_data
    
    def prepare_ranking_data(self):
        """Prepare training data for the ranking model."""
        X_data = []
        y_data = []
        
        logger.info("Preparing ranking model data...")
        # Limit the number of examples to avoid memory issues
        for i, (clue, correct_answer) in enumerate(self.clue_answer_pairs):
            if i >= 5000:  # Process up to 5000 pairs for ranking model
                break
            
            try:
                # Create positive example (clue + correct answer)
                features = self.extract_features(clue, correct_answer)
                features.update(self.extract_answer_features(correct_answer))
                features.update(self.compute_clue_answer_similarity(clue, correct_answer))
                X_data.append(features)
                y_data.append(1)  # Positive label
                
                # Create negative examples (clue + incorrect answers)
                incorrect_answers = self.sample_incorrect_answers(correct_answer, k=2)
                for incorrect in incorrect_answers:
                    features = self.extract_features(clue, incorrect)
                    features.update(self.extract_answer_features(incorrect))
                    features.update(self.compute_clue_answer_similarity(clue, incorrect))
                    X_data.append(features)
                    y_data.append(0)  # Negative label
            except Exception as e:
                logger.error(f"Error processing ranking pair: {e}")
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X_data)
        
        return X_df, y_data
    
    def sample_incorrect_answers(self, correct_answer, k=3):
        """Sample k incorrect answers of similar length to the correct answer."""
        correct_len = len(correct_answer)
        
        # Get a list of all answers in our dataset with similar length
        all_answers = [ans for _, ans in self.clue_answer_pairs]
        candidates = [ans for ans in all_answers if abs(len(ans) - correct_len) <= 1 and ans != correct_answer]
        
        # If we don't have enough candidates, add from dictionary
        if len(candidates) < k * 3:  # We want more candidates than k for better sampling
            dict_candidates = [word for word in self.crossword_dictionary 
                              if abs(len(word) - correct_len) <= 1 and word != correct_answer]
            candidates.extend(dict_candidates)
        
        # Deduplicate
        candidates = list(set(candidates))
        
        # Sample k incorrect answers
        if len(candidates) > k:
            return np.random.choice(candidates, k, replace=False)
        else:
            return candidates[:k]  # Return all candidates if fewer than k
    
    def train_models(self):
        """Train both classification and ranking models on the crossword dataset."""
        # Train the primary classifier
        logger.info("Training classification model...")
        X_df, y_data = self.prepare_training_data()
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_data, test_size=0.2, random_state=42
        )
        
        # Train a RandomForest classifier
        classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=30,
            min_samples_split=5,
            n_jobs=-1, 
            random_state=42
        )
        classifier.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Classification model accuracy: {accuracy:.4f}")
        
        # Save the classification model
        joblib.dump(classifier, self.classifier_path)
        
        # Train the ranking model
        logger.info("Training ranking model...")
        X_rank_df, y_rank = self.prepare_ranking_data()
        
        # Split into training and testing sets
        X_rank_train, X_rank_test, y_rank_train, y_rank_test = train_test_split(
            X_rank_df, y_rank, test_size=0.2, random_state=42
        )
        
        # Train a RandomForest classifier for ranking
        ranking_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=20,
            min_samples_split=10,
            n_jobs=-1, 
            random_state=42
        )
        ranking_model.fit(X_rank_train, y_rank_train)
        
        # Evaluate the ranking model
        y_rank_pred = ranking_model.predict(X_rank_test)
        ranking_accuracy = accuracy_score(y_rank_test, y_rank_pred)
        logger.info(f"Ranking model accuracy: {ranking_accuracy:.4f}")
        
        # Save the ranking model
        joblib.dump(ranking_model, self.ranking_model_path)
        
        # Generate model performance visualization
        self.visualize_model_performance(classifier, X_test, y_test)
        
        return classifier, ranking_model
    
    def visualize_model_performance(self, model, X_test, y_test):
        """Create and save visualizations of model performance."""
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Feature importance
            plt.figure(figsize=(12, 8))
            features = X_test.columns
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title('Feature Importances')
            plt.bar(range(min(20, len(importances))), importances[indices][:20])
            plt.xticks(range(min(20, len(importances))), [features[i] for i in indices][:20], rotation=90)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(self.data_dir, 'model_feature_importance.png'))
            plt.close()
            
            # Top-k accuracy metrics
            top_k_accuracy = self.calculate_top_k_accuracy(model, X_test, y_test)
            
            # Plot top-k accuracy
            plt.figure(figsize=(10, 6))
            k_values = list(top_k_accuracy.keys())
            accuracy_values = list(top_k_accuracy.values())
            
            plt.plot(k_values, accuracy_values, marker='o')
            plt.title('Top-K Accuracy')
            plt.xlabel('K Value')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(self.data_dir, 'top_k_accuracy.png'))
            plt.close()
            
            # Confusion matrix (limited to top classes due to potentially many answers)
            try:
                top_answers = pd.Series(y_test).value_counts().head(10).index
                mask = pd.Series(y_test).isin(top_answers)
                
                if sum(mask) > 0:
                    plt.figure(figsize=(10, 8))
                    cm = pd.crosstab(
                        pd.Series(np.array(y_test)[mask]), 
                        pd.Series(y_pred[mask]), 
                        rownames=['Actual'], 
                        colnames=['Predicted']
                    )
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix (Top 10 Classes)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.data_dir, 'model_confusion_matrix.png'))
                    plt.close()
            except Exception as e:
                logger.error(f"Error generating confusion matrix: {e}")
        
        except Exception as e:
            logger.error(f"Error visualizing model performance: {e}")
    
    def calculate_top_k_accuracy(self, model, X_test, y_test):
        """Calculate top-k accuracy for various k values."""
        probs = model.predict_proba(X_test)
        
        # Calculate top-k accuracy for k=1, 3, 5, 10
        top_k_accuracy = {}
        for k in [1, 3, 5, 10]:
            correct_in_top_k = 0
            for i, true_label in enumerate(y_test):
                # Get indices of top k predictions
                top_k_indices = np.argsort(probs[i])[::-1][:k]
                top_k_classes = [model.classes_[idx] for idx in top_k_indices]
                if true_label in top_k_classes:
                    correct_in_top_k += 1
            
            top_k_accuracy[k] = correct_in_top_k / len(y_test)
        
        return top_k_accuracy
    
    def get_candidate_answers(self, length=None, known_letters=None):
        """Get candidate answers that satisfy given constraints."""
        # Start with all known answers from the dataset
        all_answers = set(answer.upper() for _, answer in self.clue_answer_pairs)
        
        # Add common crossword words from the dictionary
        all_answers.update(self.crossword_dictionary)
        
        # Filter by length if specified
        if length is not None:
            candidates = {answer for answer in all_answers if len(answer) == length}
        else:
            candidates = all_answers
        
        # Filter by known letters if specified
        if known_letters and isinstance(known_letters, dict):
            filtered = set()
            for answer in candidates:
                matches = True
                for pos, letter in known_letters.items():
                    if pos < len(answer) and answer[pos] != letter.upper():
                        matches = False
                        break
                if matches:
                    filtered.add(answer)
            candidates = filtered
        
        # Limit the number of candidates to avoid memory issues
        candidates = list(candidates)
        if len(candidates) > 10000:
            logger.info(f"Limiting candidates from {len(candidates)} to 10000")
            candidates = np.random.choice(candidates, 10000, replace=False)
        
        return list(candidates)
    
    def predict_answer(self, clue, length=None, known_letters=None):
        """Predict the answer for a given clue using a hybrid approach."""
        try:
            # Extract features from the clue
            clue_features = self.extract_features(clue)
            
            # Fill in the answer length if provided
            if length is not None:
                clue_features['answer_length'] = length
            
            # Approach 1: Direct classification (quickly get some candidates)
            try:
                # Convert to DataFrame
                features_df = pd.DataFrame([clue_features])
                
                # Get top classification predictions with probabilities
                y_probs = self.classifier.predict_proba(features_df)
                class_indices = np.argsort(y_probs[0])[::-1][:30]  # Get top 30 predictions
                top_classes = [self.classifier.classes_[idx] for idx in class_indices]
                
                # Filter predictions by length and known letters if needed
                if length is not None or known_letters:
                    filtered_classes = []
                    for answer in top_classes:
                        if length is not None and len(answer) != length:
                            continue
                        if known_letters:
                            matches = True
                            for pos, letter in known_letters.items():
                                if pos < len(answer) and answer[pos] != letter.upper():
                                    matches = False
                                    break
                            if not matches:
                                continue
                        filtered_classes.append(answer)
                    direct_predictions = filtered_classes[:10]  # Limit to top 10
                else:
                    direct_predictions = top_classes[:10]
            except Exception as e:
                logger.error(f"Error in classification prediction: {e}")
                direct_predictions = []
            
            # Approach 2: Generate and rank candidates
            # Get candidate answers based on constraints
            candidates = self.get_candidate_answers(length, known_letters)
            
            if not candidates:
                logger.warning("No candidates found matching constraints")
                if not direct_predictions:
                    return []
                else:
                    return [(p, 1.0) for p in direct_predictions]
            
            # Score candidates using the ranking model
            candidate_scores = []
            batch_size = 500  # Process candidates in batches to avoid memory issues
            
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i+batch_size]
                batch_features = []
                
                for candidate in batch:
                    features = self.extract_features(clue, candidate)
                    features.update(self.extract_answer_features(candidate))
                    features.update(self.compute_clue_answer_similarity(clue, candidate))
                    batch_features.append(features)
                
                # Convert to DataFrame
                batch_df = pd.DataFrame(batch_features)
                
                # Predict probability of being correct
                try:
                    batch_probs = self.ranking_model.predict_proba(batch_df)[:, 1]  # Probability of class 1 (correct)
                    candidate_scores.extend(list(zip(batch, batch_probs)))
                except Exception as e:
                    logger.error(f"Error ranking candidates batch: {e}")
            
            # Sort candidates by score
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_predictions = candidate_scores[:20]  # Top 20 from ranking
            
            # Combine both approaches with preference for direct predictions
            combined_predictions = {}
            
            # Add direct predictions with boosted confidence
            for pred in direct_predictions:
                combined_predictions[pred] = 1.5  # Boosted base confidence
            
            # Add ranked predictions
            for pred, score in ranked_predictions:
                if pred in combined_predictions:
                    combined_predictions[pred] = max(combined_predictions[pred], score)
                else:
                    combined_predictions[pred] = score
            
            # Create final sorted list
            final_predictions = [(pred, score) for pred, score in combined_predictions.items()]
            final_predictions.sort(key=lambda x: x[1], reverse=True)
            
            return final_predictions[:10]  # Return top 10
        
        except Exception as e:
            logger.error(f"Error in predict_answer: {e}")
            return []
    
    def predict_with_pattern(self, clue, pattern):
        """Predict an answer that matches a specific pattern (e.g., '?A?T')."""
        # Convert pattern to a dictionary of known letters
        known_letters = {}
        length = len(pattern)
        
        for i, char in enumerate(pattern):
            if char != '?' and char.isalpha():
                known_letters[i] = char.upper()
        
        # Get predictions with the constraints
        return self.predict_answer(clue, length, known_letters)
    
    def record_feedback(self, clue, predicted_answer, correct_answer, is_correct):
        """Record user feedback for model improvement."""
        feedback = {
            'clue': clue,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'timestamp': pd.Timestamp.now()
        }
        
        self.feedback_data.append(feedback)
        
        # Save feedback to CSV
        pd.DataFrame(self.feedback_data).to_csv(self.feedback_path, index=False)
        
        # Check if we should retrain the model
        if len(self.feedback_data) % 1 == 0:  # Consider retraining after every 100 feedback items
            logger.info("Considering model retraining based on feedback...")
            correct_count = sum(1 for item in self.feedback_data[-100:] if item['is_correct'])
            if correct_count < 70:  # Less than 70% accuracy
                self.update_dataset_with_feedback()
                self.train_models()
    
    def update_dataset_with_feedback(self):
        """Update the dataset with user feedback for model improvement."""
        # Extract correct answers from feedback
        new_pairs = []
        for item in self.feedback_data:
            if item['correct_answer'] and item['clue']:
                new_pairs.append((item['clue'], item['correct_answer']))
        
        # Add to existing dataset
        existing_pairs = set(self.clue_answer_pairs)
        for pair in new_pairs:
            if pair not in existing_pairs:
                existing_pairs.add(pair)
        
        # Update dataset
        self.clue_answer_pairs = list(existing_pairs)
        
        # Save updated dataset
        clues, answers = zip(*self.clue_answer_pairs)
        pd.DataFrame({'clue': clues, 'answer': answers}).to_csv(self.dataset_path, index=False)
        
        logger.info(f"Dataset updated with feedback data. New size: {len(self.clue_answer_pairs)}")
    
    def evaluate_model(self):
        """Evaluate model performance on a test set and generate metrics."""
        # Prepare evaluation data
        X_df, y_data = self.prepare_training_data()
        
        # Split into train and test sets
        _, X_test, _, y_test = train_test_split(X_df, y_data, test_size=0.2, random_state=42)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}")
        
        # Calculate top-k accuracy
        top_k_accuracy = self.calculate_top_k_accuracy(self.classifier, X_test, y_test)
        for k, acc in top_k_accuracy.items():
            logger.info(f"Top-{k} accuracy: {acc:.4f}")
        
        # Generate visualizations
        self.visualize_model_performance(self.classifier, X_test, y_test)
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'top_k_accuracy': top_k_accuracy
        }
    
    def suggest_answer_improvements(self, clue, answer):
        """Suggest improvements for an answer based on the model's knowledge."""
        # Get the model's top predictions
        predictions = self.predict_answer(clue)
        
        if not predictions:
            return "No suggestions available."
        
        # Check if the given answer is among top predictions
        is_in_top = any(pred[0].upper() == answer.upper() for pred in predictions)
        
        suggestions = []
        
        if not is_in_top:
            suggestions.append(f"The answer '{answer}' is not among the model's top predictions.")
            top_pred = predictions[0][0]
            suggestions.append(f"Top suggestion: '{top_pred}'")
        
        # Check if the answer fits the typical pattern for this clue type
        clue_features = self.extract_features(clue)
        answer_features = self.extract_answer_features(answer)
        similarity = self.compute_clue_answer_similarity(clue, answer)['clue_answer_similarity']
        
        if similarity < 0.1:
            suggestions.append(f"The semantic similarity between clue and answer is low ({similarity:.3f}).")
        
        # Check if answer is in our crossword dictionary
        if answer.upper() not in self.crossword_dictionary:
            suggestions.append("This answer is not in our common crossword answers dictionary.")
        
        # Check length appropriateness
        avg_word_len = 5.0  # Example average value
        if len(answer) > 10:
            suggestions.append(f"The answer is quite long ({len(answer)} letters) compared to average ({avg_word_len}).")
        
        return suggestions if suggestions else "The answer looks good!"


# Web application for the crossword solver
app = Flask(__name__)

# Initialize the model
model = None

# Add a flag to ensure initialization happens only once
model_initialized = False

@app.before_request
def initialize_model():
    global model, model_initialized
    if not model_initialized:
        model = CrosswordMLModel()
        model_initialized = True

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predicting answers."""
    try:
        data = request.get_json()
        clue = data.get('clue', '')
        length = data.get('length')
        pattern = data.get('pattern')
        
        if length and length.isdigit():
            length = int(length)
        else:
            length = None
        
        # Convert pattern to known letters
        known_letters = None
        if pattern:
            known_letters = {}
            for i, char in enumerate(pattern):
                if char != '?' and char.isalpha():
                    known_letters[i] = char.upper()
        
        # Get predictions
        predictions = model.predict_answer(clue, length, known_letters)
        
        # Format response
        response = {
            'clue': clue,
            'predictions': [
                {'answer': pred[0], 'confidence': float(pred[1])} 
                for pred in predictions
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing prediction request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """API endpoint for recording user feedback."""
    try:
        data = request.get_json()
        clue = data.get('clue', '')
        predicted = data.get('predicted_answer', '')
        correct = data.get('correct_answer', '')
        is_correct = data.get('is_correct', False)
        
        # Record feedback
        model.record_feedback(clue, predicted, correct, is_correct)
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """API endpoint for getting model statistics."""
    try:
        metrics = model.evaluate_model()
        
        # Get dataset stats
        dataset_size = len(model.clue_answer_pairs)
        dictionary_size = len(model.crossword_dictionary)
        
        # Count feedback items
        feedback_count = len(model.feedback_data)
        correct_predictions = sum(1 for item in model.feedback_data if item.get('is_correct', False))
        
        response = {
            'model_metrics': metrics,
            'dataset_size': dataset_size,
            'dictionary_size': dictionary_size,
            'feedback_total': feedback_count,
            'correct_predictions': correct_predictions
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error generating stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/expand_dictionary', methods=['POST'])
def expand_dictionary():
    """API endpoint for expanding the crossword dictionary from web sources."""
    try:
        data = request.get_json()
        source = data.get('source', 'default')
        
        words_added = update_dictionary_from_source(source)
        
        return jsonify({
            'status': 'success',
            'words_added': words_added
        })
    
    except Exception as e:
        logger.error(f"Error expanding dictionary: {e}")
        return jsonify({'error': str(e)}), 500

def update_dictionary_from_source(source='default'):
    """Update the crossword dictionary from various web sources."""
    words_added = 0
    
    try:
        if source == 'default':
            # Sample source: a list of common crossword answers
            url = "https://www.xwordinfo.com/popular"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                table = soup.find('table', class_='XTab2 autoalt')
                words = []

                for row in table.find_all("tr")[1:]:  # skip the header
                    word_cell = row.find_all("td")[-1]  # last column is 'Words'
                    links = word_cell.find_all("a")
                    for link in links:
                        word = link.text.strip()
                        if word:
                            words.append(word)
                
                # Add to dictionary
                original_size = len(model.crossword_dictionary)
                model.crossword_dictionary.update(words)
                words_added = len(model.crossword_dictionary) - original_size
                
                # Save the updated dictionary
                with open(model.dictionary_path, 'w') as f:
                    for word in sorted(model.crossword_dictionary):
                        f.write(f"{word}\n")
                
                logger.info(f"Added {words_added} new words to dictionary")
        
        elif source == 'wordnet':
            # Add more words from WordNet
            new_words = set()
            for synset in list(wordnet.all_synsets())[:20000]:  # Increased limit
                word = synset.name().split('.')[0].upper()
                if 2 <= len(word) <= 15 and word.isalpha():
                    new_words.add(word)
            
            # Add to dictionary
            original_size = len(model.crossword_dictionary)
            model.crossword_dictionary.update(new_words)
            words_added = len(model.crossword_dictionary) - original_size
            
            # Save the updated dictionary
            with open(model.dictionary_path, 'w') as f:
                for word in sorted(model.crossword_dictionary):
                    f.write(f"{word}\n")
            
            logger.info(f"Added {words_added} new words to dictionary from WordNet")
    
    except Exception as e:
        logger.error(f"Error in update_dictionary_from_source: {e}")
    
    return words_added

if __name__ == "__main__":
    # Initialize the model
    model = CrosswordMLModel()
    
    # Optional: Evaluate model on startup
    model.evaluate_model()
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)