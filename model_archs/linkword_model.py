import re
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
# Define your target linking_words and their token IDs
LINKING_WORDS_TO_MASK = [
    # Coordinating linking_words
    "and", "but", "or", "nor", "for", "so", "yet",

    # Subordinating linking_words
    "because", "although", "since", "unless", "while", "whereas", "if", 
     "as",

    # Correlative linking_words
    "both", "either", "neither", 
    "also", "whether",

    # Conjunctive Adverbs (Transition Words)
    "however", "therefore", "moreover", "consequently", "thus", 
    "meanwhile", "furthermore", "likewise", "nonetheless", "similarly", 


    # Other Linking Words
    "besides", "specifically", "particularly", 
]

class LinkwordModel:

    def __init__(self, w2v_model, bert_model, bert_tokenizer):
        self.w2v_model = w2v_model
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
    
    def predict(self, combined_sentence):
        combined_sentence = combined_sentence.split()
        used_links = []
        masked_sentence = []
        for word in combined_sentence:
            # Keep only alphabetic characters for comparison and lowercase the word
            cleaned_word = re.sub(r'[^a-zA-Z]', '', word).lower()
            
            # Check if the cleaned word is in the linking words to mask
            if cleaned_word in LINKING_WORDS_TO_MASK:
                # Replace the word with "[MASK]" while keeping punctuation
                masked_word = "[MASK]" + word[len(cleaned_word):]  # Retain punctuation
                masked_sentence.append(masked_word)  # Add the masked word
                used_links.append(cleaned_word)
            else:
                masked_sentence.append(word)  # Keep the original word

        # Join the masked sentence back into a string
        masked_sentence_str = ' '.join(masked_sentence)
        combined_sentence = masked_sentence_str
      

        # Tokenize the input sentence
        input_ids = self.bert_tokenizer.encode(combined_sentence, return_tensors="pt")
        mask_indices = (input_ids == self.bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]  # Get mask indices

        # Get predictions from BERT
        predicted_tokens = {}
        with torch.no_grad():
            outputs = self.bert_model(input_ids)
            predictions = outputs.logits

        # Retrieve the predicted token IDs for the masked positions
        for mask_idx in mask_indices:
            mask_token_logits = predictions[0, mask_idx]
            
            # Get the top predictions
            top_predictions = torch.topk(mask_token_logits, 20).indices  # Get top 10 predictions
            
            found_linking_word = False  # Flag to indicate if a valid linking word was found
            for prediction_id in top_predictions:
                predicted_token = self.bert_tokenizer.decode(prediction_id.item())
                
                # Check if the predicted token is in the predefined list of linking words
                if predicted_token in LINKING_WORDS_TO_MASK:
                    predicted_tokens[mask_idx.item()] = predicted_token
                    found_linking_word = True
                    break  # Stop searching once a valid word is found

            # Optionally, handle cases where no valid linking word was found
            if not found_linking_word:
                predicted_tokens[mask_idx.item()] = "No valid linking word found"

        # Print the prediction for each masked position
        for idx, token in predicted_tokens.items():
            print(f"Prediction for masked token at index {idx}: {token}")

        predicted_links = (list(predicted_tokens.values()))
        print(used_links)


        score = 0
        for i1, i2 in zip(predicted_links, used_links):
            embeddings1 = self.w2v_model[i1]
            embeddings2 = self.w2v_model[i2]
            sim = 1-cosine(embeddings1, embeddings2)
            score += sim

        score = score/len(used_links)
        return score

        
