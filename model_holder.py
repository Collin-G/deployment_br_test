import model_archs.cnn as cnn
import model_archs
import torch
import numpy as np
import spacy

import gensim.downloader as api
from transformers import BertTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer

import model_archs.coherence_model
import model_archs.linkword_model
import model_archs.star_model
import model_archs.filler_word_model
import model_archs.cnn

SPACY_MODEL = spacy.load("en_core_web_sm")
W2V_MODEL = api.load("glove-wiki-gigaword-50")
BERT_MODEL = BertForMaskedLM.from_pretrained('bert-base-uncased')
BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class ModelHolder():

    def __init__(self, model_type, state_path=None, map_location=None):
        """Class following factory pattern for initializing various models

        Args:
            model_type (str): type of model being used
            state_path (str, optional): path to pretrained states of model if applicable. Defaults to None.
            map_location (torch.device, optional): _description_. Defaults to None.
        """
        self.model_type = model_type
        self.state_path = state_path
        self.map_location = map_location

        self.model = None

        
        if model_type == "cnn":
            self.model = model_archs.cnn.CNN()
            self.setup_with_weights()
            
        
        elif model_type == "star":
            self.model = model_archs.star_model.StarModel(SBERT_MODEL, SPACY_MODEL, self.state_path, self.map_location)
            pass

        elif model_type == "coherency":
            self.model = model_archs.coherence_model.CoherenceModel(SPACY_MODEL, SBERT_MODEL, W2V_MODEL)
            pass

        elif model_type == "linkword":
            self.model = model_archs.linkword_model.LinkwordModel(W2V_MODEL, BERT_MODEL, BERT_TOKENIZER)
            pass

        elif model_type == "filler_word":
            self.model = model_archs.filler_word_model.FillerWordModel()

    
    def setup_with_weights(self):
        # Initialize model if pytorch
        state = torch.load(self.state_path, map_location=self.map_location)

        self.model.load_state_dict(state)
        self.model.eval()
        

    # def setup_without_weights(self):
    #     pass

    def predict(self, features):
        if self.model_type == "cnn":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                # Audio
                output = np.array(torch.sigmoid(self.model(feature_tensor))).mean()
            
            return output
        
        else:
            return self.model.predict(features)


