from sklearn.metrics.pairwise import cosine_similarity

class CoherenceModel():

    def __init__(self, spacy_model, sbert_model, w2v_model):
        self.spacy_model = spacy_model
        self.sbert_model = sbert_model
        self.w2v_model = w2v_model


    def predict(self, features):
        return self.get_coherence(self.paragraph_to_sentences(features))[0][0]


    def calculate_coherency_score(self, sentence1, sentence2):
        # Process the sentences with spaCy
        doc1 = self.spacy_model(sentence1)
        doc2 = self.spacy_model(sentence2)
    

        # Extract relevant words (nouns, verbs, adjectives, adverbs)
        relevant_words1 = [
            token.text.lower() for token in doc1 
            if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and not token.is_stop
        ]
        relevant_words2 = [
            token.text.lower() for token in doc2 
            if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and not token.is_stop
        ]
        # print(relevant_words1, relevant_words2)

        
        # Get embeddings for relevant words
        embeddings1 = [self.w2v_model[word] for word in relevant_words1 if word in self.w2v_model.key_to_index]
        embeddings2 = [self.w2v_model[word] for word in relevant_words2 if word in self.w2v_model.key_to_index]


        # Check if we have any embeddings to work with
        if embeddings1 and embeddings2:
            # Calculate cosine similarity between the embeddings
            sim_matrix = cosine_similarity(embeddings1, embeddings2)
            coherency_score = sim_matrix.mean()  # Average the similarities
        else:
            coherency_score = 0.0

        return coherency_score

    def get_coherence(self, paragraph, alpha=0.5):
        
      
        total = 0
        for i in range(len(paragraph)-1):
            word_part = 0
            sentence_part = 0
            for k in range(i+1,len(paragraph)):

                # Words between sentences with high simillarity allow for a higher score.
                word_part+= self.calculate_coherency_score(paragraph[i], paragraph[k])
            

                # Higher semantic difference between sentences allow for high score
                sbert1 = self.sbert_model.encode(paragraph[i])
                sbert2 = self.sbert_model.encode(paragraph[k])
                
                sentence_part += cosine_similarity(sbert1.reshape(1,-1), sbert2.reshape(1,-1))
                # print(sentence_part)
            total += alpha*(word_part/(len(paragraph)-i))+(1-(1-alpha)*(sentence_part/(len(paragraph)-i)))

        return (total/len(paragraph))
    
    def paragraph_to_sentences(self, paragraph):
        # Process the paragraph using spaCy
        doc = self.spacy_model(paragraph)
        # Extract sentences from the spaCy Doc object
        sentences = [sent.text for sent in doc.sents]
        return sentences