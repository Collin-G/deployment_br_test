
import torch.nn as nn
import torch
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First layer
        self.relu = nn.ReLU()                  # Activation function
        self.fc2 = nn.Linear(128, 64)          # Second layer
        self.fc3 = nn.Linear(64, num_classes)  # Output layer (4 classes)
        self.softmax = nn.Softmax(dim=1)    # Softmax for multi-class

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)
    
class StarModel():
    
    def __init__(self, sbert_model, spacy_model, state_path, map_location):
        self.sbert_model = sbert_model
        self.spacy_model = spacy_model

        self.neural_net = SimpleNN(384, 4)
        self.state_path = state_path
        self.map_location = map_location

        self.setup_with_weights()

    def setup_with_weights(self):
        state = torch.load(self.state_path, map_location=self.map_location)

        self.neural_net.load_state_dict(state)
        self.neural_net.eval()


    def predict(self, paragraph):
        sentences = self.paragraph_to_sentences(paragraph)
        return self.get_score(self.nn_predict(sentences))
        
      
    def paragraph_to_sentences(self, paragraph):
        # Process the paragraph using spaCy
        doc = self.spacy_model(paragraph)
        # Extract sentences from the spaCy Doc object
        sentences = [sent.text for sent in doc.sents]
        return sentences
    
    def nn_predict(self, features):
        test_embeddings = self.sbert_model.encode(features, convert_to_tensor=True)

        with torch.no_grad():
            self.neural_net.eval()  # Set the model to evaluation mode
            test_outputs = self.neural_net(test_embeddings)
            predicted_classes = test_outputs.argmax(dim=1).numpy()  # Get predicted class indices
        
        return np.array(predicted_classes)
        # Print predictions

    def get_score(self, matrix):
        argmax_indices = np.argmax(matrix, axis=1)  # Indices of max elements
        max_values = matrix[np.arange(matrix.shape[0]), argmax_indices]  # Corresponding max values

        # Combine indices and values into an array
        # result = np.column_stack((argmax_indices, max_values))

        no_change = 0
        for i in range(len(argmax_indices)):
            if i > 0:
                if argmax_indices[i] == argmax_indices[i-1]:
                    no_change += 1
        
        unique_elements, counts = np.unique(argmax_indices, return_counts=True)


        score = (max_values.mean() + no_change/len(argmax_indices) + len(unique_elements)/4)/3
        return score

    


    

        


            

