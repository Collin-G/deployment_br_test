import string
class FillerWordModel():

    def __init__(self):
        pass

    def predict(self, features):
        """Gives score for filler words

        Args:
            features (str): paragraph of person's speech to text data

        Returns:
            float: 1 - percentage of fillerwords in text
        """
        features = self.remove_punctuation(features)
        count , words = self.count_filler_words(features)
        return 1-(count/words)
    
    
    def remove_punctuation(self, text):
        # Translation table for removing punctuation
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def count_filler_words(self, text):
        # List of common filler words
        filler_words = {"um", "uh", "like", "you know", "so", "actually", "basically", "literally", "maybe", "perhaps", "probably",  "yeah", "okay"}
        
        # Split the text into words
        words = text.lower().split()
        print(words)
        # Initialize a counter for filler words
        filler_count = 0
        
        # Loop through each word in the text
        for word in words:
            if word in filler_words:
                filler_count += 1

        return filler_count, len(words)