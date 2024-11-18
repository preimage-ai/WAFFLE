from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


class SentenceEmbeddings:
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.bag_of_words_embeddings = {}
    
    def set_bag_of_words(self, bag_of_words):
        """
        Sets the bag of words to be used for the sentence embeddings.
        """
        self.bag_of_words_embeddings = {
            word: self.get_embedding(word) for word in tqdm(bag_of_words)
        }
        
    def get_closest_word(self, sentence):
        """
        Returns the closest word in the bag of words to the given sentence.
        """
        sentence_embedding = self.get_embedding(sentence)
        closest_word = None
        closest_distance = float("inf")
        for word, word_embedding in self.bag_of_words_embeddings.items():
            distance = util.pytorch_cos_sim(sentence_embedding, word_embedding)
            if distance < closest_distance:
                closest_distance = distance
                closest_word = word
        return closest_word

    def get_embedding(self, sentence):
        """
        Returns the embedding of the given sentence.
        """
        return self.model.encode(sentence, convert_to_tensor=True)

    def get_similarity(self, sentence, sentence_to_compare_to):
        """
        Returns the similarity score between the given sentence and the sentence to compare to.
        """
        sentence_embeddings = self.get_embedding(sentence)
        sentence_to_compare_to_embedding = self.model.encode(
            sentence_to_compare_to, convert_to_tensor=True
        )
        cosine_score = util.pytorch_cos_sim(
            sentence_embeddings, sentence_to_compare_to_embedding
        )
        return cosine_score.tolist()[0][0]