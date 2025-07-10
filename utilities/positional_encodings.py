import numpy as np
import math
from utilities.tokenizer import return_embedding_vector

def get_embedding_vectors(sentence):
    embedding_vectors = return_embedding_vector(sentence)
    return embedding_vectors

def get_positional_encoding(embedding_vectors):
    final_positional_values = []    
    for i in range(embedding_vectors.shape[1]):
        values = []
        for f in range((embedding_vectors.shape[2] // 2)):

            value_even = np.sin(i/(np.power(10000,(2*f/embedding_vectors.shape[2]))))
            value_odd = np.cos(i/(np.power(10000,(2*f/embedding_vectors.shape[2]))))
            
            values.append(value_even)
            values.append(value_odd)
            
        final_positional_values.append(values)
        
    return final_positional_values

def vectors_with_positional(sentence):
    embedding_vectors = get_embedding_vectors(sentence)        
    final_positional_values = get_positional_encoding(embedding_vectors)            
    final_positional_values = np.array(final_positional_values)

    embedding_vectors += final_positional_values[np.newaxis, :, :]
    print(f"Embedding Vectors Shape: {embedding_vectors.shape}")
    return embedding_vectors

