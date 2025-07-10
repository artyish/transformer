import numpy as np
from utilities.tokenizer import return_embedding_vector
import tensorflow as tf

def get_embedding_vectors(tokens):
    embedding_vectors = return_embedding_vector(tokens)
    return embedding_vectors

def get_positional_encoding(embedding_vectors):
    final_positional_values = []    
    for i in range(embedding_vectors.shape[1]):
        values = []
        print(embedding_vectors.shape)
        for f in range((embedding_vectors.shape[2] // 2)):

            value_even = np.sin(i/(np.power(10000,(2*f/embedding_vectors.shape[2]))))
            value_odd = np.cos(i/(np.power(10000,(2*f/embedding_vectors.shape[2]))))
            
            values.append(value_even)
            values.append(value_odd)
            
        final_positional_values.append(values)
        
    return tf.cast(tf.convert_to_tensor(final_positional_values), dtype=tf.float32)


def vectors_with_positional(embvec):
    embedding_vectors = embvec      
    final_positional_values = get_positional_encoding(embedding_vectors)            
    final_positional_values = tf.expand_dims(final_positional_values, 0)

    embedding_vectors += final_positional_values
    print(f"Embedding Vectors Shape: {embedding_vectors.shape}")
    return embedding_vectors

