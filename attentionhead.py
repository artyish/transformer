from utilities.positional_encodings import vectors_with_positional
import tensorflow as tf
import math

# here we infer self attention :D
def infer_self_attention(sentence):
    number_of_heads = 4
    get_vectors = vectors_with_positional(sentence)
    print(get_vectors.shape)

    # use that to calculate the dimension for weight matrices
    dimension = get_vectors.shape[2]
    weight_shape_y = dimension // number_of_heads # this is for single head attention
    weight_shape_x = get_vectors.shape[2]

    W_Q = tf.keras.layers.Dense(units=dimension, use_bias=False) # dimension used for 4 head attention
    W_K = tf.keras.layers.Dense(units=dimension, use_bias=False)
    W_V = tf.keras.layers.Dense(units=dimension, use_bias=False)

    Q = W_Q(get_vectors)
    K = W_K(get_vectors)
    V = W_V(get_vectors)


    Q = tf.reshape(Q, (1, 5, 4, 8))
    Q = tf.transpose(Q, perm=[0, 2, 1, 3])

    K = tf.reshape(K, (1, 5, 4, 8))
    K = tf.transpose(K, perm=[0, 2, 1, 3])

    V = tf.reshape(V, (1, 5, 4, 8))
    V = tf.transpose(V, perm=[0, 2, 1, 3])


    scores = tf.matmul(Q, K, transpose_b=True)
    scores = scores / tf.math.sqrt(tf.cast(weight_shape_y, tf.float32))
    attention_weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(attention_weights, V)

    output = tf.transpose(output, perm=[0, 2, 1, 3])  
    output = tf.reshape(output, (1, 5, 32))
    
    return output