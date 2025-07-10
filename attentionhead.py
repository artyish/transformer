import tensorflow as tf
import math

# here we infer self attention :D
def infer_self_attention(vectors, W_Q, W_K, W_V):
    number_of_heads = 4
    get_vectors = vectors
    print(get_vectors.shape)

    # use that to calculate the dimension for weight matrices
    dimension = get_vectors.shape[2]
    weight_shape_y = dimension // number_of_heads # this is for single head attention

    Q = W_Q(get_vectors)
    K = W_K(get_vectors)
    V = W_V(get_vectors)

    batch_size, seq_len, d_model = Q.shape


    Q = tf.reshape(Q, (batch_size, seq_len, number_of_heads, weight_shape_y))
    Q = tf.transpose(Q, perm=[0, 2, 1, 3])

    K = tf.reshape(K, (batch_size, seq_len, number_of_heads, weight_shape_y))
    K = tf.transpose(K, perm=[0, 2, 1, 3])

    V = tf.reshape(V, (batch_size, seq_len, number_of_heads, weight_shape_y))
    V = tf.transpose(V, perm=[0, 2, 1, 3])


    scores = tf.matmul(Q, K, transpose_b=True)
    scores = scores / tf.math.sqrt(tf.cast(weight_shape_y, tf.float32))
    attention_weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(attention_weights, V)

    output = tf.transpose(output, perm=[0, 2, 1, 3])  
    output = tf.reshape(output, (batch_size, seq_len, dimension))
    
    return output