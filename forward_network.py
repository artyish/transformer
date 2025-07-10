import tensorflow as tf

def infer_feed_forward(weights):
    attention_weights = weights
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32)                       # d_ff → d_model
    ])
    
    ffn_output = ffn(attention_weights)
    
    output = tf.keras.layers.LayerNormalization()(ffn_output)
    
    return output
