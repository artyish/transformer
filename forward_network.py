from attentionhead import infer_self_attention

def infer_feed_forward(sentence):
    attention_weights = infer_self_attention(sentence)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),  # d_model → d_ff
        tf.keras.layers.Dense(32)                       # d_ff → d_model
    ])
