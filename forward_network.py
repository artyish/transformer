import tensorflow as tf

def infer_feed_forward(weights, ffn_layer, norm_layer):
    attention_weights = weights
    ffn_output = ffn_layer(weights)
    return norm_layer(ffn_output)
