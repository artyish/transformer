import tensorflow as tf
from utilities.positional_encodings import vectors_with_positional
from attentionhead import infer_self_attention
from forward_network import infer_feed_forward
from utilities.tokenizer import get_vocab_dictionary
from utilities.tokenizer import return_token

tokens = "Where does the sun rise?"
def generate_text():
    vocab_dict = get_vocab_dictionary()
    tokenised_sentence = return_token(tokens, vocab_dict)
    max_tokens = 50
    
    id_to_word = {v: k for k, v in vocab_dict.items()}
    vocab_size = len(vocab_dict)
    
    input_tensor = tf.constant([tokenised_sentence])
    
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32)
    W_Q = tf.keras.layers.Dense(32, use_bias=False)
    W_K = tf.keras.layers.Dense(32, use_bias=False)
    W_V = tf.keras.layers.Dense(32, use_bias=False)

    ffn_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32)
    ])
    norm_layer = tf.keras.layers.LayerNormalization()
    final_logits_layer = tf.keras.layers.Dense(vocab_size)

    # === Restore from saved checkpoint ===
    ckpt = tf.train.Checkpoint(
        embedding=embedding_layer,
        W_Q=W_Q, W_K=W_K, W_V=W_V,
        ffn=ffn_layer, norm=norm_layer,
        final_dense=final_logits_layer
    )
    manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    
    if manager.latest_checkpoint:
        print(f"✅ Restored from {manager.latest_checkpoint}")
    else:
        print("⚠️ No checkpoint found, weights are random")
        
        
    
    for steps in range(max_tokens):
        embedded = embedding_layer(input_tensor)
        
        embedded = vectors_with_positional(embedded)
        
        output = infer_self_attention(embedded, W_Q, W_K, W_V)
        
        logits = infer_feed_forward(output, ffn_layer, norm_layer)
        
        logits = final_logits_layer(logits)

        
        next_token_logits = logits[:, -1, :]
        next_token = tf.argmax(next_token_logits, axis=-1).numpy()[0]

        if next_token == vocab_dict["<EOS>"]:
            break

        tokenised_sentence.append(next_token)
        input_tensor = tf.constant([tokenised_sentence])

    return " ".join(id_to_word.get(tok, "<UNK>") for tok in tokenised_sentence)

value = generate_text()
print(value)