import tensorflow as tf
from utilities.positional_encodings import vectors_with_positional
from attentionhead import infer_self_attention
from forward_network import infer_feed_forward
from utilities.tokenizer import get_vocab_dictionary
from utilities.tokenizer import return_token

tokens = "Hello brother"
def generate_text():
    vocab_dict = get_vocab_dictionary()
    tokenised_sentence = return_token(tokens, vocab_dict)
    max_tokens = 50
    
    id_to_word = {v: k for k, v in vocab_dict.items()}
    vocab_size = len(vocab_dict)
    
    for steps in range(max_tokens):
        
        embedded = vectors_with_positional(tokenised_sentence)
        output = infer_self_attention(embedded)
        logits = infer_feed_forward(output)
        last_token_logits = logits[:, -1, :]
        next_token_id = tf.argmax(last_token_logits, axis=-1).numpy()[0]

        if next_token_id == vocab_dict["<EOS>"]:
            break

        tokenised_sentence.append(next_token_id)
        
    return " ".join(id_to_word.get(tok, "<UNK>") for tok in tokenised_sentence)

value = generate_text()
print(value)