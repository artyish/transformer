import tensorflow as tf

def get_vocab_dictionary():
    text = """
    Once upon a time, in a land full of trees, there was a little cherry tree. The cherry tree was very sad because it did not have any friends. All the other trees were big and strong, but the cherry tree was small and weak. The cherry tree was envious of the big trees.

    One day, the cherry tree felt a tickle in its branches. It was a little spring wind. The wind told the cherry tree not to be sad. The wind said, "You are special because you have sweet cherries that everyone loves." The cherry tree started to feel a little better.

    As time went on, the cherry tree grew more and more cherries. All the animals in the land came to eat the cherries and play under the cherry tree. The cherry tree was happy because it had many friends now. The cherry tree learned that being different can be a good thing. And they all lived happily ever after.
    """

    vocab_dictionary = {}
    vocab_dictionary.update({
    "<PAD>": 0,  
    "<UNK>": 1,
    "<EOS>": 2
    })

    input_num = 3
    for word in text.split():
        if word in vocab_dictionary:
            continue

        vocab_dictionary.update({word : input_num})
        input_num += 1
        
    return vocab_dictionary

def return_token(sentence, vocab_dictionary):
    tokenized_sentence = []
    for word in sentence.split():
        tokenized_sentence.append(vocab_dictionary.get(word, vocab_dictionary["<UNK>"]))
    tokenized_sentence.append(vocab_dictionary["<EOS>"])
    return tokenized_sentence

def return_embedding_vector(sentence):
    vocab_dictionary = get_vocab_dictionary()
    length = len(vocab_dictionary)
    
    tokens = return_token(sentence, vocab_dictionary)
    tokens_in_tensor = tf.constant([tokens])
    
    embedding_layer = tf.keras.layers.Embedding(input_dim=length, output_dim=32) 
    embedding_vectors = embedding_layer(tokens_in_tensor)
    
    print("Embedding shape:", embedding_vectors.shape)

    return embedding_vectors
