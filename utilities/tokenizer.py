import tensorflow as tf
from datasets import load_dataset

def get_vocab_dictionary():
    dataset = load_dataset("squad")
    small_train = dataset["train"].select(range(4000))

    vocab_dictionary = {}
    vocab_dictionary.update({
    "<PAD>": 0,  
    "<UNK>": 1,
    "<EOS>": 2
    })

    input_num = 3
    for example in small_train:
        combined_text = example["question"] + " " + example["context"]
        for word in combined_text.lower().split():  
            if word not in vocab_dictionary:
                vocab_dictionary.update({word: input_num})
                input_num += 1
        
    return vocab_dictionary

def return_token(sentence, vocab_dictionary):
    tokenized_sentence = []
    for word in sentence.split():
        tokenized_sentence.append(vocab_dictionary.get(word, vocab_dictionary["<UNK>"]))
    tokenized_sentence.append(vocab_dictionary["<EOS>"])
    return tokenized_sentence

def return_embedding_vector(tensor):
    vocab_dictionary = get_vocab_dictionary()
    length = len(vocab_dictionary)
    
    #tokens = tokens
    tokens_in_tensor = tensor
    
    embedding_layer = tf.keras.layers.Embedding(input_dim=length, output_dim=32) 
    embedding_vectors = embedding_layer(tokens_in_tensor)
    
    print("Embedding shape:", embedding_vectors.shape)

    return embedding_vectors
