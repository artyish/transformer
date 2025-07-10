import tensorflow as tf
from utilities.positional_encodings import vectors_with_positional
from attentionhead import infer_self_attention
from forward_network import infer_feed_forward
from utilities.tokenizer import get_vocab_dictionary, return_token
from datasets import load_dataset

dataset = load_dataset("squad")
small_train = dataset["train"].select(range(4000)) 

vocab = get_vocab_dictionary()
vocab_size = len(vocab)

text = ""
for example in small_train:
    text += example["question"] + " " + example["context"] + " "
    

SEQ_LEN = 20
input_sequences = []
target_sequences = []

for example in small_train:
    text = example["question"] + " " + example["context"]
    tokens = return_token(text, vocab)
    

    for i in range(len(tokens) - SEQ_LEN):
        input_sequences.append(tokens[i:i+SEQ_LEN])
        target_sequences.append(tokens[i+1:i+SEQ_LEN+1])


dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))
dataset = dataset.shuffle(100).batch(128).prefetch(tf.data.AUTOTUNE)

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

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

ckpt = tf.train.Checkpoint(
    embedding=embedding_layer,
    W_Q=W_Q, W_K=W_K, W_V=W_V,
    ffn=ffn_layer, norm=norm_layer,
    final_dense=final_logits_layer,
    optimizer=optimizer
)
manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

@tf.function
def train_step(inp, tgt):
    with tf.GradientTape() as tape:
        x = embedding_layer(inp)
        x = vectors_with_positional(x)
        x = infer_self_attention(x, W_Q, W_K, W_V)
        x = infer_feed_forward(x, ffn_layer, norm_layer)
        logits = final_logits_layer(x)
        loss = loss_fn(tgt, logits)
    variables = (
        embedding_layer.trainable_variables +
        W_Q.trainable_variables + W_K.trainable_variables + W_V.trainable_variables +
        ffn_layer.trainable_variables + norm_layer.trainable_variables +
        final_logits_layer.trainable_variables
    )
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss

for epoch in range(50):
    total_loss = 0.0
    for inp, tgt in dataset:
        loss = train_step(inp, tgt)
        total_loss += loss.numpy()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    manager.save()