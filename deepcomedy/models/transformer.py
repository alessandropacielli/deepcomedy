import numpy as np
import tensorflow as tf
import wandb

from .layers import *

from strsimpy.normalized_levenshtein import NormalizedLevenshtein

import time


################################################################# Masking ##########################################################################
def create_padding_mask(seq):
    """
    The padding mask is used in the Encoder and Decoder layers to mask padding tokens.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    The look-ahead mask is used by the Decoder only.
    It is used to mask future positions in the Decoder input so that the self-attention layer
    only takes into account previous tokens.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    """
    Creates a padding mask and a look-ahead mask, then combines them taking the maximum of the two.
    """

    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


################################################################# Positional encoding ################################################################
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    Positional encoding is used to take into account the order of tokens in a sentence.
    The output of this function will be added to the embeddings.
    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


####################################################################### Metrics ######################################################################
def padded_loss(real, pred, loss_function):
    """
    Returns average loss excluding padded sequence elements.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_function(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def padded_accuracy(real, pred):
    """
    Returns average accuracy excluding padded sequence elements.
    """
    accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


################################################################# Transformer ################################################################
class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        rate=0.1,
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate
        )

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask
    ):

        enc_output = self.encoder(
            inp, training, enc_padding_mask
        )  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class TransformerTrainer(object):
    def __init__(
        self,
        transformer,
        optimizer=None,
        loss_function=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        ),
        checkpoint_every=5,
        checkpoint_save_path=None,
    ):
        super(TransformerTrainer, self).__init__()

        self.transformer = transformer

        self.loss_function = loss_function

        if optimizer is None:
            learning_rate = TransformerCustomSchedule(self.transformer.d_model)
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
            )
        else:
            self.optimizer = optimizer

        self.checkpoint_every = checkpoint_every

        if checkpoint_save_path is not None:

            ckpt = tf.train.Checkpoint(
                transformer=self.transformer, optimizer=self.optimizer
            )

            self.checkpoint_manager = tf.train.CheckpointManager(
                ckpt, checkpoint_save_path, max_to_keep=5
            )

            # if a checkpoint exists, restore the latest checkpoint.
            if self.checkpoint_manager.latest_checkpoint:
                ckpt.restore(self.checkpoint_manager.latest_checkpoint)
                print("Latest checkpoint restored!!")
        else:
            self.checkpoint_manager = None

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.Mean(name="val_accuracy")

    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(
                inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = padded_loss(tar_real, predictions, self.loss_function)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.transformer.trainable_variables)
        )

        self.train_loss(loss)
        self.train_accuracy(padded_accuracy(tar_real, predictions))

    def val_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        predictions, _ = self.transformer(
            inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask
        )
        loss = padded_loss(tar_real, predictions, self.loss_function)

        self.val_loss(loss)
        self.val_accuracy(padded_accuracy(tar_real, predictions))

    def train(
        self,
        dataset,
        epochs,
        log_wandb=False,
        validation_dataset=None,
        validation_every=5,
    ):
        """
        As validation score you can pass the return value of `make_syll_score`
        """

        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            # Train step
            for (batch, (inp, tar)) in enumerate(dataset):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print(
                        f"Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
                    )
            if log_wandb:
                wandb.log(
                    {
                        "train_loss": self.train_loss.result(),
                        "train_accuracy": self.train_accuracy.result(),
                    },
                    step=epoch + 1,
                )

            # Validation step only if validation dataset is not null and
            if validation_dataset and ((epoch + 1) % validation_every == 0):
                for (batch, (inp, tar)) in enumerate(validation_dataset):
                    self.val_step(inp, tar)

                    if batch % 50 == 0:
                        print(
                            f"Epoch {epoch + 1} Batch {batch} Validation Loss {self.val_loss.result():.4f} Validation Accuracy {self.val_accuracy.result():.4f}"
                        )

                if log_wandb:
                    wandb.log(
                        {
                            "val_loss": self.val_loss.result(),
                            "val_accuracy": self.val_accuracy.result(),
                        },
                        step=epoch + 1,
                    )

                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

            if (
                self.checkpoint_manager is not None
                and (epoch + 1) % self.checkpoint_every == 0
            ):
                ckpt_save_path = self.checkpoint_manager.save()
                print(f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}")

            print(
                f"Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
            )

            print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")


def choose_greedy(logits):
    # select the last character from the seq_len dimension
    predicted_ids = tf.argmax(logits[:, -1:, :], axis=-1)
    return predicted_ids


def choose_topk(logits, k=5, temperature=0.5):

    # TODO not batched

    predictions = logits[:, -1:, :]  # (batch_size, 1, vocab_size)
    predictions, indices = tf.math.top_k(predictions, k=k)

    predictions /= temperature
    predictions = predictions[0]
    indices = indices[0][0]
    predicted_ids = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
    predicted_ids = indices[predicted_ids]

    predicted_ids = tf.expand_dims(predicted_ids, 0)
    predicted_ids = tf.expand_dims(predicted_ids, 0)

    return predicted_ids


def stop_after_quatrain(output, stop_symbol):
    # TODO not batched!! Only checks first output, ok for generation
    n_verses = sum(output.numpy()[0] == stop_symbol)
    return n_verses == 4


def stop_after_stop_symbol(output, stop_symbol):
    """
    Checks that all generated verses in output (batch_size, vocab_size) have at least one stop symbol
    """
    return all(list(map(lambda x: stop_symbol in x, output)))


def evaluate(
    transformer,
    encoder_input,
    decoder_input,
    stop_symbol,
    max_length=200,
    choose_next_token=choose_greedy,
    stopping_condition=stop_after_quatrain,
):
    """
    Predicts the output of the model given the input_sequence.
    The input_sequence is encoded by the Encoder, then its output is fed to the Decoder,
    whose output is fed back into the Decoder until the stop_symbol token is produced.

    This function works with a batch of inputs and stops when all outputs include a stop symbol.
    """

    output = decoder_input

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output
    )

    enc_output = transformer.encoder(
        encoder_input, False, enc_padding_mask
    )  # (batch_size, inp_seq_len, d_model)

    for _ in range(max_length):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output
        )

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, _ = transformer.decoder(
            output, enc_output, False, combined_mask, dec_padding_mask
        )

        predictions = transformer.final_layer(dec_output)

        predicted_ids = choose_next_token(predictions)

        # concatenate the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat(
            [
                tf.cast(output, dtype=tf.int32),
                tf.cast(predicted_ids, dtype=tf.int32),
            ],
            axis=-1,
        )

        if stopping_condition(output, stop_symbol):
            return output

    return output


def make_transformer_model(
    config, input_vocab_size, target_vocab_size, checkpoint_save_path=None
):
    """
    Creates a transformer model and its trainer from wandb config dictionary
    """
    transformer = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=config["dff"],
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=1000,
        pe_target=1000,
        rate=0.1,
    )

    transformer_trainer = TransformerTrainer(
        transformer, checkpoint_save_path=checkpoint_save_path
    )

    return transformer, transformer_trainer


def save_transformer_model(transformer, path):
    transformer.save_weights(path)


def load_transformer_model(
    config, input_vocab_size, target_vocab_size, target_tokenizer, path
):
    transformer = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=config["dff"],
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=1000,
        pe_target=1000,
        rate=0.1,
    )
    # In order to load the new weights the model should be called once for the variables to be initialized

    # Any inp, tar is ok here
    start_symbol = target_tokenizer.word_index["<GO>"]

    inp = tf.convert_to_tensor([[start_symbol]])
    tar = tf.convert_to_tensor([[start_symbol]])

    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)

    transformer(inp, tar, False, enc_padding_mask, look_ahead_mask, dec_padding_mask)

    transformer.load_weights(path)

    return transformer


###################################################################### Custom Schedule ###################################################################
class TransformerCustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerCustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
