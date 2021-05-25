import numpy as np
import tensorflow as tf

from .layers import *

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


@tf.function
def train_step(
    transformer, inp, tar, loss_function, optimizer, train_loss, train_accuracy
):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
        )
        loss = padded_loss(tar_real, predictions, loss_function)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(padded_accuracy(tar_real, predictions))


def train_transformer(
    transformer,
    dataset,
    epochs,
    optimizer=None,
    loss_function=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    ),
    checkpoint_every=5,
    checkpoint_save_path="./checkpoints/train",
):

    if optimizer is None:
        learning_rate = TransformerCustomSchedule(transformer.d_model)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
    else:
        optimizer = optimizer

    if checkpoint_save_path is not None:

        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

        checkpoint_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_save_path, max_to_keep=5
        )

        # if a checkpoint exists, restore the latest checkpoint.
        if checkpoint_manager.latest_checkpoint:
            ckpt.restore(checkpoint_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(dataset):
            train_step(
                transformer,
                inp,
                tar,
                loss_function,
                optimizer,
                train_loss,
                train_accuracy,
            )

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
                )

        if checkpoint_manager is not None and (epoch + 1) % checkpoint_every == 0:
            ckpt_save_path = checkpoint_manager.save()
            print(f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}")

        print(
            f"Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
        )

        print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")


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


####################################################################### Decoder-only model ###############################################################
class DecoderOnlyModel(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        pe,
        rate=0.1,
    ):
        super(DecoderOnlyModel, self).__init__()

        self.d_model = d_model

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, vocab_size, pe, rate
        )

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, training, look_ahead_mask):

        enc_output = self.encoder(
            inp,
            training,
            look_ahead_mask,
        )  # (batch_size, inp_seq_len, d_model)

        final_output = self.final_layer(
            enc_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output


class DecoderOnlyTrainer(object):
    def __init__(
        self,
        decoder,
        optimizer=None,
        loss_function=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        ),
        checkpoint_every=5,
        checkpoint_save_path="./checkpoints/train",
    ):
        super(DecoderOnlyTrainer, self).__init__()

        self.decoder = decoder

        self.loss_function = loss_function

        if optimizer is None:
            learning_rate = TransformerCustomSchedule(self.decoder.d_model)
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
            )
        else:
            self.optimizer = optimizer

        self.checkpoint_every = checkpoint_every
        if checkpoint_save_path is not None:

            ckpt = tf.train.Checkpoint(
                transformer=self.decoder, optimizer=self.optimizer
            )

            self.checkpoint_manager = tf.train.CheckpointManager(
                ckpt, checkpoint_save_path, max_to_keep=5
            )

            # if a checkpoint exists, restore the latest checkpoint.
            if self.checkpoint_manager.latest_checkpoint:
                ckpt.restore(self.checkpoint_manager.latest_checkpoint)
                print("Latest checkpoint restored!!")

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    @tf.function
    def train_step(self, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        _, combined_mask, _ = create_masks(tar, tar_inp)

        with tf.GradientTape() as tape:
            predictions = self.decoder(tar_inp, True, combined_mask)
            loss = padded_loss(tar_real, predictions, self.loss_function)

        gradients = tape.gradient(loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(padded_accuracy(tar_real, predictions))

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, inp) in enumerate(dataset):
                self.train_step(inp)

                if batch % 50 == 0:
                    print(
                        f"Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}"
                    )

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