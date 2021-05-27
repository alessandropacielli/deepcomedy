import tensorflow as tf
import numpy as np
import tensorflow as tf

from .transformer import Encoder

from strsimpy.normalized_levenshtein import NormalizedLevenshtein

import time

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