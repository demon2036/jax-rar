from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.debug
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree


def cosine_similarity(x, y, axis=-1, eps=1e-8):
    """
    Computes cosine similarity between two arrays along a given axis.

    Args:
        x: A JAX array.
        y: A JAX array.
        axis: The axis along which to compute the similarity (default: -1).
        eps: A small constant to avoid division by zero.

    Returns:
        A JAX array of cosine similarity values.
    """
    # Normalize the input vectors
    x_norm = x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)
    y_norm = y / (jnp.linalg.norm(y, axis=axis, keepdims=True) + eps)

    # Compute the cosine similarity as the dot product of normalized vectors.
    return jnp.sum(x_norm * y_norm, axis=axis)

def get_cos_sim(image_features,labels):
    sim = cosine_similarity(image_features[:,None,:], image_features[None,...,])



    sim = (sim + 1) / 2
    label_mask = labels[:, None] != labels[None, :]
    mask_sin = sim * label_mask
    return mask_sin


class TrainModule(nn.Module):
    model: Any
    ref_model: Any
    beta:float =0.1

    def __call__(self, tokens: Array, labels: Array,siglip_feature:Array, dino_feature:Array,     det: bool = True,*args,**kwargs) -> ArrayTree:
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.
        """
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
        labels = nn.one_hot(labels, self.model.labels) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)
        """

        # input_ids: jnp.ndarray, condition: jnp.ndarray,
        # rngs: dict,  # 要求传入 {"dropout": dropout_key, "sample": sample_key}
        # return_labels: bool = False,
        # orders: Optional[jnp.ndarray] = None,
        # is_sampling: bool = True

        #
        # condition=self.model.preprocess_condition(labels,self.make_rng('dropout'),cond_drop_prob=0.1)
        condition = self.model.preprocess_condition(labels, self.make_rng('dropout'), cond_drop_prob=0.1)
        logits=self.model.train_dpo(tokens, condition)
        loss_ce=optax.softmax_cross_entropy_with_integer_labels(logits[:,:-1],tokens).mean()
        # return {'loss':loss.mean()}


        labels=self.model.preprocess_condition(labels,self.make_rng('dropout'),cond_drop_prob=0.0)




        mask_sin=get_cos_sim(dino_feature,labels)
        # print(mask_sin)

        pair_idx=jnp.argmax(mask_sin,axis=-1)
        print(pair_idx.shape)
        pair_label=labels[pair_idx]
        rejected_token=tokens[pair_idx]

        chosen_ids=tokens
        rejected_ids=rejected_token

        # jax.debug.print("{bar}  {pair_label} {labels}", bar=mask_sin,pair_label=pair_label ,labels=labels)
        # print(f'{rejected_token.shape=}')

        chosen_logits = self.model.train_dpo(tokens, labels)
        rejected_logits = self.model.train_dpo(rejected_token, labels)

        chosen_ref_logits = self.ref_model.train_dpo(tokens, labels)
        rejected_ref_logits = self.ref_model.train_dpo(rejected_token, labels)

        chosen_ref_logits=jax.lax.stop_gradient(chosen_ref_logits)
        rejected_ref_logits=jax.lax.stop_gradient(rejected_ref_logits)

        chosen_logps_seq = jnp.take_along_axis(  # [B, S]
            jax.nn.log_softmax(chosen_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
        )[..., 0]
        chosen_logps = jnp.sum(chosen_logps_seq , axis=-1)  # [B]
        chosen_ref_logps_seq = jnp.take_along_axis(  # [B, S]
            jax.nn.log_softmax(chosen_ref_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
        )[..., 0]
        chosen_ref_logps = jnp.sum(chosen_ref_logps_seq , axis=-1)  # [B]
        chosen_logratios = chosen_logps - chosen_ref_logps  # [B]

        rejected_logps_seq = jnp.take_along_axis(  # [B, S]
            jax.nn.log_softmax(rejected_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
        )[..., 0]
        rejected_logps = jnp.sum(rejected_logps_seq , axis=-1)  # [B]
        rejected_ref_logps_seq = jnp.take_along_axis(  # [B, S]
            jax.nn.log_softmax(rejected_ref_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
        )[..., 0]
        rejected_ref_logps = jnp.sum(rejected_ref_logps_seq , axis=-1)  # [B]
        rejected_logratios = rejected_logps - rejected_ref_logps  # [B]

        logratios_delta = self.beta * (chosen_logratios - rejected_logratios)  # [B]

        # print(f'{chosen_logps_seq.shape=}')

        loss=-jax.nn.log_sigmoid(self.beta * logratios_delta).mean()
        chosen_rewards=self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        reward_accuracies = (chosen_rewards > rejected_rewards).astype(jnp.float32)


        return {'loss':loss+loss_ce,
                'chosen_rewards':chosen_rewards.mean(),
                'rejected_rewards':rejected_rewards.mean(),
                'reward_accuracies':reward_accuracies.mean(),
                'loss_ce':loss_ce
                }
        # return self.model(tokens, det=det)

        # loss = self.criterion((logits := self.model(images, det=det)), labels)
        # labels = labels == labels.max(-1, keepdims=True)
        #
        # # Instead of directly comparing the maximum classes of predicted logits with the
        # # given one-hot labels, we will check if the predicted classes are within the
        # # label set. This approach is equivalent to traditional methods in single-label
        # # classification and also supports multi-label tasks.
        # preds = jax.lax.top_k(logits, k=5)[1]
        # accs = jnp.take_along_axis(labels, preds, axis=-1)
        # return {"loss": loss, "acc1": accs[:, 0], "acc5": accs.any(-1)}

