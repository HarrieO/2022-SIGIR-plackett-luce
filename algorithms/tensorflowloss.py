# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import tensorflow as tf
import numpy as np
import utils.plackettluce as pl

def placement_policy_gradient(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  np_scores = scores.numpy()[:,0]
  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    np_scores,
                                    n_samples,
                                    cutoff=cutoff,
                                    return_full_rankings=True)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  sampled_scores = tf.gather(scores, sampled_rankings)[:,:,0]
  denom = tf.math.cumulative_logsumexp(sampled_scores, axis=1, reverse=True)
  sample_log_prob = sampled_scores[:,:cutoff]-denom[:,:cutoff]

  rewards = rank_weights[None,:cutoff]*labels[sampled_rankings[:,:cutoff]]
  cum_rewards = tf.cumsum(rewards, axis=1, reverse=True)
  
  result = tf.reduce_sum(tf.reduce_mean(
                  sample_log_prob*cum_rewards, axis=0))
  return -result

def policy_gradient(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  np_scores = scores.numpy()[:,0]
  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    np_scores,
                                    n_samples,
                                    cutoff=cutoff,
                                    return_full_rankings=True)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  sampled_scores = tf.gather(scores, sampled_rankings)[:,:,0]
  denom = tf.math.cumulative_logsumexp(sampled_scores, axis=1, reverse=True)
  sample_log_prob = sampled_scores[:,:cutoff]-denom[:,:cutoff]
  final_prob_loss = tf.reduce_sum(sample_log_prob, axis=1)

  rewards = np.sum(rank_weights[None,:cutoff]
                   *labels[sampled_rankings[:,:cutoff]], axis=1)
  result = tf.reduce_mean(
                  final_prob_loss*rewards, axis=0)
  return -result