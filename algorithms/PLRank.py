# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl
import utils.ranking as rnk

def PL_rank_1(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  if n_docs == 1:
    return np.zeros_like(scores)

  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    scores,
                                    n_samples,
                                    cutoff=cutoff)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  srange = np.arange(n_samples)
  crange = np.arange(cutoff)

  weighted_labels = labels[sampled_rankings]*rank_weights[None,:cutoff]
  cumsum_labels = np.cumsum(weighted_labels[:,::-1], axis=1)[:,::-1]

  np.add.at(result, sampled_rankings, cumsum_labels)
  result /= n_samples

  placed_mask = np.zeros((n_samples, cutoff-1, n_docs), dtype=np.bool)
  placed_mask[srange[:,None],
              crange[None,:-1],
              sampled_rankings[:,:-1]] = True
  placed_mask[:,:] = np.cumsum(placed_mask, axis=1)

  total_denom = np.logaddexp.reduce(scores)
  minus_denom = np.logaddexp.accumulate(scores[sampled_rankings[:,:-1]], axis=1)
  denom_per_rank = np.log(1. - np.exp(minus_denom-total_denom)) + total_denom
  prob_per_rank = np.empty((n_samples, cutoff, n_docs), dtype=np.float64)
  prob_per_rank[:,0,:] = np.exp(scores[None,:] - total_denom)
  prob_per_rank[:,1:,:] = np.exp(scores[None,None,:] - denom_per_rank[:,:,None])
  prob_per_rank[:,1:,:][placed_mask] = 0.

  minus_weights = np.mean(
    np.sum(prob_per_rank*cumsum_labels[:,:,None], axis=1)
    , axis=0, dtype=np.float64)

  result -= minus_weights

  return result


def PL_rank_2(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  if n_docs == 1:
    return np.zeros_like(scores)

  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    scores,
                                    n_samples,
                                    cutoff=cutoff)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  srange = np.arange(n_samples)
  crange = np.arange(cutoff)

  relevant_docs = np.where(np.not_equal(labels, 0))[0]
  n_relevant_docs = relevant_docs.size

  weighted_labels = labels[sampled_rankings]*rank_weights[None,:cutoff]
  cumsum_labels = np.cumsum(weighted_labels[:,::-1], axis=1)[:,::-1]

  np.add.at(result, sampled_rankings[:,:-1], cumsum_labels[:,1:])
  result /= n_samples

  placed_mask = np.zeros((n_samples, cutoff-1, n_docs), dtype=np.bool)
  placed_mask[srange[:,None],
              crange[None,:-1],
              sampled_rankings[:,:-1]] = True
  placed_mask[:,:] = np.cumsum(placed_mask, axis=1)

  total_denom = np.logaddexp.reduce(scores)
  minus_denom = np.logaddexp.accumulate(scores[sampled_rankings[:,:-1]], axis=1)
  denom_per_rank = np.log(np.maximum(1. - np.exp(minus_denom-total_denom), 10**-8)) + total_denom
  prob_per_rank = np.empty((n_samples, cutoff, n_docs), dtype=np.float64)
  prob_per_rank[:,0,:] = np.exp(scores[None,:] - total_denom)
  prob_per_rank[:,1:,:] = np.exp(scores[None,None,:] - denom_per_rank[:,:,None])
  prob_per_rank[:,1:,:][placed_mask] = 0.

  result -= np.mean(
    np.sum(prob_per_rank*cumsum_labels[:,:,None], axis=1)
    , axis=0, dtype=np.float64)
  result[relevant_docs] += np.mean(
    np.sum(prob_per_rank[:,:,relevant_docs]*(
                          rank_weights[None,:cutoff,None]
                          *labels[None,None,relevant_docs]), axis=1)
    , axis=0, dtype=np.float64)

  return result

def PL_rank_3(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  if n_docs == 1:
    return np.zeros_like(scores)

  scores = scores.copy() - np.amax(scores) + 10.

  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    scores,
                                    n_samples,
                                    cutoff=cutoff,
                                    return_full_rankings=True)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  cutoff_sampled_rankings = sampled_rankings[:,:cutoff]

  srange = np.arange(n_samples)

  relevant_docs = np.where(np.not_equal(labels, 0))[0]
  n_relevant_docs = relevant_docs.size

  weighted_labels = labels[cutoff_sampled_rankings]*rank_weights[None,:cutoff]
  cumsum_labels = np.cumsum(weighted_labels[:,::-1], axis=1)[:,::-1]

  np.add.at(result, cutoff_sampled_rankings[:,:-1], cumsum_labels[:,1:])
  result /= n_samples

  exp_scores = np.exp(scores).astype(np.float64)
  denom_per_rank = np.cumsum(exp_scores[sampled_rankings[:,::-1]], axis=1)[:,:-cutoff-1:-1]

  cumsum_weight_denom = np.cumsum(rank_weights[:cutoff]/denom_per_rank, axis=1)
  cumsum_reward_denom = np.cumsum(cumsum_labels/denom_per_rank, axis=1)  

  if cutoff < n_docs:
    second_part = -exp_scores[None,:]*cumsum_reward_denom[:,-1,None]
    second_part[:,relevant_docs] += (labels[relevant_docs][None,:]
        *exp_scores[None,relevant_docs]*cumsum_weight_denom[:,-1,None])
  else:
    second_part = np.empty((n_samples, n_docs), dtype=np.float64)

  sampled_direct_reward = labels[cutoff_sampled_rankings]*exp_scores[cutoff_sampled_rankings]*cumsum_weight_denom
  sampled_following_reward = exp_scores[cutoff_sampled_rankings]*cumsum_reward_denom
  second_part[srange[:,None], cutoff_sampled_rankings] = sampled_direct_reward - sampled_following_reward

  return result + np.mean(second_part, axis=0)