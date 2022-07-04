# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl

def gumbel_stochasticrank(rank_weights, labels, scores, n_samples=None, sampled_rankings=None, sampled_inv_rankings=None, sampled_values=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs-1)

  if n_docs == 1:
    return np.zeros_like(scores)

  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    (sampled_rankings,
     _,_,_,
     sampled_values) = pl.gumbel_sample_rankings(
                                    scores,
                                    n_samples,
                                    cutoff=cutoff+1,
                                    return_gumbel=True)
  else:
    n_samples = sampled_rankings.shape[0]

  delta_weights = rank_weights[:cutoff].copy()
  if rank_weights.shape[0] < n_docs:
    delta_weights[:-1] -= rank_weights[1:cutoff]
  else:
    delta_weights -= rank_weights[1:cutoff+1]

  ranking_values = sampled_values[np.arange(n_samples)[:,None], sampled_rankings]
  ranking_scores = scores[sampled_rankings[:,:cutoff]]

  pairwise_diff = ranking_values[:,:cutoff,None] - scores[None,None,:]
  pairwise_pdf = np.exp(-pairwise_diff-np.exp(-pairwise_diff))

  sampled_labels = labels[sampled_rankings]
  delta_pairwise = delta_weights[None,:,None]*(labels[None,None,:] - sampled_labels[:,:cutoff,None])

  sampled_grad = pairwise_pdf*delta_pairwise
  sampled_mask = np.greater_equal(sampled_values[:,None,:], ranking_values[:,:cutoff,None])
  sampled_grad[sampled_mask] = 0
  sampled_grad = np.sum(sampled_grad, axis=1)

  pairwise_diff = ranking_values[:,1:,None] - ranking_scores[:,None,:]
  pairwise_pdf = np.exp(-pairwise_diff-np.exp(-pairwise_diff))
  delta_moved = delta_weights[None,:,None]*(sampled_labels[:,None,:cutoff] - sampled_labels[:,1:,None])

  k_range = np.arange(cutoff)
  k_mask = np.less_equal(k_range[:,None],k_range[None,:])[None,:,:]
  grad_cor = pairwise_pdf*np.where(k_mask, delta_moved, 0)
  grad_cor = np.sum(grad_cor, axis=1)

  np.add.at(sampled_grad, (np.arange(n_samples)[:,None], sampled_rankings[:,:cutoff]), grad_cor)

  return np.mean(sampled_grad, axis=0)

def normal_pdf(x):
  return np.exp(-x**2./(2.))/(2.*np.pi*1.)**.5

def normal_stochasticrank(rank_weights, labels, scores, n_samples=None, sampled_rankings=None, sampled_inv_rankings=None, sampled_values=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs-1)

  if n_docs == 1:
    return np.zeros_like(scores)

  # scores = scores/np.linalg.norm(scores)

  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    (sampled_rankings, _,
     sampled_values) = pl.normal_sample_rankings(
                                    scores,
                                    n_samples,
                                    cutoff=cutoff+1,
                                    return_gumbel=True)
  else:
    n_samples = sampled_rankings.shape[0]

  delta_weights = rank_weights[:cutoff].copy()
  if rank_weights.shape[0] < n_docs:
    delta_weights[:-1] -= rank_weights[1:cutoff]
  else:
    delta_weights -= rank_weights[1:cutoff+1]

  ranking_values = sampled_values[np.arange(n_samples)[:,None], sampled_rankings]
  ranking_scores = scores[sampled_rankings[:,:cutoff]]

  pairwise_diff = ranking_values[:,:cutoff,None] - scores[None,None,:]
  pairwise_pdf = normal_pdf(pairwise_diff)

  sampled_labels = labels[sampled_rankings]
  delta_pairwise = delta_weights[None,:,None]*(labels[None,None,:] - sampled_labels[:,:cutoff,None])

  sampled_grad = pairwise_pdf*delta_pairwise
  sampled_mask = np.greater_equal(sampled_values[:,None,:], ranking_values[:,:cutoff,None])
  sampled_grad[sampled_mask] = 0
  sampled_grad = np.sum(sampled_grad, axis=1)

  pairwise_diff = ranking_values[:,1:,None] - ranking_scores[:,None,:]
  pairwise_pdf = normal_pdf(pairwise_diff)
  delta_moved = delta_weights[None,:,None]*(sampled_labels[:,None,:cutoff] - sampled_labels[:,1:,None])

  k_range = np.arange(cutoff)
  k_mask = np.less_equal(k_range[:,None],k_range[None,:])[None,:,:]
  grad_cor = pairwise_pdf*np.where(k_mask, delta_moved, 0)
  grad_cor = np.sum(grad_cor, axis=1)

  np.add.at(sampled_grad, (np.arange(n_samples)[:,None], sampled_rankings[:,:cutoff]), grad_cor)

  return np.mean(sampled_grad, axis=0)
