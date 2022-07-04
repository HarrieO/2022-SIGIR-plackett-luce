# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl
import utils.ranking as rnk

def lambdaloss(rank_weights,
               labels,
               scores):
  n_docs = scores.shape[0]
  cutoff = min(rank_weights.shape[0], n_docs)

  (greater_i, lesser_i) = np.where(np.greater(
                            labels[:,None],
                            labels[None,:]))
  if greater_i.shape[0] == 0:
    return np.zeros(n_docs, dtype=np.float64)

  ranking, inv_ranking = rnk.cutoff_ranking(scores, cutoff, invert=True)
  delta_rank = np.abs(inv_ranking[greater_i]
                      - inv_ranking[lesser_i])
  if n_docs > cutoff:
    safe_rank_weights = np.zeros(n_docs)
    safe_rank_weights[:cutoff] = rank_weights
  else:
    safe_rank_weights = rank_weights

  delta_weight = (safe_rank_weights[delta_rank-1]
                   - safe_rank_weights[delta_rank])
  pair_weight = delta_weight * (labels[greater_i]
                                - labels[lesser_i])

  exp_score_diff = np.exp(
              scores[lesser_i]
              - scores[greater_i]
              )

  pair_deriv = pair_weight*exp_score_diff/(
               (exp_score_diff + 1.)*np.log(2.))

  doc_weights = np.zeros(n_docs, dtype=np.float64)
  np.add.at(doc_weights, greater_i, pair_deriv)
  np.add.at(doc_weights, lesser_i, -pair_deriv)

  return doc_weights