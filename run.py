# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import tensorflow as tf
import json

import algorithms.stochasticrank as sr
import algorithms.PLRank as plr
import algorithms.tensorflowloss as tfl
import utils.dataset as dataset
import utils.nnmodel as nn
import utils.evaluate as evl

parser = argparse.ArgumentParser()
parser.add_argument("output_path", type=str,
                    help="Path to output model.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset.")
parser.add_argument("--dataset_info_path", type=str,
                    default="local_dataset_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--cutoff", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
parser.add_argument("--num_samples", required=True,
                    help="Number of samples for gradient estimation ('dynamic' applies the dynamic strategy).")
parser.add_argument("--num_eval_samples", type=int,
                    help="Number of samples for metric calculation in evaluation.",
                    default=10**2)
parser.add_argument("--loss", type=str, required=True,
                    help="Name of the loss to use (PL_rank_1/PL_rank_2/PL_rank_3/lambdaloss/policygradient/placementpolicygradient/stochasticrank_normal/stochasticrank_PL).")
parser.add_argument("--learning_rate", type=float,
                    help="The Learning Rate.",
                    default=0.001)

args = parser.parse_args()

cutoff = args.cutoff
num_samples = args.num_samples
num_eval_samples = args.num_eval_samples
learning_rate = args.learning_rate

if num_samples == 'dynamic':
  dynamic_samples = True
else:
  dynamic_samples = False
  num_samples = int(num_samples)

n_epochs = 400
maximum_train_time = 200*60

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                )
fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

max_ranking_size = np.min((cutoff, data.max_query_size()))

model_params = {
    'hidden units': [32,32],
    'learning_rate': learning_rate,
    'learning_rate_decay': 1.,
  }

model = nn.init_model(model_params)

epoch_results = []
timed_results = []

longest_possible_metric_weights = 1./np.log2(np.arange(data.max_query_size()) + 2)
metric_weights = longest_possible_metric_weights[:max_ranking_size]
train_labels = 2**data.train.label_vector-1
vali_labels = 2**data.validation.label_vector-1
test_labels = 2**data.test.label_vector-1
ideal_train_metrics = evl.ideal_metrics(data.train, metric_weights, train_labels)
ideal_vali_metrics = evl.ideal_metrics(data.validation, metric_weights, vali_labels)
ideal_test_metrics = evl.ideal_metrics(data.test, metric_weights, test_labels)


vali_result = evl.compute_results(data.validation,
                                  model, metric_weights,
                                  vali_labels, ideal_vali_metrics,
                                  num_eval_samples)
test_result = evl.compute_results(data.test,
                                  model, metric_weights,
                                  test_labels, ideal_test_metrics,
                                  num_eval_samples)
epoch_results.append({'steps': 0,
                'epoch': 0,
                'epoch check': 0,
                'train time': 0,
                'total time': 0,
                'validation result': vali_result,
                'test result': test_result,
                'num_samples': num_samples,
                'time check': 0})
timed_results.append(epoch_results[-1])

print('EPOCH: 0000.00 TIME: 0000'
      ' VALI: exp: %0.04f det: %0.04f'
      ' TEST: exp: %0.04f det: %0.04f' % (
      vali_result['normalized expectation'], vali_result['normalized maximum likelihood'],
      test_result['normalized expectation'], test_result['normalized maximum likelihood'],))

real_start_time = time.time()
total_train_time = 0
last_total_train_time = time.time()
method_train_time = 0

n_queries = data.train.num_queries()

if num_samples == 'dynamic':
  dynamic_samples = True
  float_num_samples = 10.
  add_per_step = 90./(n_queries*40.)
  max_num_samples = 1000

time_points = np.linspace(0, maximum_train_time, 200+1)
time_i = 1
n_times = time_points.shape[0]

steps = 0
batch_size = 256
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=model_params['learning_rate'],
    decay_steps=n_queries/batch_size,
    decay_rate=model_params['learning_rate_decay'])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

epoch_i = -1
while time_i < n_times:
  epoch_i += 1
  if dynamic_samples:
    num_samples = int(max(1, np.ceil(200*np.sqrt(epoch_i))))
  query_permutation = np.random.permutation(n_queries)
  for batch_i in range(int(np.ceil(n_queries/batch_size))):
    batch_queries = query_permutation[batch_i*batch_size:(batch_i+1)*batch_size]
    cur_batch_size = batch_queries.shape[0]
    batch_ranges = np.zeros(cur_batch_size+1, dtype=np.int64)
    batch_features = [data.train.query_feat(batch_queries[0])]
    batch_ranges[1] = batch_features[0].shape[0]
    for i in range(1, cur_batch_size):
      batch_features.append(data.train.query_feat(batch_queries[i]))
      batch_ranges[i+1] = batch_ranges[i] + batch_features[i].shape[0]
    batch_features = np.concatenate(batch_features, axis=0)

    with tf.GradientTape() as tape:
      batch_tf_scores = model(batch_features)
      loss = 0
      batch_doc_weights = np.zeros(batch_features.shape[0], dtype=np.float64)
      use_doc_weights = False
      for i, qid in enumerate(batch_queries):
        q_labels =  data.train.query_values_from_vector(
                                  qid, train_labels)
        q_feat = batch_features[batch_ranges[i]:batch_ranges[i+1],:]
        q_ideal_metric = ideal_train_metrics[qid]

        if q_ideal_metric != 0:
          q_metric_weights = metric_weights #/q_ideal_metric #uncomment for NDCG
          
          q_tf_scores = batch_tf_scores[batch_ranges[i]:batch_ranges[i+1]]

          last_method_train_time = time.time()
          if args.loss == 'policygradient':
            loss += tfl.policy_gradient(
                                      q_metric_weights,
                                      q_labels,
                                      q_tf_scores,
                                      n_samples=num_samples
                                      )
            method_train_time += time.time() - last_method_train_time
          elif args.loss == 'placementpolicygradient':
            loss += tfl.placement_policy_gradient(
                                      q_metric_weights,
                                      q_labels,
                                      q_tf_scores,
                                      n_samples=num_samples
                                      )
            method_train_time += time.time() - last_method_train_time
          else:
            q_np_scores = q_tf_scores.numpy()[:,0]
            if args.loss == 'lambdaloss':
              doc_weights = ll.lambdaloss(
                                        q_metric_weights,
                                        q_labels,
                                        q_np_scores,
                                        )
            elif args.loss == 'PL_rank_1':
              doc_weights = plr.PL_rank_1(
                                        q_metric_weights,
                                        q_labels,
                                        q_np_scores,
                                        n_samples=num_samples)
            elif args.loss == 'PL_rank_2':
              doc_weights = plr.PL_rank_2(
                                        q_metric_weights,
                                        q_labels,
                                        q_np_scores,
                                        n_samples=num_samples)
            elif args.loss == 'PL_rank_3':
              doc_weights = plr.PL_rank_3(
                                        q_metric_weights,
                                        q_labels,
                                        q_np_scores,
                                        n_samples=num_samples)
            elif args.loss == 'stochasticrank_normal':
              doc_weights = sr.normal_stochasticrank(
                                        q_metric_weights,
                                        q_labels,
                                        q_np_scores,
                                        n_samples=num_samples)
            elif args.loss == 'stochasticrank_PL':
              doc_weights = sr.gumbel_stochasticrank(
                                        q_metric_weights,
                                        q_labels,
                                        q_np_scores,
                                        n_samples=num_samples)
            else:
              raise NotImplementedError('Unknown loss %s' % args.loss)
            method_train_time += time.time() - last_method_train_time

            batch_doc_weights[batch_ranges[i]:batch_ranges[i+1]] = doc_weights
            use_doc_weights = True

      if use_doc_weights:
        loss = -tf.reduce_sum(batch_tf_scores[:,0] * batch_doc_weights)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    steps += cur_batch_size
    if dynamic_samples:
      float_num_samples = 10 + steps*add_per_step
      num_samples = min(int(np.round(float_num_samples)), max_num_samples)
    cur_epoch = steps/float(n_queries)
    next_total_train_time = total_train_time + (time.time() - last_total_train_time)
    if (next_total_train_time >= time_points[time_i]):
      total_train_time += time.time() - last_total_train_time
      vali_result = evl.compute_results(data.validation,
                                    model, metric_weights,
                                    vali_labels, ideal_vali_metrics,
                                    num_eval_samples)
      test_result = evl.compute_results(data.test,
                                    model, metric_weights,
                                    test_labels, ideal_test_metrics,
                                    num_eval_samples)

      print('EPOCH: %07.2f TIME: %04d'
            ' VALI: exp: %0.4f det: %0.04f'
            ' TEST: exp: %0.4f det: %0.04f' % (
            cur_epoch, total_train_time,
            vali_result['normalized expectation'], vali_result['normalized maximum likelihood'],
            test_result['normalized expectation'], test_result['normalized maximum likelihood'],))
      
      cur_result =  {'steps': steps,
                        'epoch': cur_epoch,
                        'train time': method_train_time,
                        'total time': total_train_time,
                        'validation result': vali_result,
                        'test result': test_result,
                        'num_samples': num_samples,
                        'time check': time_points[time_i]}

      while time_i < n_times and total_train_time >= time_points[time_i]:
        timed_results.append(cur_result)
        time_i += 1
        if time_i < n_times:
          cur_result =  {'steps': steps,
                          'epoch': cur_epoch,
                          'train time': method_train_time,
                          'total time': total_train_time,
                          'validation result': vali_result,
                          'test result': test_result,
                          'num_samples': num_samples,
                          'time check': time_points[time_i]}
      if time_i >= n_times:
        break
      last_total_train_time = time.time()

output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'run name': args.loss.replace('_', ' '),
  'loss': args.loss.replace('_', ' '),
  'model hyperparameters': model_params,
  'epoch results': epoch_results,
  'time results': timed_results,
  'number of samples': num_samples,
  'number of evaluation samples': num_eval_samples,
  'cutoff': cutoff,
}
if dynamic_samples:
  output['number of samples'] = 'dynamic'

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)
