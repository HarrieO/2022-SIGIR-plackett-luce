# Learning-to-Rank at the Speed of Sampling: Plackett-Luce Gradient Estimation With Minimal Computational Complexity
This repository contains the code used for the experiments in "Learning-to-Rank at the Speed of Sampling: Plackett-Luce Gradient Estimation With Minimal Computational Complexity" published at SIGIR 2022 ([available here](https://harrieo.github.io//publication/2022-sigir-short)).

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our SIGIR 2022 paper:
```
@inproceedings{oosterhuis2022plrank,
  Author = {Oosterhuis, Harrie},
  Booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`22)},
  Organization = {ACM},
  Title = {Learning-to-Rank at the Speed of Sampling: Plackett-Luce Gradient Estimation With Minimal Computational Complexity},
  Year = {2022}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Usage
-------

This code makes use of [Python 3](https://www.python.org/), the [numpy](https://numpy.org/) and the [tensorflow](https://www.tensorflow.org/) packages, make sure they are installed.

A file is required that explains the location and details of the LTR datasets available on the system, for the Yahoo! Webscope, MSLR-Web30k, and Istella datasets an example file is available. Copy the file:
```
cp example_datasets_info.txt local_dataset_info.txt
```
Open this copy and edit the paths to the folders where the train/test/vali files are placed. (Note that the Istella dataset does not have a validation set by default, I recommend partitioning 10% from the training data to creat a validation set.)

Here are some command-line examples that illustrate how the results in the paper can be replicated.
First create a folder to store the resulting models:
```
mkdir local_output
```
The experiments are all based on *run.py* with the *--loss* flag to indicate the loss to use: PL_rank_2/PL_rank_3/stochasticrank_PL (the losses from the [SIGIR`21 PL-rank paper](https://harrieo.github.io//publication/2021-plrank) are also implemented); *--cutoff* indicates the top-k that is being optimized, e.g. 5 for DCG@5; *--num_samples* the number of samples to use per gradient estimation (with *dynamic* for a dynamic strategy); *--dataset* indicates the dataset name, e.g. *Webscope_C14_Set1*.
The following command optimizes DCG@5 with PL-Rank-2 and with 100 samples on the Yahoo! dataset:
```
python3 run.py local_output/yahoo_ndcg5_dynamic_plrank2.txt --num_samples 100 --loss PL_rank_3 --cutoff 5 --dataset Webscope_C14_Set1
```