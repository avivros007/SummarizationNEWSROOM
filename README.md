# Attempting to Improve Pointer-Generator Networks for Summariztion

---

## About this code
The code is based on the [pointer-generator code](https://github.com/abisee/pointer-generator) from Abigail See. It adds a few new features to the original Pointer-Generator Architecture, and evaluates their performance on the NEWSROOM dataset.
The features are:
1. Length Prediction - predict summary length with a pretrained model and crop generated summary according to predicted length.
2. GAN Regularization - use a discriminator network to make generated summaries resemble reference summaries.
3. Stacked Encoder - use an encoder of a few layers.
4. Pretrained Word Embeddings - use pretrained word embeddings instead of trainable embeddings.
All the features are available using the suitable flags.

This code is in Python 2.

## How to run

### Get the dataset
follow the instructions [here](https://summari.es/download/) to obtain the NEWSROOM dataset and save it to the files: train.data, dev.data, test.data.

### Run training
To train your model, run:

```
python run_summarization.py --mode=train --data_path=/path/to/data/train.data --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

This will create a subdirectory of your specified `log_root` called `myexperiment` where all checkpoints and other data will be saved. Then the model will start training using the `train.data` file as training data.

**Warning**: Using default settings as in the above command, both initializing the model and running training iterations will probably be quite slow. To make things faster, try setting the following flags (especially `max_enc_steps` and `max_dec_steps`) to something smaller than the defaults specified in `run_summarization.py`: `hidden_dim`, `emb_dim`, `batch_size`, `max_enc_steps`, `max_dec_steps`, `vocab_size`. 

**Increasing sequence length during training**: Note that to obtain the results described in the paper, we increase the values of `max_enc_steps` and `max_dec_steps` in stages throughout training (mostly so we can perform quicker iterations during early stages of training). If you wish to do the same, start with small values of `max_enc_steps` and `max_dec_steps`, then interrupt and restart the job with larger values when you want to increase them.

### Run (concurrent) eval
You may want to run a concurrent evaluation job, that runs your model on the validation set and logs the loss. To do this, run:

```
python run_summarization.py --mode=eval --data_path=/path/to/data/dev.data --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

Note: you want to run the above command using the same settings you entered for your training job.

**Restoring snapshots**: The eval job saves a snapshot of the model that scored the lowest loss on the validation data so far. You may want to restore one of these "best models", e.g. if your training job has overfit, or if the training checkpoint has become corrupted by NaN values. To do this, run your train command plus the `--restore_best_model=1` flag. This will copy the best model in the eval directory to the train directory. Then run the usual train command again.

### Run beam search decoding
To run beam search decoding:

```
python run_summarization.py --mode=decode --data_path=/path/to/data/test.data --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

Note: you want to run the above command using the same settings you entered for your training job (plus any decode mode specific flags like `beam_size`).

This will repeatedly load random examples from your specified datafile and generate a summary using beam search. The results will be printed to screen.

**Visualize your output**: Additionally, the decode job produces a file called `attn_vis_data.json`. This file provides the data necessary for an in-browser visualization tool that allows you to view the attention distributions projected onto the text. To use the visualizer, follow the instructions [here](https://github.com/abisee/attn_vis).

If you want to run evaluation on the entire validation or test set and get ROUGE scores, set the flag `single_pass=1`. This will go through the entire dataset in order, writing the generated summaries to file, and then run evaluation using [pyrouge](https://pypi.python.org/pypi/pyrouge). (Note this will *not* produce the `attn_vis_data.json` files for the attention visualizer).

### Evaluate with ROUGE
`decode.py` uses the Python package [`pyrouge`](https://pypi.python.org/pypi/pyrouge) to run ROUGE evaluation. `pyrouge` provides an easier-to-use interface for the official Perl ROUGE package, which you must install for `pyrouge` to work. Here are some useful instructions on how to do this:
* [How to setup Perl ROUGE](http://kavita-ganesan.com/rouge-howto)
* [More details about plugins for Perl ROUGE](http://www.summarizerman.com/post/42675198985/figuring-out-rouge)

**Note:** As of 18th May 2017 the [website](http://berouge.com/) for the official Perl package appears to be down. Unfortunately you need to download a directory called `ROUGE-1.5.5` from there. As an alternative, it seems that you can get that directory from [here](https://github.com/andersjo/pyrouge) (however, the version of `pyrouge` in that repo appears to be outdated, so best to install `pyrouge` from the [official source](https://pypi.python.org/pypi/pyrouge)).

### Tensorboard
Run Tensorboard from the experiment directory (in the example above, `myexperiment`). You should be able to see data from the train and eval runs. If you select "embeddings", you should also see your word embeddings visualized.
