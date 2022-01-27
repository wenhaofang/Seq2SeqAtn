## Seq2Seq

This is a Seq2Seq model with various attention mechanisms, decoding strategies, and other tricks.

Datasets:

* `dataset1`: [news-commentary-v14.de-en](http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz)

Models:

* `model1`: [Seq2Seq with Attention Bahdanau](https://arxiv.org/abs/1409.0473)
* `model2`: [Seq2Seq with Attention Luong](https://arxiv.org/abs/1508.04025)

### Data Process

```shell
PYTHONPATH=. python dataprocess/process.py
```

### Unit Test

* for loader

```shell
PYTHONPATH=. python loaders/loader1.py
```

* for module

```shell
PYTHONPATH=. python modules/module1.py # Seq2Seq with Attention Bahdanau
PYTHONPATH=. python modules/module2.py # Seq2Seq with Attention Luong
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples for each module:

```shell
# module1
python main.py --module 1
```

```shell
# module2
python main.py --module 2
```
