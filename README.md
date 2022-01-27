## Seq2Seq

This is a Seq2Seq model with various attention mechanisms, decoding strategies, and other tricks.

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
