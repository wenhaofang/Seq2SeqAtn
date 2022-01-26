## Seq2Seq

This is a Seq2Seq model with various attention mechanisms, decoding strategies, and other tricks.

### Data Process

```shell
PYTHONPATH=. python dataprocess/process.py
```

### Unit Test

* for loader

```shell
PYTHONPATH=. python loaders/loader.py
```

* for module

```shell
# Seq2Seq with Attention Bahdanau
PYTHONPATH=. python modules/Seq2SeqAttentionBahdanau.py
# Seq2Seq with Attention Luong
PYTHONPATH=. python modules/Seq2SeqAttentionLuong.py
```
