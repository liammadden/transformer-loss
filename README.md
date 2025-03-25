# Decoder-only Transformer

We train a transformer model on a subset of the TinyStories dataset, https://arxiv.org/abs/2305.07759. The transformer model is the composition of a token embedding, a self-attention head, and a two-layer FNN with GELU activation. We train it on the cross-entropy loss using Adam.

## Installing Required Packages

To install the required python packages, use the following command:

```
pip install -r requirements.txt
```

## Running the Code

To run the code, use the following command:

```
python run_experiments.py
```