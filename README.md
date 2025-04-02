# Decoder-only Transformer

We train a transformer model on a subset of the TinyStories dataset, https://arxiv.org/abs/2305.07759. The transformer model is the composition of a token embedding, a self-attention head, and a two-layer FNN with GELU activation. We train it on the cross-entropy loss using Adam. This is a modification of the code in curtfox/decoder-memory-capacity for https://arxiv.org/abs/2405.13718. We take a subset of 100 stories for our training set and 100 stories for our test set, truncating each story to 100 words. The overall vocabulary ends up having size 1858. We vary the number of neurons in the first FNN layer from 10 to 90, setting the embedding dimension equal to it. For training, we take 1000 epochs of Adam using a full batch. We plot the final training error and final test error as functions of the number of parameters. While the final training error decreases monotonically as the number of parameters increases, the final test error decreases then increases, with a minimum at 158,698 parameters. This is consistent with the traditional bias-variance tradeoff, not the modern double descent curve. This suggests that, given a data set size, too many parameters and epochs will indeed lead to overfitting, so they must be increased jointly with the data set size in order to decrease the test loss.

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
