# Iris Classification: Multi-Class with Real Data

## The Iris Dataset

The Iris dataset is one of the most well-known datasets in machine learning,
introduced by statistician Ronald Fisher in 1936. It contains 150 samples of
iris flowers, evenly split across three species:

- **Setosa** — easily separable from the other two
- **Versicolor** — partially overlaps with Virginica
- **Virginica** — partially overlaps with Versicolor

Each sample has four features:

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

The task is to predict the species from these four measurements. This is a
**multi-class classification** problem (3 classes, not just 2).

## Multi-Class Classification

With two classes, the network outputs 2 logits and picks the larger one. With
three or more classes, the same idea scales directly: the final layer outputs
one logit per class, and the predicted class is the argmax.

The network architecture for Iris is: 4 inputs → 16 hidden → 16 hidden → 3
outputs. The final layer has 3 neurons — one per species.

## Softmax and Cross-Entropy for Multiple Classes

Softmax generalizes to any number of classes:

```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

For 3 classes, this produces a 3-element probability vector like [0.7, 0.2, 0.1].
Cross-entropy loss works the same way — it only looks at the predicted
probability for the correct class:

```
loss = -(1/n) Σ log(p_correct_class)
```

Candle's `loss::cross_entropy` handles both softmax and the log computation
internally, so you pass raw logits and integer labels.

## Train/Test Split

In previous exercises, we trained and evaluated on the same data. This is
dangerous with real data because the model might **memorize** the training
examples instead of learning general patterns.

A **train/test split** holds back a portion of the data (typically 20-30%) that
the model never sees during training. Accuracy on this held-out **test set** is
a much better estimate of real-world performance:

- **Training accuracy** measures how well the model fits the data it learned from.
- **Test accuracy** measures how well it generalizes to new, unseen data.

If training accuracy is high but test accuracy is low, the model is
**overfitting**. If both are low, the model may need more capacity or training.

## What to Expect

Iris is a relatively easy dataset. A simple 2-layer network should reach
>90% test accuracy within a few hundred epochs. Setosa is nearly always
classified perfectly; most errors occur between Versicolor and Virginica,
which genuinely overlap in feature space.
