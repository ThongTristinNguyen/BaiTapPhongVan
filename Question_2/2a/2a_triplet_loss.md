
# Triplet Loss

**Triplet Loss** is a loss function used in deep learning, especially in classification and recognition tasks such as face recognition. The goal is to maximize the distance between different data points (negative) and minimize the distance between similar data points (positive).

## Definition
Triplet Loss compares the distance between an Anchor (A) and a Positive (P) with the distance between the Anchor and a Negative (N), adding a margin to ensure that dissimilar samples are farther apart than similar samples.

### Mathematical Formula
The formula for Triplet Loss is given by:

```
L(A, P, N) = max(0, ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha)
```

Where:
- `f(x)`: Represents the embedding function that maps the input `x` into the feature space.
- `||f(A) - f(P)||^2`: The distance between the anchor and the positive samples.
- `||f(A) - f(N)||^2`: The distance between the anchor and the negative samples.
- `alpha`: A margin that ensures dissimilar samples are farther apart by at least this value.