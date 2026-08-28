Here is the text rewritten with readable equations. I’ve repaired the broken symbols and made the notation consistent.

# Backpropagation over a mini-batch

The four equations are correct under the conventions explained below. Two details need care:

- The first equation **defines** the error signal; it does not, by itself, tell us how to compute it.
- When propagating backward, the activation derivative must belong to the **previous layer**.

## Notation and dimensions

For layer $l$, the forward pass is:

$$
Z^l = W^l A^{l-1} + b^l
$$

$$
A^l = g^l(Z^l)
$$

Here, the superscript $l$ identifies a layer—it is **not an exponent**. The bias vector is added to every column.

Assume training examples are stored as columns:

| Symbol | Shape | Meaning |
|---|---|---|
| $A^{l-1}$ | $n_{l-1} \times m$ | Inputs to layer $l$ |
| $W^l$ | $n_l \times n_{l-1}$ | Weight matrix |
| $b^l$ | $n_l \times 1$ | Bias vector |
| $Z^l$ | $n_l \times m$ | Values before applying the activation |
| $A^l$ | $n_l \times m$ | Values after applying the activation |
| $dZ^l$ | $n_l \times m$ | Error signals before batch averaging |

Also:

- $n_l$: number of neurons in layer $l$.
- $m$: number of examples in the batch.
- $T$: matrix transpose.
- $\odot$: element-wise multiplication.
- $g^{l\,\prime}$: derivative of the activation function at layer $l$.

Let $\mathcal{L}^{(i)}$ denote the loss for example $i$. The average batch loss is:

$$
J = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}^{(i)}
$$

**Convention used below:** each column of $dZ^l$ contains a single example’s loss gradient. We apply the averaging factor $1/m$ when calculating $dW^l$ and $db^l$.

---

## 1. Error signal at a layer

For one example, the error signal is defined as:

$$
\boxed{dZ^l = \frac{\partial \mathcal{L}}{\partial Z^l}}
$$

This measures how sensitive the loss $L$ is to the pre-activation value $Z^l$ at layer $l$. Think of it as "how much blame does this layer's raw output deserve for the final error?" Every other gradient in the layer is derived from this term — it's the central quantity that gets passed backward through the network.This measures how sensitive the loss $L$ is to the pre-activation value $Z^l$ at layer $l$. Think of it as "how much blame does this layer's raw output deserve for the final error?" Every other gradient in the layer is derived from this term — it's the central quantity that gets passed backward through the network.

Because:

$$
A^l = g^l(Z^l)
$$

the chain rule gives, for an element-wise activation:

$$
\boxed{dZ^l = dA^l \odot g^{l\,\prime}(Z^l)}
$$

where:

$$
dA^l = \frac{\partial \mathcal{L}}{\partial A^l}
$$

These equations also apply to a batch by arranging the individual examples’ gradients into columns.

### Why is $dZ^l$ useful?

The weights and biases enter the calculation through:

$$
Z^l = W^l A^{l-1} + b^l
$$

Once we know $dZ^l$, we can calculate the weight gradient, bias gradient, and error passed to the preceding layer.

### Special case: sigmoid output with binary cross-entropy

Let $L$ identify the final layer. For one example with a single binary output:

$$
A^L = \sigma(Z^L)
$$

The binary cross-entropy loss is:

$$
\mathcal{L}
=
-\left[y\log A^L + (1-y)\log(1-A^L)\right]
$$

Its derivative with respect to the output activation is:

$$
\frac{\partial \mathcal{L}}{\partial A^L}
=
\frac{A^L-y}{A^L(1-A^L)}
$$

The sigmoid derivative is:

$$
\frac{\partial A^L}{\partial Z^L}
=
A^L(1-A^L)
$$

Multiplying these using the chain rule:

$$
\begin{aligned}
dZ^L
&=
\frac{\partial \mathcal{L}}{\partial A^L}
\frac{\partial A^L}{\partial Z^L}
\\[4pt]
&=
\frac{A^L-y}{A^L(1-A^L)}
\;A^L(1-A^L)
\\[4pt]
&= A^L-y
\end{aligned}
$$

Therefore, for the whole batch:

$$
\boxed{dZ^L = A^L-Y}
$$

Here, $Y$ contains the target labels.

This derivation assumes sigmoid output with binary cross-entropy. Do not assume the same formula holds for every activation and loss combination.

---

## 2. Weight gradient

The weight gradient for the average batch loss is:

$$
\boxed{
dW^l
=
\frac{\partial J}{\partial W^l}
=
\frac{1}{m}\,dZ^l(A^{l-1})^T
}
$$

### Where does this come from?

For neuron $j$ and one training example:

$$
Z_j^l
=
\sum_k W_{jk}^l A_k^{l-1} + b_j^l
$$

The derivative with respect to weight $W_{jk}^l$ is:

$$
\frac{\partial Z_j^l}{\partial W_{jk}^l}
=
A_k^{l-1}
$$

Applying the chain rule:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W_{jk}^l}
&=
\frac{\partial \mathcal{L}}{\partial Z_j^l}
\frac{\partial Z_j^l}{\partial W_{jk}^l}
\\[4pt]
&=
dZ_j^l A_k^{l-1}
\end{aligned}
$$

In matrix form, for one example:

$$
\frac{\partial \mathcal{L}}{\partial W^l}
=
dZ^l(A^{l-1})^T
$$

For a batch, matrix multiplication sums these contributions across examples, and dividing by $m$ averages them:

$$
dW^l
=
\frac{1}{m}\,dZ^l(A^{l-1})^T
$$

### Shape check

$$
\underbrace{dZ^l}_{n_l \times m}
\;
\underbrace{(A^{l-1})^T}_{m \times n_{l-1}}
\quad\longrightarrow\quad
\underbrace{dW^l}_{n_l \times n_{l-1}}
$$

The result has the same shape as $W^l$.

### Intuition

For one example:

$$
\text{weight gradient}
=
\text{destination error signal}
\times
\text{source activation}
$$

This tells you how to adjust the weights $W^l$. Intuitively: the gradient for a weight connecting neuron $i$ (previous layer) to neuron $j$ (this layer) depends on **two things multiplied together** — how active neuron $i$ was ($A^{l-1}$) and how much error is flowing out of neuron $j$ ($dZ^l$). If a neuron was very active and contributed to a big error, its connecting weight gets a big correction. The $\frac{1}{m}$ averages this over the whole batch.

---

## 3. Bias gradient

The bias gradient is:

$$
\boxed{
db^l
=
\frac{\partial J}{\partial b^l}
=
\frac{1}{m}\sum_{i=1}^{m}dZ^{l(i)}
}
$$

Here, $dZ^{l(i)}$ means the column of $dZ^l$ corresponding to example $i$.

**The sum is across training examples, not across neurons.**

### Where does this come from?

For neuron $j$ and one example:

$$
Z_j^l
=
\sum_k W_{jk}^l A_k^{l-1} + b_j^l
$$

Because:

$$
\frac{\partial Z_j^l}{\partial b_j^l}=1
$$

the chain rule gives:

$$
\frac{\partial \mathcal{L}}{\partial b_j^l}
=
\frac{\partial \mathcal{L}}{\partial Z_j^l}
=
dZ_j^l
$$

Averaging across examples gives the batch bias gradient.

### Shape

$$
db^l \in \mathbb{R}^{n_l \times 1}
$$

This matches the shape of $b^l$.

In NumPy:

```python
db = np.sum(dZ, axis=1, keepdims=True) / m
```

### Intuition

A bias shifts a neuron’s pre-activation equally for every example. Its gradient is therefore the neuron’s average error signal.

---

## 4. Propagating the error to the preceding layer



This is the step that makes it "back-propagation" — it computes the error term for the *previous* layer using the current layer's error.
- $(W^l)^T dZ^l$: pushes the error backward through the weights (redistributing blame to the previous layer's neurons, weighted by how strongly they were connected)
- $\cdot\, g'(Z^{l-1})$: element-wise multiplies by the derivative of the previous layer's activation function, since if that neuron's activation function was flat (near-zero slope) at that point, changing $Z^{l-1}$ wouldn't have changed the output much, so it shouldn't get much blame either  
  
To propagate from layer $l$ into hidden layer $l-1$:

$$
\boxed{
dZ^{l-1}
=
\left((W^l)^T dZ^l\right)
\odot
g^{(l-1)\,\prime}(Z^{l-1})
}
$$

The activation derivative belongs to **layer $l-1$**.

This calculation has two steps.

### Step A: Propagate through the linear transformation

Since:

$$
Z^l = W^l A^{l-1} + b^l
$$

we get:

$$
\boxed{
dA^{l-1} = (W^l)^T dZ^l
}
$$

The transposed weight matrix distributes the current layer’s error backward through its connections.

### Step B: Propagate through the activation

Since:

$$
A^{l-1} = g^{l-1}(Z^{l-1})
$$

we get:

$$
\boxed{
dZ^{l-1}
=
dA^{l-1}
\odot
g^{(l-1)\,\prime}(Z^{l-1})
}
$$

Substituting Step A into Step B gives the combined equation above.

### Shape check

$$
\underbrace{(W^l)^T}_{n_{l-1} \times n_l}
\;
\underbrace{dZ^l}_{n_l \times m}
\quad\longrightarrow\quad
\underbrace{dA^{l-1}}_{n_{l-1} \times m}
$$

Both $dA^{l-1}$ and the activation derivative have shape $n_{l-1}\times m$, so they can be multiplied element by element.

### What does the activation derivative do?

It determines how the incoming gradient is scaled at each neuron.

For ReLU:

$$
g'(z)=
\begin{cases}
1, & z>0\\
0, & z<0
\end{cases}
$$

At $z=0$, ReLU is not differentiable; implementations commonly use $0$.

A neuron with negative pre-activation blocks the gradient through that ReLU.

For sigmoid:

$$
g'(z)=\sigma(z)\left(1-\sigma(z)\right)
$$

When the sigmoid output is close to $0$ or $1$, its derivative is small, so the gradient passing backward is reduced.

---

## Important: apply the averaging factor only once

The convention used here is:

$$
dZ^L = A^L-Y
$$

followed by:

$$
dW^l=\frac{1}{m}\,dZ^l(A^{l-1})^T
$$

$$
db^l=\frac{1}{m}\sum_{i=1}^{m}dZ^{l(i)}
$$

Another valid convention incorporates $1/m$ into $dZ$ from the start. In that case, **do not divide by $m$ again** when calculating $dW$ and $db$.

Both conventions work. Mixing them averages the gradients twice.

## Compact summary

For element-wise hidden-layer activations:

$$
\boxed{
\begin{aligned}
dZ^l
&= dA^l \odot g^{l\,\prime}(Z^l)
\\[6pt]
dW^l
&= \frac{1}{m}\,dZ^l(A^{l-1})^T
\\[6pt]
db^l
&= \frac{1}{m}\sum_{i=1}^{m}dZ^{l(i)}
\\[6pt]
dZ^{l-1}
&= \left((W^l)^T dZ^l\right)
   \odot g^{(l-1)\,\prime}(Z^{l-1})
\end{aligned}
}
$$

These form a valid matrix summary of backpropagation under the stated batch convention. Calling this particular collection “the four fundamental equations” is a teaching convention, rather than universal terminology.