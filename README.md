## Minimal Automatic Differentiation

This project contains a minimal automatic differentiation engine (inspired by [micrograd](https://github.com/karpathy/micrograd)). Automatic differentiation works by building a mathematical DAG and subsequently applying the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) of differentiation (and it's [multivariable](https://en.wikipedia.org/wiki/Chain_rule#Multivariable_case) extension) in the backward
pass to compute gradients.

We use this scalar autodiff engine to define a `Module`, similar to PyToch, that automatically registers model parameters. Based on this `Module` we build a quintic polynomial model and fit it with vanilla stochastic gradient descent to a small generated dataset. The resulting fit of the model looks like this:

![result](https://github.com/cschoeller/autodiff/assets/1900532/80861daa-e0b8-4c08-a73e-881d4e6325b6)

This minimal example illustrates how modern deep learning engines work. The mechanism is silmiar, but instead of scalar variables they are implemented for tensors and optionally executed on GPU.
