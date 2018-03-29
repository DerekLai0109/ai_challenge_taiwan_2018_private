# PyTorch or TensorFlow?

YingTing 2018/03/28

This post describes differences between PyTorch and TensorFlow, but without performance \(speed / memory usage\) trade-offs.

### Summary

PyTorch is better for rapid prototyping in research for small scale projects. TensorFlow is better for large-scale deployments, especially when cross-platform and embedded deployment is a consideration.

### Ramp-up Time

_Winner: PyTorch_

PyTorch is a GPU enabled drop-in replacement for NumPy, PyTorch equipped with higher-level functionality for building and training deep neural networks than TensorFlow.

TensorFlow is a programming language embedded within Python. TensorFlow code get "compiled" into a graph by Python and then run by the TensorFlow execution engine.

