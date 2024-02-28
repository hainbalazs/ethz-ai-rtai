# Toeplitz matrix factorization

In this Jupyter Notebook file, I demonstrated how can the effect of a torch.Conv2d convolutional layer be represented as a matrix multiplication. This transformation supports convolutional layers up to arbitrary input size and dimension, kernel dimension, 0-based padding, and stride values. I eventually verified in the notebook that the transformation indeed accurately reconstructs the desired output. 

It was a neccessary step to enable symbolic verification of convolutional neural networks and thus implement [DeepPoly](https://ggndpsngh.github.io/files/DeepPoly.pdf) (see deeppoly-project/code/deeppoly.py) as part of the coursework for Reliable and Trustworthy AI (263-2400-00L). 
Given the resulted linear operation, it is now natural to impose abstract convex shapes as bounds on the convolution layer for certification, while understanding that efficiency issues might arise with this approach.

## DeepPoly

In this course project the aim was to implement DeepPoly based on the publication above and the course material, to create a deterministic, convex-relation based neural network verifier, that empirically should achieve the best approximation precision in contrast to other relaxations.

### Task
Implement a sound and at least as precise version of DeepPoly than the original one for fully connected, convolution, relu and leaky relu layers.

## Required skills
Python, Pytorch, Linear Algebra, AI

## Authorship
The toeplitz matrix case study was entirely developed by Balázs Hain, whereas the DeepPoly project was a collaboration between Balázs Hain and Panagiotis Grontas.