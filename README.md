# Kirigami

RNA secondary structure prediction via deep learning.

From [Wikipedia](https://en.wikipedia.org/wiki/Kirigami):

> Kirigami (切り紙) is a variation of origami, the Japanese art of folding paper. In kirigami, the paper is cut as well as being folded, resulting in a three-dimensional design that stands away from the page.

The Kirigami pipeline both folds RNA molecules via a fully convolutional neural network (FCN) and uses Nussinov-style dynamic programming to recursively cut them into subsequences for pre- and post-processing.

All code is written idiomatically according to the [Lightning](https://www.pytorchlightning.ai) specification for PyTorch.
