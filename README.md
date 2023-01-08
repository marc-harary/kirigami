# Kirigami

RNA secondary structure prediction via deep learning.

From [Wikipedia](https://en.wikipedia.org/wiki/Kirigami):

> Kirigami (切り紙) is a variation of origami, the Japanese art of folding paper. In kirigami, the paper is cut as well as being folded, resulting in a three-dimensional design that stands away from the page.

The Kirigami pipeline both uses dynamic programming to recursively cut RNA molecules into subsequences and convolutional neural network (CNN) to fold them.

All code is written idiomatically according to the [PyTorch Lightning](https://www.pytorchlightning.ai) specifiation for PyTorch.
