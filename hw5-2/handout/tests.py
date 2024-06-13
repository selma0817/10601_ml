"""
tests.py

Unit tests for HW5 neural network implementation.

To run one test: python -m unittest tests.TestRandomInit.test_shape
To run one set of tests: python -m unittest tests.TestRandomInit
To run all tests: python -m unittest tests
"""

import unittest
import numpy as np
import pickle as pk
from numpy.testing import assert_allclose

from neuralnet import (
    Linear, Sigmoid, SoftMaxCrossEntropy, NN,
    zero_init, random_init
)

TOLERANCE = 1e-4

with open("unittest_data.pk", "rb") as f:
    data = pk.load(f)


class TestRandomInit(unittest.TestCase):

    def test_shape(self):
        shape = (10, 5)
        random_matrix = random_init(shape)
        assert random_matrix.shape == shape

    def test_bounds(self):
        random_matrix = random_init((5, 7))
        assert (-0.1 <= random_matrix).all()
        assert (0.1 >= random_matrix).all()

    def test_variation_row(self):
        random_matrix = random_init((20, 1))
        assert len(np.unique(random_matrix)) > 1

    def test_variation_column(self):
        random_matrix = random_init((1, 20))
        assert len(np.unique(random_matrix)) > 1


class TestLinear(unittest.TestCase):
    """
    Note that these tests assume the shape of your weight matrix
    <output dimension> followed by <input dimension>, so if you
    wrote it the other way around, you may have to fiddle around with
    the shapes in these tests to make them pass.
    """

    def test_forward(self):
        T1, _ = data["linear_forward"]
        w, X, soln = T1[0], T1[1][1:], T1[2]
        # init the Linear layer arbitrarily, then fill in the weight matrix 
        # for the test case
        layer = Linear(1, 1, zero_init, 0.0)
        layer.w = w
        a = layer.forward(X)
        assert_allclose(np.squeeze(a), soln)

    def test_bias_zeroinit(self):
        in_size = 5
        x = np.zeros(in_size)
        zero_linear = Linear(input_size=in_size, output_size=10,
                             weight_init_fn=zero_init, learning_rate=1)
        assert np.count_nonzero(zero_linear.forward(x)) == 0

    def test_bias_randominit(self):
        in_size = 5
        x = np.zeros(in_size)
        zero_linear = Linear(input_size=in_size, output_size=10,
                             weight_init_fn=random_init, learning_rate=1)
        assert np.count_nonzero(zero_linear.forward(x)) == 0

    def test_backward(self):
        T = data["linear_backward"]
        X, w, dxsoln, dwsoln = T[0][1:], T[1], T[2], T[3]
        layer = Linear(1, 1, zero_init, 0.0)
        layer.w = w
        z = layer.forward(X)  # forward pass to ensure layer caches values
        dz = np.ones_like(z)  # use all 1's for gradient w.r.t output
        dx = layer.backward(dz)
        dw = layer.dw
        assert_allclose(np.squeeze(dx), dxsoln)
        assert_allclose(np.squeeze(dw), dwsoln)


class TestSigmoid(unittest.TestCase):
    def test_forward_1(self):
        T1, _ = data["sigmoid_forward"]
        a, soln = T1
        sigmoid = Sigmoid()
        z = sigmoid.forward(a)
        assert_allclose(np.squeeze(z), soln)

    def test_forward_2(self):
        _, T2 = data["sigmoid_forward"]
        a, soln = T2
        sigmoid = Sigmoid()
        z = sigmoid.forward(a)
        assert_allclose(np.squeeze(z), soln)

    def test_backward(self):
        T = data["sigmoid_backward"]
        z, soln = T
        sigmoid = Sigmoid()
        _ = sigmoid.forward(z)
        dz = sigmoid.backward(1)
        assert_allclose(np.squeeze(dz), soln)


class TestSoftmax(unittest.TestCase):
    def test_softmax_forward_1(self):
        T1, T2 = data["softmax_forward"]
        z, soln = T1
        yh = SoftMaxCrossEntropy()._softmax(z)
        assert_allclose(np.squeeze(yh), soln)

    def test_softmax_forward_2(self):
        T1, T2 = data["softmax_forward"]
        z, soln = T2
        yh = SoftMaxCrossEntropy()._softmax(z)
        assert_allclose(np.squeeze(yh), soln)

    def test_crossentropy(self):
        T = data["ce_forward"]
        yh, y, soln = T
        loss = SoftMaxCrossEntropy()._cross_entropy(y, yh)
        assert_allclose(np.squeeze(loss), soln)

    def test_backward(self):
        T = data["ce_backward"]
        y, yh, soln = T
        db = SoftMaxCrossEntropy().backward(y, yh)
        assert_allclose(np.squeeze(db), soln)


class TestNN(unittest.TestCase):

    def test_forward(self):
        T1, _ = data["forward_backward"]
        x, y, soln, _, _ = T1
        x = x[1:]
        nn = NN(input_size=len(x), hidden_size=4, output_size=10,
                learning_rate=1, weight_init_fn=zero_init)
        yh, loss = nn.forward(x, 1)
        assert_allclose(np.squeeze(yh), soln)

    def test_backward(self):
        T1, _ = data["forward_backward"]
        x, y, soln_yh, soln_d_w1, soln_d_w2 = T1
        x = x[1:]
        nn = NN(input_size=len(x), hidden_size=4, output_size=10,
                learning_rate=1, weight_init_fn=zero_init)
        yh, _ = nn.forward(x, y)
        nn.backward(y, yh)
        assert_allclose(np.squeeze(nn.linear1.dw), soln_d_w1)
        assert_allclose(np.squeeze(nn.linear2.dw), soln_d_w2)
