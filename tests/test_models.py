import numpy as np
import torch

from src.models.nn_residual import ResidualRNN


def test_residual_rnn_forward():
    model = ResidualRNN(input_dim=2, hidden_dim=8, rnn_type="gru")
    x = torch.randn(4, 5, 2)
    out = model(x)
    assert out.shape == (4, 2)
