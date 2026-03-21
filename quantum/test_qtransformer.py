import torch
from quantum.transformer.q_transformer import QTransformer


def test_qtransformer_forward_output_shape_and_dtype():

    torch.manual_seed(11)
    model = QTransformer()
    x = torch.randn(2, 4, 5)

    out = model(x)

    assert out.shape == (2, 4, 1)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all().item()


def test_qtransformer_backward_pass_produces_gradients():

    torch.manual_seed(19)
    model = QTransformer()
    x = torch.randn(2, 4, 5)

    out = model(x)
    loss = out.mean()
    loss.backward()

    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all().item()