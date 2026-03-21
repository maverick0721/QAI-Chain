import torch
from quantum.models.qnn import QNN


def test_qnn_forward_output_shape_and_dtype():

    torch.manual_seed(7)
    model = QNN()
    x = torch.randn(2, 5).float()

    out = model(x)

    assert out.shape == (2, 1)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all().item()


def test_qnn_forward_is_deterministic_with_fixed_seed():

    torch.manual_seed(123)
    model_a = QNN()
    x_a = torch.randn(2, 5).float()
    out_a = model_a(x_a)

    torch.manual_seed(123)
    model_b = QNN()
    x_b = torch.randn(2, 5).float()
    out_b = model_b(x_b)

    assert torch.allclose(out_a, out_b, atol=1e-6)