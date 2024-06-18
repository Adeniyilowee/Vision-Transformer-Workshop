import torch
import pytest
from vit_model.model import Patch_Embeddings, Transformer_Encoder_Block, Head, MSAttention_Block, MultiHeadAttention, MLP_Block, Vision_Transformer


@pytest.fixture(scope="session")
def sample_input_PE():

    return torch.randn(1, 3, 224, 224)


def test_initialization_PE():

    model = Patch_Embeddings()
    assert model.patch_size == 16
    assert isinstance(model.patcher, torch.nn.Conv2d)
    assert isinstance(model.flatten, torch.nn.Flatten)


def test_forward_pass_PE(sample_input_PE):

    model = Patch_Embeddings()
    output = model(sample_input_PE)

    # Check output shape
    assert output.shape == (1, 196, 768)

    # Ensure the permutation is correct
    assert output.permute(0, 2, 1).shape == (1, 768, 196)


@pytest.fixture(scope="session")
def sample_input_TEB():

    return torch.randn(1, 196, 768)


def test_initialization_TEB():

    model = Transformer_Encoder_Block()
    assert isinstance(model.msa_block, MSAttention_Block)
    assert isinstance(model.mlp_block, MLP_Block)


def test_forward_pass_TEB(sample_input_TEB):

    model = Transformer_Encoder_Block()
    output = model(sample_input_TEB)

    # Check output shape
    assert output.shape == sample_input_TEB.shape


@pytest.fixture(scope="session")
def sample_input_H():

    return {'query': torch.randn(1, 196, 768),
            'key': torch.randn(1, 196, 768),
            'value': torch.randn(1, 196, 768)}


def test_initialization_H():
    # Test initialization with default parameters
    model = Head(embed_dim=768, head_size=64, dropout=0.1)
    assert isinstance(model.key, torch.nn.Linear)
    assert isinstance(model.query, torch.nn.Linear)
    assert isinstance(model.value, torch.nn.Linear)
    assert isinstance(model.dropout, torch.nn.Dropout)


def test_forward_pass_H(sample_input_H):

    model = Head(embed_dim=768, head_size=64, dropout=0.1)
    output = model(sample_input_H['query'], sample_input_H['key'], sample_input_H['value'])

    # Check output shape
    assert output.shape == (1, 196, 64)


@pytest.fixture(scope="session")
def sample_input_MHA():
    # Fixture providing sample input tensors
    return {'query': torch.randn(1, 196, 768),
            'key': torch.randn(1, 196, 768),
            'value': torch.randn(1, 196, 768)}


def test_initialization_MHA():
    # Test initialization with default parameters
    model = MultiHeadAttention(embed_dim=768, num_heads=12, dropout=0.1)
    assert isinstance(model.heads, torch.nn.ModuleList)
    assert len(model.heads) == 12
    assert isinstance(model.proj, torch.nn.Linear)
    assert isinstance(model.dropout, torch.nn.Dropout)


def test_forward_pass_MHA(sample_input_MHA):
    # Test forward pass of the model
    model = MultiHeadAttention(embed_dim=768, num_heads=12, dropout=0.1)
    output = model(sample_input_MHA['query'], sample_input_MHA['key'], sample_input_MHA['value'])

    # Check output shape
    assert output.shape == (1, 196, 768)


@pytest.fixture(scope="session")
def sample_input_MSAB():

    return torch.randn(1, 196, 768)


def test_initialization_MSAB():
    # Test initialization with default parameters
    model = MSAttention_Block()
    assert isinstance(model.layer_norm, torch.nn.LayerNorm)
    assert isinstance(model.multihead_attn, MultiHeadAttention)


def test_forward_pass_MSAB(sample_input_MSAB):
    # Test forward pass of the model
    model = MSAttention_Block()
    output = model(sample_input_MSAB)

    # Check output shape
    assert output.shape == sample_input_MSAB.shape


@pytest.fixture(scope="session")
def sample_input_MLPB():

    return torch.randn(1, 196, 768)


def test_initialization_MLPB():

    model = MLP_Block()
    assert isinstance(model.layer_norm, torch.nn.LayerNorm)
    assert isinstance(model.mlp, torch.nn.Sequential)


def test_forward_pass_MLPB(sample_input_MLPB):

    model = MLP_Block()
    output = model(sample_input_MLPB)

    # Check output shape
    assert output.shape == sample_input_MLPB.shape


@pytest.fixture(scope="session")
def sample_input_VT():

    return torch.randn(1, 3, 224, 224)


def test_initialization_VT():
    # Test initialization with default parameters
    model = Vision_Transformer()
    assert model.num_patches == (224 * 224) // (16 * 16)
    assert isinstance(model.class_embedding, torch.nn.Parameter)
    assert model.position_embedding.shape == (1, model.num_patches + 1, 768)
    assert isinstance(model.embedding_dropout, torch.nn.Dropout)
    assert isinstance(model.patch_embedding, Patch_Embeddings)
    assert isinstance(model.transformer_encoder, torch.nn.Sequential)
    assert isinstance(model.classifier, torch.nn.Sequential)


@pytest.mark.skip(reason="Not done.. might fail")
def test_forward_pass_VT(sample_input_VT):
    # Test forward pass of the model
    model = Vision_Transformer()
    output = model(sample_input_VT)

    # Check output shape
    assert output.shape == (1, 1000)


@pytest.mark.xfail
def test_divide_by_zero():
    assert 1/0 == 1
