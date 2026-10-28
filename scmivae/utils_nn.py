import math
import torch
import torch.nn as nn
from numbers import Number

# x-tanh activation function
class Xtanh(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return self.beta * x.tanh() + self.alpha * x

# Weight initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

# Generic MLP class
class MLP(nn.Module):
    def __init__(
        self,
        input_dim, output_dim, hidden_dim,
        n_layers, dropout=0.0,
        activation="lrelu",
        slope=0.1, scale=1.0,
        device="cpu",        
        use_batchnorm=False,
        *args, **kwargs
    ):
        super().__init__()
        self.device = device

        # Hidden dimension normalization
        if isinstance(hidden_dim, Number):
            hidden_dim = [hidden_dim] * (n_layers - 1)
        elif not isinstance(hidden_dim, list):
            raise ValueError(f"hidden_dim must be int or list, got {type(hidden_dim)}")

        # Activation normalization
        if isinstance(activation, str):
            activation = [activation] * (n_layers - 1)
        elif not isinstance(activation, list):
            raise ValueError(f"activation must be str or list, got {type(activation)}")

        if len(activation) != n_layers - 1:
            raise ValueError("activation list length must match number of hidden layers")

        # Activation factory
        def get_activation(name):
            name = name.lower()
            if name == "relu":
                return nn.ReLU()
            elif name == "lrelu":
                return nn.LeakyReLU(negative_slope=slope)
            elif name == "xtanh":
                return Xtanh(alpha=slope, beta=scale)
            elif name == "sigmoid":
                return nn.Sigmoid()
            elif name == "softplus":
                return nn.Softplus()
            elif name == "none":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown activation: {name}")

        # Build layers
        layer_dims = [input_dim] + hidden_dim + [output_dim]
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < n_layers - 1:  # no act/dropout/bn after final layer
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(layer_dims[i+1]))
                layers.append(get_activation(activation[i]))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        return self.net(x)

# Categorical embedding class
def heuristic_embedding_dim(n_categories, min_dim=12):
    # heuristic: sqrt capped at 50
    return min(min_dim, math.ceil(math.sqrt(n_categories)))

# Categorical embedding class with heuristic embedding dimension
class CatEmbedding(nn.Module):
    def __init__(self, category_sizes, embedding_dim, freeze_embeddings=False):
        """
        category_sizes: list of number of categories for each categorical variable
        """
        super().__init__()
        if embedding_dim is None:
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=n, embedding_dim=heuristic_embedding_dim(n))
                for n in category_sizes
            ])
            self.embedding_dims = [heuristic_embedding_dim(n) for n in category_sizes]
            self.output_dim = sum(heuristic_embedding_dim(n) for n in category_sizes)
        else:
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim)
                for n in category_sizes
            ])
            self.embedding_dims = [embedding_dim] * len(category_sizes)
            self.output_dim = embedding_dim * len(category_sizes)

        if freeze_embeddings:
            for emb in self.embeddings:
                emb.weight.requires_grad = False

    def forward(self, x):
        """
        x: tensor of shape [batch_size, num_categorical_features]
           each column is the index for that categorical variable
        """
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]  # list of [batch_size, emb_dim]
        return torch.cat(embedded, dim=1)  # [batch_size, total_emb_dim]
    
    def check_embedding_info(self, embedding_idx: int = 0):
        """
        Diagnostic method to check embedding configuration.
        
        Parameters
        ----------
        embedding_idx : int
            Index of the embedding to check (default: 0)
        
        Returns
        -------
        dict
            Dictionary with embedding information including num_embeddings, 
            embedding_dim, padding_idx, and weight shape.
        """
        if embedding_idx >= len(self.embeddings):
            raise IndexError(f"embedding_idx {embedding_idx} out of range. "
                           f"Only {len(self.embeddings)} embeddings available.")
        
        emb = self.embeddings[embedding_idx]
        info = {
            'num_embeddings': emb.num_embeddings,
            'embedding_dim': emb.embedding_dim,
            'padding_idx': emb.padding_idx,
            'weight_shape': emb.weight.shape,
            'weight_requires_grad': emb.weight.requires_grad,
        }
        return info
    
    def test_out_of_range_index(self, embedding_idx: int = 0, test_index: int = None, device: str = 'cpu'):
        """
        Test what happens when an out-of-range index is passed to the embedding.
        
        Parameters
        ----------
        embedding_idx : int
            Which embedding to test (default: 0)
        test_index : int, optional
            Index to test. If None, uses num_embeddings + 3
        device : str
            Device to run test on ('cpu', 'cuda', 'mps')
        
        Returns
        -------
        dict
            Dictionary with test results including:
            - 'error_raised': bool, whether IndexError was raised
            - 'output': torch.Tensor or None, the output if no error
            - 'output_is_zero': bool, whether output is all zeros
            - 'test_index': int, the index that was tested
        """
        if embedding_idx >= len(self.embeddings):
            raise IndexError(f"embedding_idx {embedding_idx} out of range. "
                           f"Only {len(self.embeddings)} embeddings available.")
        
        emb = self.embeddings[embedding_idx]
        if test_index is None:
            test_index = emb.num_embeddings + 3
        
        # Create test tensor
        test_tensor = torch.tensor([test_index], device=device, dtype=torch.long)
        
        result = {
            'test_index': test_index,
            'num_embeddings': emb.num_embeddings,
            'error_raised': False,
            'error_type': None,
            'error_message': None,
            'output': None,
            'output_is_zero': None,
        }
        
        try:
            output = emb(test_tensor)
            result['output'] = output
            result['output_is_zero'] = torch.allclose(output, torch.zeros_like(output))
            result['error_raised'] = False
        except IndexError as e:
            result['error_raised'] = True
            result['error_type'] = 'IndexError'
            result['error_message'] = str(e)
        except Exception as e:
            result['error_raised'] = True
            result['error_type'] = type(e).__name__
            result['error_message'] = str(e)
        
        return result
