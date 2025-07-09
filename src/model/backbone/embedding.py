import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class IdentityEmbedding(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        shape = list(input.shape)
        shape[-1] = self.output_size
        output = input.expand(*shape)
        return output


class FourierEmbedding(nn.Module):
    def __init__(self, input_size, output_size, max_period=10000.):
        super().__init__()
        assert output_size % 2 == 0, "Output size must be even"
        self.input_size = input_size
        self.output_size = output_size
        self.max_period = max_period

        half_dim = output_size // 2
        freq_indices = torch.arange(half_dim, dtype=torch.float32) / half_dim
        self.register_buffer('freqs', torch.exp(-math.log(max_period) * freq_indices))

    def forward(self, input):
        args = input @ self.freqs.unsqueeze(0) * 2 * math.pi
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_mode, embedding_size):
        super().__init__()
        self.embedding_mode = embedding_mode
        self.embedding_size = embedding_size
        if self.embedding_size > 0:
            if embedding_mode == 'identity':
                self.time_embedding = IdentityEmbedding(self.embedding_size)
                self.is_time = False
            elif embedding_mode == 'fourier':
                self.time_embedding = FourierEmbedding(1, self.embedding_size)
                self.is_time = True
            else:
                raise ValueError('Embedding mode {} not supported'.format(embedding_mode))
        else:
            self.time_embedding = None

    def forward(self, input):
        if self.time_embedding is not None:
            if input.dim() == 1:
                input = input.unsqueeze(-1)
            embedding = self.time_embedding(input)
        else:
            embedding = None
        return embedding


class ConditionEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_size, offset=1):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_size = embedding_size
        if embedding_size > 0:
            self.cond_embedding = nn.Embedding(num_embedding, embedding_size)
            self.is_cond = True
        else:
            self.cond_embedding = None
            self.is_cond = False
        self.offset = offset

    def forward(self, cond):
        if self.cond_embedding is not None and cond is not None:
            embedding = self.cond_embedding(cond + self.offset)
        else:
            embedding = None
        return embedding


class DataEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_size, transpose=True, *args, **kwargs):
        super().__init__()
        self.orthonormal_embedding = OrthonormalEmbedding(num_embedding, embedding_size)
        self.transpose = transpose

    def forward(self, x):
        x = x.view(*x.shape[:2], -1)
        weight = self.orthonormal_embedding.weight
        if self.transpose:
            weight = weight.T
        x = F.linear(x, weight, None)
        return x


class OrthonormalEmbedding(nn.Embedding):
    def __init__(self, num_embedding, embedding_size, requires_grad=False, num_iters=200, *args, **kwargs):
        super().__init__(num_embedding, embedding_size, *args, **kwargs)
        self.num_iters = num_iters
        if num_embedding <= embedding_size:
            with torch.no_grad():
                self.weight.data = self.gram_schmidt(self.weight.data.detach())
                # self.weight.data = torch.nn.functional.normalize(self.weight.data, p=2, dim=-1)
        else:
            self.num_iters = num_iters
            self.weight.data = self.ebv(self.weight.data, self.num_iters).detach()
        self.weight.requires_grad = requires_grad

    @staticmethod
    def gram_schmidt(vectors, eps=1e-10):
        basis = []
        for v in vectors:
            w = v.clone()
            for u in basis:
                w -= torch.dot(u, v) * u
            w_norm = torch.linalg.norm(w)
            w /= max(w_norm, eps)
            basis.append(w)
        basis = torch.stack(basis, dim=0)
        return basis

    @staticmethod
    def ebv(vectors, num_iters=200):
        vectors.requires_grad = True
        optimizer = optim.SGD([vectors], lr=1, momentum=0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters,
                                                         eta_min=0)
        for i in range(num_iters):
            norm = torch.linalg.norm(vectors, dim=-1, keepdim=True)
            norm_vectors = vectors / norm
            cosine = norm_vectors @ norm_vectors.t()
            cosine = torch.triu(cosine, diagonal=1)
            row_idx, col_idx = torch.triu_indices(*cosine.size(), offset=1)
            cosine = cosine[row_idx, col_idx]
            loss = cosine.abs().sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        with torch.no_grad():
            norm = torch.linalg.norm(vectors, dim=-1, keepdim=True)
            evd = vectors / norm
        return evd
