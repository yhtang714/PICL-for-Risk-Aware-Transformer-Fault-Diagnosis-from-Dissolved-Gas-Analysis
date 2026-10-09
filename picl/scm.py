from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn

_LOG_2PI = math.log(2.0 * math.pi)


class LinearGaussianSCM(nn.Module):

    def __init__(self, n_vars: int, n_sources: int, init_log_var: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.n_sources = n_sources
        self.log_sigma2 = nn.Parameter(torch.full((n_sources, n_vars), float(init_log_var)))
        self.mu = nn.Parameter(torch.zeros(n_vars))

    def noise_variance(self, source_idx: int):
        if source_idx < 0 or source_idx >= self.n_sources:
            return torch.exp(self.log_sigma2.mean(dim=0)).clamp(min=1e-06)
        return torch.exp(self.log_sigma2[source_idx]).clamp(min=1e-06)

    @staticmethod
    def _solve_identity(W_eff):
        n = W_eff.shape[0]
        I = torch.eye(n, dtype=W_eff.dtype, device=W_eff.device)
        try:
            return torch.linalg.solve(I - W_eff, I)
        except RuntimeError:
            return torch.linalg.solve(I - W_eff + 1e-06 * I, I)

    @staticmethod
    def covariance(W_eff, sigma2):
        T = LinearGaussianSCM._solve_identity(W_eff)
        return T.T @ torch.diag(sigma2) @ T

    def initialize_mu_from_data(self, data, miss_mask=None):
        with torch.no_grad():
            if miss_mask is None:
                self.mu.copy_(data.mean(dim=0))
            else:
                obs = (~miss_mask).float()
                num = (data * obs).sum(dim=0)
                denom = obs.sum(dim=0).clamp(min=1.0)
                self.mu.copy_(num / denom)

    def log_likelihood(self, data, sources, miss_mask, W_eff, sample_weights: Optional[torch.Tensor] = None):
        device = data.device
        centred = data - self.mu.unsqueeze(0)
        total = torch.zeros((), dtype=centred.dtype, device=device)
        for s in torch.unique(sources):
            s_int = int(s.item())
            sel = sources == s
            sub_w = sample_weights[sel] if sample_weights is not None else None
            if sub_w is not None and sub_w.sum().item() == 0:
                continue
            sigma2 = self.noise_variance(s_int)
            Sigma = self.covariance(W_eff, sigma2)
            sub_data = centred[sel]
            sub_miss = miss_mask[sel]
            pattern = sub_miss.to(torch.int8)
            uniq_rows, inverse = torch.unique(pattern, dim=0, return_inverse=True)
            for grp_i in range(uniq_rows.shape[0]):
                mask_row = uniq_rows[grp_i].to(torch.bool)
                observed = ~mask_row
                if not observed.any():
                    continue
                rows = inverse == grp_i
                xs = sub_data[rows][:, observed]
                Sigma_oo = Sigma[observed][:, observed]
                Sigma_oo = Sigma_oo + 1e-06 * torch.eye(Sigma_oo.shape[0], dtype=Sigma_oo.dtype, device=Sigma_oo.device)
                L = torch.linalg.cholesky(Sigma_oo)
                z = torch.linalg.solve_triangular(L, xs.T, upper=False)
                quad = (z * z).sum(dim=0)
                logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
                d_obs = xs.shape[1]
                ll = -0.5 * (quad + logdet + d_obs * _LOG_2PI)
                if sub_w is not None:
                    ll = ll * sub_w[rows]
                total = total + ll.sum()
        return total

    def impute_conditional_mean(self, data, sources, miss_mask, W_eff):
        out = data.clone()
        n_vars = data.shape[1]
        n_faults = n_vars - 5
        centred = data - self.mu.unsqueeze(0)
        for s in torch.unique(sources):
            s_int = int(s.item())
            sel = sources == s
            sub_data = centred[sel]
            sub_miss = miss_mask[sel]
            sigma2 = self.noise_variance(s_int)
            Sigma = self.covariance(W_eff, sigma2)
            uniq_rows, inverse = torch.unique(sub_miss.to(torch.int8), dim=0, return_inverse=True)
            for grp_i in range(uniq_rows.shape[0]):
                mask_row = uniq_rows[grp_i].to(torch.bool)
                if not mask_row.any():
                    continue
                observed = ~mask_row
                rows = inverse == grp_i
                xs = sub_data[rows][:, observed]
                Sigma_oo = Sigma[observed][:, observed]
                Sigma_mo = Sigma[mask_row][:, observed]
                Sigma_oo = Sigma_oo + 1e-06 * torch.eye(Sigma_oo.shape[0], dtype=Sigma_oo.dtype, device=Sigma_oo.device)
                cond_mean_centred = xs @ torch.linalg.solve(Sigma_oo, Sigma_mo.T)
                mu_miss = self.mu[mask_row]
                cond_mean = cond_mean_centred + mu_miss.unsqueeze(0)
                abs_rows = torch.nonzero(sel, as_tuple=True)[0][rows]
                for j_local, j_global in enumerate(mask_row.nonzero(as_tuple=True)[0].tolist()):
                    out[abs_rows, j_global] = cond_mean[:, j_local]
        gas_block = out[:, n_faults:]
        is_simplex = torch.allclose(
            data[:, n_faults:].sum(dim=1), torch.ones(data.shape[0], dtype=data.dtype, device=data.device), atol=0.001
        )
        if is_simplex:
            gas_block = gas_block.clamp(min=0.0)
            row_sum = gas_block.sum(dim=1, keepdim=True)
            row_sum = torch.where(row_sum <= 0, torch.ones_like(row_sum), row_sum)
            out[:, n_faults:] = gas_block / row_sum
        return out

    def intervene_on_faults(self, W_eff, f_int, n_faults):
        n_vars = W_eff.shape[0]
        W_int = W_eff.clone()
        W_int[:, :n_faults] = 0.0
        I = torch.eye(n_vars, dtype=W_eff.dtype, device=W_eff.device)
        T = torch.linalg.solve(I - W_int, I)
        mu_f = self.mu[:n_faults].unsqueeze(0)
        mu_y = self.mu[n_faults:].unsqueeze(0)
        f_centred = f_int - mu_f
        T_fy = T[:n_faults, n_faults:]
        return f_centred @ T_fy + mu_y
