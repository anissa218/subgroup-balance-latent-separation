import torch

def sep_surrogate(z: torch.Tensor, a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = a.view(-1)
    z0 = z[a == 0]
    z1 = z[a == 1]

    if z0.shape[0] < 2 or z1.shape[0] < 2:
        return z.new_tensor(0.0)

    mu0 = z0.mean(dim=0)
    mu1 = z1.mean(dim=0)

    z_pooled = torch.cat([z0, z1], dim=0)
    var = z_pooled.var(dim=0, unbiased=False) + eps
    inv_var = 1.0 / var

    diff = mu1 - mu0
    delta2 = (diff * inv_var * diff).sum()
    return delta2


def sep_surrogate_conditional(
    z: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    eps: float = 1e-6,
    min_per_cell: int = 2,
    equal_weight_labels: bool = True,
) -> torch.Tensor:
    z = z.view(z.size(0), -1)
    y = y.view(-1)
    a = a.view(-1)

    unique_y = torch.unique(y)
    losses = []
    weights = []

    for y_val in unique_y:
        mask_y = y == y_val
        if mask_y.sum() < 2 * min_per_cell:
            continue

        z_y = z[mask_y]
        a_y = a[mask_y]

        loss_y = sep_surrogate(z_y, a_y, eps=eps)
        if loss_y.item() == 0.0:
            continue

        losses.append(loss_y)

        if equal_weight_labels:
            weights.append(z.new_tensor(1.0))
        else:
            weights.append(mask_y.float().mean())

    if not losses:
        return z.new_tensor(0.0)

    weights = torch.stack(weights)
    weights = weights / weights.sum()
    losses = torch.stack(losses)

    return (weights * losses).sum()
