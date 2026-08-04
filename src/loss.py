import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss - هسته اصلی SimCLR.
    z1, z2: (B, D) بردارهای projection برای دو view مختلف از همون batch تصاویر
    """
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # (2B, D)
    similarity_matrix = torch.matmul(representations, representations.T)  # (2B, 2B)
    similarity_matrix = similarity_matrix / temperature

    # ماسک برای حذف شباهت یک نمونه با خودش
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity_matrix.masked_fill_(mask, -1e9)

    # positive pair برای نمونه i در بازه [0, B) با نمونه i+B است و برعکس
    positive_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z1.device)

    loss = F.cross_entropy(similarity_matrix, positive_indices)
    return loss
