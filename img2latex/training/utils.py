import torch


def edit_distance(token1: torch.Tensor, token2: torch.Tensor) -> int:
    len1 = token1.size(0)
    distances = torch.zeros((len1 + 1, len1 + 1))

    distances[:, 0] = torch.arange(len1 + 1)
    distances[0, :] = torch.arange(len1 + 1)

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len1 + 1):
        for t2 in range(1, len1 + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len1][len1].item()


def make_training_report(
    loss: float, levenstein: float, accuracy: float, lr=float
) -> str:
    return f"Loss={loss:.4f} | Acc={accuracy:.4f} | ED={levenstein:.1f} | LR={lr:.1e}"
