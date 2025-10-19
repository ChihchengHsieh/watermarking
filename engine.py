import tqdm
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------- training / eval loops ----------------
def train_epoch(model, loader, opt, crit, device):
    model.train()
    running_loss = 0.0
    for X, y in tqdm(loader, desc="train", leave=False):
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)


def eval_model(model, loader, crit, device):
    model.eval()
    trues, preds, probs = [], [], []
    running_loss = 0.0
    with torch.no_grad():
        for X, y in tqdm(loader, desc="eval", leave=False):
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            loss = crit(out, y)
            p = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            pred = (p > 0.5).astype(int).tolist()
            probs.extend(p.tolist())
            preds.extend(pred)
            trues.extend(y.cpu().numpy().tolist())
            running_loss += loss.item() * X.size(0)

    acc = accuracy_score(trues, preds)
    try:
        auc = roc_auc_score(trues, probs)
    except Exception:
        auc = float("nan")
    return acc, auc, (running_loss / len(loader.dataset))

def safe_eval_call(model, loader):
    """
    Call eval_model and handle two possible return signatures:
      (acc, auc) or (acc, auc, loss)
    Returns (acc, auc, loss_or_nan)
    """
    res = eval_model(model, loader)
    if isinstance(res, (list, tuple)):
        if len(res) == 3:
            return float(res[0]), float(res[1]), float(res[2])
        elif len(res) == 2:
            return float(res[0]), float(res[1]), float("nan")
    # fallback
    return float(res), float("nan"), float("nan")
