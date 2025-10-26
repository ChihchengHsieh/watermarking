from torchvision import models
import torch.nn as nn
import torch
import os

class PNSRWarpper(nn.Module):
    def __init__(self, base_model):
        super(PNSRWarpper, self).__init__()
        self.base_model = base_model
        # grab the original in_features of the fc layer
        self.fc_in_features = base_model.fc.in_features
        # modify the fc layer to accept additional pnsr input
        self.base_model.fc = nn.Identity()
        # instead of one layer, make it 2
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, pnsr):
        x = self.base_model(x)  ##
        x = torch.cat([x, pnsr.unsqueeze(1)], dim=1)
        x = self.fc(x)
        return x


# ---------------- small model helper ----------------
def make_model(in_channels, include_psnr=False):
    model = models.resnet18(pretrained=False)
    # adapt first conv
    model.conv1 = nn.Conv2d(
        in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=(model.conv1.bias is not None),
    )
    if include_psnr:
        print("Using PNSR wrapper for the model.")
        model = PNSRWarpper(model)
    else:
        print("Using standard model without PNSR.")
        model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_checkpoint(model, cp_path, device, opt=None):
    if os.path.exists(cp_path):
        print(f"Loading checkpoint: {cp_path}")
        ckpt = torch.load(cp_path, map_location=device)


        # ------------ Comment out this part when loading new pattern ------------
        # remove dict start with fc. for now.
        # for k in list(ckpt['model_state_dict'].keys()):
        #     if k.startswith("fc.") or k.startswith("conv1."):
        #         ckpt['model_state_dict'].pop(k)

        # ckpt.pop('optimizer_state_dict', None)

        # print("Remaining keys in checkpoint:")
        # print(ckpt['model_state_dict'].keys())

        # two possible styles:
        # 1) legacy single state_dict saved by torch.save(model.state_dict())
        # 2) full checkpoint dict saved with model_state_dict, optimizer_state_dict, epoch, maybe scheduler
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            if "optimizer_state_dict" in ckpt:
                try:
                    opt.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print("Warning: couldn't fully load optimizer state:", e)
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
            else:
                start_epoch = 1
            print(f"Restored model and optimizer. Resuming from epoch {start_epoch}")

            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_epoch = ckpt.get("best_epoch", None)

            return model, start_epoch, best_val_loss, best_epoch
        else:
            # assume ckpt is a plain state_dict
            try:
                model.load_state_dict(ckpt)
                print(
                    "Loaded model.state_dict() from checkpoint (optimizer state not present)."
                )
                return model, 1, float("inf"), None
            except Exception as e:
                raise RuntimeError(
                    "Checkpoint format not recognized and failed to load model:", e
                )
    else:
        print("No checkpoint found; training from scratch.")
