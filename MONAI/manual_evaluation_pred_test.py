import glob, os, nibabel as nib
import numpy as np
from monai.metrics import DiceMetric
import torch

pred_dir = r"C:\ruta\a\auto3dseg_workdir\ensemble_output"
gt_dir   = r"C:\Users\Asus\Documents\TFG\Data\kits_unificado\labelsTs"

dice_metric = DiceMetric(include_background=False, reduction="mean")

scores = []
for pred_path in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
    case_name = os.path.basename(pred_path)
    gt_path   = os.path.join(gt_dir, case_name)
    if not os.path.exists(gt_path):
        print(f"GT no encontrado para {case_name}, se omite.")
        continue

    pred = nib.load(pred_path).get_fdata().astype(np.int16)
    gt   = nib.load(gt_path).get_fdata().astype(np.int16)

    # convertir a tensores (N,C,*spatial) -> one‑hot multi‑clase
    num_classes = 4  # fondo + 3 clases
    pred_tensor = torch.from_numpy(
        (np.arange(num_classes)[:, None, None, None] == pred).astype(np.uint8)
    )
    gt_tensor   = torch.from_numpy(
        (np.arange(num_classes)[:, None, None, None] == gt).astype(np.uint8)
    )

    dice = dice_metric(pred_tensor, gt_tensor).item()  # Dice medio de las clases
    scores.append((case_name, dice))
    print(f"{case_name}: Dice = {dice:.4f}")

mean_dice = np.mean([s[1] for s in scores])
print(f"\nDice medio en hold‑out = {mean_dice:.4f}")
