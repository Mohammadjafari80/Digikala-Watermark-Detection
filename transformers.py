import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transforms = A.Compose(
    [
        A.Resize(width=224, height=224),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.7),
        A.MultiplicativeNoise(multiplier=[0.5, 2], per_channel=True, p=0.7),
        A.OpticalDistortion(p=0.9),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.8),
        A.ColorJitter(p=0.7),
        A.RandomToneCurve(p=0.7),
        A.RandomGamma(p=0.7),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.2, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), p=0.7),
        ToTensorV2(),
    ]
)

train_transforms_simple = A.Compose(
    [
        A.Resize(width=224, height=224),
        ToTensorV2(),
    ]
)