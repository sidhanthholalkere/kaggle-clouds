import albumentations as albu

def training1(h=320, w=640):
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(h, w)
    ]
    return albu.Compose(train_transform)


def valid1(h=320, w=640):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(h, w)
    ]
    return albu.Compose(test_transform)
