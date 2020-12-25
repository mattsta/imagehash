import imagehash
from PIL import Image

SAVE_IMAGES = False

# Load image
full_image = Image.open("../tests/data/peppers.png")
width, height = full_image.size
# Hash it
full_hash = imagehash.crop_resistant_hash(full_image)

# Crop it
for x in range(5, 50, 5):
    start = x / 100
    end = 1 - start
    crop_img = full_image.crop(
        (start * width, start * height, end * width, end * height)
    )
    crop_hash = imagehash.crop_resistant_hash(crop_img)
    if SAVE_IMAGES:
        crop_img.save("crop_{}.png".format(str(x).zfill(2)))
    crop_diff = full_hash.hash_diff(crop_hash)
    print(
        "Cropped {}% from each side. Hash has {} matching segments with {} total hamming distance".format(
            x, crop_diff[0], crop_diff[1]
        )
    )
