import os

base = os.path.join("/Users/srinjoybhuiya/PycharmProjects/funfMRI/imagenet_kamitani_subset/training/")

import shutil

for image in os.listdir(base):
    print(image)
    cat_folder = image.split('_')[0]
    if not os.path.isdir(cat_folder):
        os.mkdir(cat_folder)
    shutil.copy(base+image, cat_folder)

