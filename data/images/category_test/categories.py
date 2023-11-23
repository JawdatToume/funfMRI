import os

base = os.path.join("images","training")
print(base)

for image in os.listdir(base):
    print(image)
    folder = image[:image.find('_')]
    if not os.path.isdir(folder):
        os.mkdir(folder)
    os.system('copy %s %s' %(os.path.join(base,image), image))
