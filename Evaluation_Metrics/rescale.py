import glob
from PIL import Image
import numpy as np
import torch

test_list = glob.glob("pred/VE-LOL-L-Cap.SCIDCE/*") 
for image in test_list:
    print(image)

    name = image.split('\\')[1]

    img = Image.open(image)

    width, height = img.size

    new_height = 400
    new_size = (width, new_height)
    resized_img = img.resize(new_size)

    resized_img.save("pred/VE-LOL-L-Cap-Resized/"+name)



