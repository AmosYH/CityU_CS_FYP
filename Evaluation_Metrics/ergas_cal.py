from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from PIL import Image
import numpy
import math
import glob

def test():
    gt_names = glob.glob('gt/VE-LOL-L-Cap/*.png')
    pred_names = glob.glob('pred/lime/*.png')
    avg_ERGAS = 0
    
    for i in range(len(gt_names)):

        gt_name = gt_names[i]
        gt_img = numpy.array(Image.open(gt_name),dtype=float)
        
        
        pred_name =  pred_names[i]
        pred_img = numpy.array(Image.open(pred_name),dtype=float)

        pred_img[gt_img==0] = 0

        ERGAS = ergas(pred_img,gt_img)
        avg_ERGAS += ERGAS
        print('The ERGAS value is ' + str(ERGAS) + '\n')
    
    avg_ERGAS /= len(gt_names)
    print('The Average ERGAS value is ' + str(avg_ERGAS) + '\n')


if __name__== '__main__':
    test() 
