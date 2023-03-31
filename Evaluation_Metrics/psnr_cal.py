from PIL import Image
import numpy
import math
import glob

def psnr(img1, img2):
    mse = numpy.mean( ((img1 - img2)) ** 2 )
    if mse == 0:
        return 'INF'
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def test():
    gt_names = glob.glob('gt/VE-LOL-L-Cap/*.png')
    pred_names = glob.glob('pred/lime/*.png')
    avg_pnsr = 0
    
    for i in range(len(gt_names)):

        gt_name = gt_names[i]
        gt_img = numpy.array(Image.open(gt_name),dtype=float)
        
        
        pred_name =  pred_names[i]
        pred_img = numpy.array(Image.open(pred_name),dtype=float)
        
    # When calculate the PSNR:
    # 1.) The pixels in ground-truth disparity map with '0' value will be neglected.

        pred_img[gt_img==0] = 0

        peaksnr = psnr(pred_img,gt_img)
        avg_pnsr += peaksnr
        print('The Peak-SNR value is ' + str(peaksnr) + '\n')
    
    avg_pnsr /= len(gt_names)
    print('The Average Peak-SNR value is ' + str(avg_pnsr) + '\n')


if __name__== '__main__':
    test() 