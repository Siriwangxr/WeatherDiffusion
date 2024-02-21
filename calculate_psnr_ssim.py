import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim

# Sample script to calculate PSNR and SSIM metrics from saved images in two directories
# using calculate_psnr and calculate_ssim functions from: https://github.com/JingyunLiang/SwinIR

data_name = 'snow100k'  # 'raindrop', 'snow100k', 'outdoorrain'

if data_name == 'snow100k':
    gt_path = './data/snow100k/Snow100K-L/gt/'
    results_path = './results/images/AllWeather/snow100k_L_500/'
elif data_name == 'outdoorrain':
    gt_path = './data/outdoorrain/test1/gt'
    results_path = './results/images/AllWeather/outdoorrain/'
else: # raindrop
    gt_path = './data/raindrop/rain_drop_test/gt'
    results_path = './results/images/AllWeather/raindrop/'


imgsName = sorted(os.listdir(results_path))
# gtsName = sorted(os.listdir(gt_path))
# assert len(imgsName) == len(gtsName)

cumulative_psnr, cumulative_ssim = 0, 0
for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    if data_name == 'snow100k':
        gtsName = imgsName[i].replace("['", "").replace("']_output.png", ".jpg")
    elif data_name == 'outdoorrain':
        gtsName = imgsName[i].split('_')[0]+"_"+imgsName[i].split('_')[1]+".png"
        gtsName = gtsName.replace("['", "")
    elif data_name == 'raindrop':
        gtsName = imgsName[i].replace("['", "").replace("']_output.png", ".jpg")
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName), cv2.IMREAD_COLOR)

    wd_new, ht_new, _ = res.shape
    wd_new = (16 * (wd_new // 16))
    ht_new = (16 * (ht_new // 16))
    res = cv2.resize(res, (wd_new, ht_new))
    gt = cv2.resize(gt, (wd_new, ht_new))

    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)))
print(results_path)
