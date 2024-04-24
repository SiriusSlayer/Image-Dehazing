from skimage.io import imread

import skimage.metrics as metrics

# Load the two images
image1 = imread('pred_FFA_ots/35_outdoor_hazy_FFA.png')
image2 = imread('/home/stud-1/UGP_NASA/FFA-main/net/GT/35_outdoor_GT.jpg')

# Calculate PSNR
psnr = metrics.peak_signal_noise_ratio(image1, image2)

# Calculate SSIM
ssim = metrics.structural_similarity(image1, image2, multichannel=True,win_size=3)

# Print the results
print(f"PSNR: {psnr:.2f}")
print(f"SSIM: {ssim:.2f}")