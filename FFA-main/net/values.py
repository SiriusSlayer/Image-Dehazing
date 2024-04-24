import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread

folder1 = '/path/to/folder1'
folder2 = '/path/to/folder2'

# Initialize variables for average PSNR and SSIM
total_psnr = 0
total_ssim = 0
count = 0
# Iterate through the files in folder1
for filename in os.listdir(folder1):
    if filename.endswith('.jpg'):
        image1 = imread(os.path.join(folder1, filename))
        image2 = imread(os.path.join(folder2, filename))
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(image1, image2)
        # Calculate SSIM
        ssim = structural_similarity(image1, image2, multichannel=True)
        # Accumulate PSNR and SSIM values
        total_psnr += psnr
        total_ssim += ssim
        count += 1
        # Print the results for each image
        print(f"Image: {filename}")
        print(f"PSNR: {psnr:.2f}")
        print(f"SSIM: {ssim:.2f}")
        print()
# Calculate average PSNR and SSIM
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count
# Print the average results
print("Average PSNR:", avg_psnr)
print("Average SSIM:", avg_ssim)