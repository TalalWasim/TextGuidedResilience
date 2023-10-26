import numpy as np

d = '''gaussian_noise
	s=1: Test Loss 0.007 | Test Acc1 59.758
	s=2: Test Loss 0.009 | Test Acc1 49.326
	s=3: Test Loss 0.014 | Test Acc1 32.854
	s=4: Test Loss 0.020 | Test Acc1 17.190
	s=5: Test Loss 0.026 | Test Acc1 5.962
shot_noise
	s=1: Test Loss 0.007 | Test Acc1 57.786
	s=2: Test Loss 0.010 | Test Acc1 45.660
	s=3: Test Loss 0.015 | Test Acc1 31.068
	s=4: Test Loss 0.021 | Test Acc1 13.816
	s=5: Test Loss 0.026 | Test Acc1 6.870
impulse_noise
	s=1: Test Loss 0.009 | Test Acc1 49.678
	s=2: Test Loss 0.012 | Test Acc1 39.856
	s=3: Test Loss 0.014 | Test Acc1 31.794
	s=4: Test Loss 0.020 | Test Acc1 16.110
	s=5: Test Loss 0.026 | Test Acc1 6.156
defocus_blur
	s=1: Test Loss 0.008 | Test Acc1 54.880
	s=2: Test Loss 0.009 | Test Acc1 47.150
	s=3: Test Loss 0.013 | Test Acc1 33.078
	s=4: Test Loss 0.016 | Test Acc1 22.212
	s=5: Test Loss 0.019 | Test Acc1 14.536
glass_blur
	s=1: Test Loss 0.009 | Test Acc1 51.580
	s=2: Test Loss 0.012 | Test Acc1 37.552
	s=3: Test Loss 0.020 | Test Acc1 16.158
	s=4: Test Loss 0.022 | Test Acc1 11.876
	s=5: Test Loss 0.024 | Test Acc1 8.178
motion_blur
	s=1: Test Loss 0.006 | Test Acc1 61.814
	s=2: Test Loss 0.009 | Test Acc1 50.854
	s=3: Test Loss 0.013 | Test Acc1 33.992
	s=4: Test Loss 0.018 | Test Acc1 19.608
	s=5: Test Loss 0.021 | Test Acc1 13.094
zoom_blur
	s=1: Test Loss 0.009 | Test Acc1 50.770
	s=2: Test Loss 0.012 | Test Acc1 41.238
	s=3: Test Loss 0.013 | Test Acc1 34.150
	s=4: Test Loss 0.016 | Test Acc1 27.898
	s=5: Test Loss 0.017 | Test Acc1 22.156
snow
	s=1: Test Loss 0.009 | Test Acc1 51.296
	s=2: Test Loss 0.016 | Test Acc1 28.088
	s=3: Test Loss 0.015 | Test Acc1 31.524
	s=4: Test Loss 0.019 | Test Acc1 20.418
	s=5: Test Loss 0.021 | Test Acc1 15.188
frost
	s=1: Test Loss 0.008 | Test Acc1 56.804
	s=2: Test Loss 0.012 | Test Acc1 39.276
	s=3: Test Loss 0.016 | Test Acc1 27.756
	s=4: Test Loss 0.017 | Test Acc1 25.754
	s=5: Test Loss 0.019 | Test Acc1 19.498
fog
	s=1: Test Loss 0.007 | Test Acc1 57.708
	s=2: Test Loss 0.009 | Test Acc1 50.968
	s=3: Test Loss 0.011 | Test Acc1 41.292
	s=4: Test Loss 0.013 | Test Acc1 34.616
	s=5: Test Loss 0.019 | Test Acc1 19.174
brightness
	s=1: Test Loss 0.005 | Test Acc1 70.762
	s=2: Test Loss 0.005 | Test Acc1 68.676
	s=3: Test Loss 0.006 | Test Acc1 65.474
	s=4: Test Loss 0.007 | Test Acc1 60.460
	s=5: Test Loss 0.008 | Test Acc1 53.322
contrast
	s=1: Test Loss 0.006 | Test Acc1 61.758
	s=2: Test Loss 0.008 | Test Acc1 54.916
	s=3: Test Loss 0.011 | Test Acc1 41.600
	s=4: Test Loss 0.018 | Test Acc1 16.474
	s=5: Test Loss 0.025 | Test Acc1 3.538
elastic_transform
	s=1: Test Loss 0.006 | Test Acc1 65.346
	s=2: Test Loss 0.011 | Test Acc1 43.232
	s=3: Test Loss 0.009 | Test Acc1 52.352
	s=4: Test Loss 0.013 | Test Acc1 39.092
	s=5: Test Loss 0.022 | Test Acc1 15.608
pixelate
	s=1: Test Loss 0.006 | Test Acc1 62.960
	s=2: Test Loss 0.007 | Test Acc1 60.650
	s=3: Test Loss 0.009 | Test Acc1 49.604
	s=4: Test Loss 0.015 | Test Acc1 31.814
	s=5: Test Loss 0.019 | Test Acc1 21.192
jpeg_compression
	s=1: Test Loss 0.006 | Test Acc1 63.846
	s=2: Test Loss 0.007 | Test Acc1 60.738
	s=3: Test Loss 0.007 | Test Acc1 58.518
	s=4: Test Loss 0.009 | Test Acc1 50.156
	s=5: Test Loss 0.013 | Test Acc1 36.526'''

d = d.split('\n')
corruption_accs = dict()

for i in range(15):
    accs = []
    for j in range(6):
        if j == 0:
            name = d[i*6 + j]
        else:
            acc = float(d[i*6 + j].split(' ')[-1])/100.
            accs.append(acc)
    corruption_accs[name] = accs

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]

# for c in CORRUPTIONS:
#     print('\t'.join([c] + list(map(str, corruption_accs[c]))))

def compute_mce(corruption_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  for i in range(len(CORRUPTIONS)):
    avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
    ce = 100 * avg_err / ALEXNET_ERR[i]
    mce += ce / 15
  return mce
print('mCE (normalized by AlexNet): ', compute_mce(corruption_accs))
