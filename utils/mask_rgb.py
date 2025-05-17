    
import numpy as np
import colorsys

def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h,s,v), axis=-1)
    return hsv

def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb
  
def mask_rgb(masks, colors=None):
  """ masks in random rgb colors

  Parameters
  ----------------

  masks: int, 2D array
      masks where 0=NO masks; 1,2,...=mask labels

  colors: int, 2D array (optional, default None)
      size [nmasks x 3], each entry is a color in 0-255 range

  Returns
  ----------------

  RGB: uint8, 3D array
      array of masks overlaid on grayscale image

  """
  if colors is not None:
      if colors.max()>1:
          colors = np.float32(colors)
          colors /= 255
      colors = rgb_to_hsv(colors)
  
  HSV = np.zeros((masks.shape[0], masks.shape[1], 3), np.float32)
  HSV[:,:,2] = 1.0
  for n in range(int(masks.max())):
      ipix = (masks==n+1).nonzero()
      if colors is None:
          HSV[ipix[0],ipix[1],0] = np.random.rand()
      else:
          HSV[ipix[0],ipix[1],0] = colors[n,0]
      HSV[ipix[0],ipix[1],1] = np.random.rand()*0.5+0.5
      HSV[ipix[0],ipix[1],2] = np.random.rand()*0.5+0.5
  RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
  return RGB