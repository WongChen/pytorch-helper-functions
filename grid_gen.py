# -----------------------------------------------------------------------------------
# generate grid from flow field, used for bilinear warping 
# support for 2D and 3D images
# written by wong
# -----------------------------------------------------------------------------------

import torch

def gridGenerator(offeset):
    '''generate a grid for bilinear sample
    Args: 
        offeset: [n, h, w, 2]
    '''
    n, h, w = offeset.shape[:-1]
    x_grid = torch.linspace(-1.0, 1.0, w).view(1, 1, w).expand(1, h, w)
    y_grid = torch.linspace(-1.0, 1.0, h).view(1, h, 1).expand(1, h, w)
    base_x = 2. / (w-1)
    base_y = 2. / (h-1)
    x_grid = x_grid + offeset[:, :, :, 0] * base_x
    y_grid = y_grid + offeset[:, :, :, 1] * base_y
    batch_pixel_coord = torch.stack((x_grid, y_grid), 1).permute(0, 2, 3, 1)
    return batch_pixel_coord

def gridGenerator3D(offeset):
    '''generate a grid for bilinear sample
    Args: 
        offeset: [n, h, w, d, 3]
    '''
    _, h, w, d = offeset.shape[:-1]
    x_grid = torch.linspace(-1.0, 1.0, d).view(1, 1, 1, d).expand(1, h, w, d)
    y_grid = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(1, h, w, d)
    z_grid = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(1, h, w, d)
    base_x = 2. / (w-1)
    base_y = 2. / (h-1)
    base_z = 2. / (d-1)
    x_grid = x_grid + offeset[:, :, :, :, 0] * base_x
    y_grid = y_grid + offeset[:, :, :, :, 1] * base_y
    z_grid = z_grid + offeset[:, :, :, :, 2] * base_z
    batch_pixel_coord = torch.stack((x_grid, y_grid, z_grid), 1).permute(0, 2, 3, 4, 1)
    return batch_pixel_coord
