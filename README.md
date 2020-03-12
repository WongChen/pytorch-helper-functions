# pytorch helper functions

## Grid generator

Generate grid from flow field, used for bilinear warping. See [grid_gen](grid_gen.py).


## Extract patches

Extracting patches from 4D tensors can be simply implemented by [torch.nn.functional.unfold](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.unfold) but unsupported for 5D tensors. And this can be achieved by [tensor.unfold](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.unfold), which is quite different from [torch.nn.functional.unfold](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.unfold), and call it at W, H, D dimension sequentially.  See [extract_3d_patches](extract_3d_patches.py).
