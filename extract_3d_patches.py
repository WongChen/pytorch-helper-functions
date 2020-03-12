import torch

def extract_3d_patches_from_tensor(prob):
    '''generate 26 neiborhood for prob>0.5, for simplicity, no dimensions for batch size and channel
    prob: network's outout probability of positive class, shape [H, W, D]
    Return:
        [26, H, W, D]
    '''
    h, w, d = prob.shape[-3:]
    prob_pad = torch.nn.functional.pad(prob, (1, 1, 1, 1, 1, 1)).view(h+2, w+2, d+2) # pad 1 for 3*3*3 patch
    prob_unfold = prob_pad.unfold(2, 3, 1).unfold(1, 3, 1).unfold(0, 3, 1).permute(0, 1, 2, 5, 4, 3).contiguous().view(h, w, d, 27)
    zeros_tensor = torch.zeros_like(prob_unfold) 
    prob = prob.view(h, w, d).unsqueeze(-1).expand(prob_unfold.shape) # [h, w, d, 27]
    rv = torch.where(prob>0.5, prob_unfold, zeros_tensor)
    index = torch.tensor(list(range(0, 13)) + list(range(14, 27)))
    rv = rv[:, :, :, index]
    return rv.permute(3, 0, 1, 2)


# test example
inp = torch.arange(0, 27.).reshape(3, 3, 3)
print(extract_3d_patches_from_tensor(inp).shape)
# print(inp[0:2, 0:2, 0:2])

