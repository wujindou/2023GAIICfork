
import torch
import torch.nn as nn
from megatron.utils import average_losses_across_data_parallel_group
import torch.nn.functional as F
def CE(output, target):
    '''
    Output: (B,L,C)。未经过softmax的logits
    Target: (B,L)
    '''
    output = output.reshape(-1, output.shape[-1])  # (*,C)
    target = target.reshape(-1).long()  # (*)
    return nn.CrossEntropyLoss()(output, target) #默认size_average=True，会把B*L所有词loss平均

def DAE_loss(loss_mask, lm_loss_):
    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group(
        [lm_loss])
    return averaged_losses[0]

    # if sop_logits is not None:
    #     sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
    #                                sentence_order.view(-1),
    #                                ignore_index=-1)
    #     sop_loss = sop_loss.float()
    #     loss = lm_loss + sop_loss
    #     averaged_losses = average_losses_across_data_parallel_group(
    #         [lm_loss, sop_loss])
    #     return loss, {'lm loss': averaged_losses[0],
    #                   'sop loss': averaged_losses[1]}
    #
    # else:
    #     loss = lm_loss
    #     averaged_losses = average_losses_across_data_parallel_group(
    #         [lm_loss])
    #     return loss, {'lm loss': averaged_losses[0]}