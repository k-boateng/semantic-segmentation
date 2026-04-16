from torch import nn

def get_ce_loss(ignore_index=255):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)