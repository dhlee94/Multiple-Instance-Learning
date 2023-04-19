import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
import logging
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(target.size(0), -1)
            target = target.view(-1).contiguous()            
        #target = target.view(-1,1)
        #logpt = F.log_softmax(input, dim=-1)
        logpt = F.logsigmoid(input)
        target = target.type(torch.long)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
class FocalBinaryTverskyLoss(Function):

    @staticmethod
    def forward(ctx, input, target):
        _alpha = 0.5
        _beta = 0.5
        _gamma = 1.0
        _epsilon = 1e-6
        _reduction = 'mean'

        batch_size = input.size(0)
        _, input_label = input.max(1)

        input_label = input_label.float()
        target_label = target.float()

        ctx.save_for_backward(input, target_label)

        input_label = input_label.view(batch_size, -1)
        target_label = target_label.view(batch_size, -1)

        ctx.P_G = torch.sum(input_label * target_label, 1)  # TP
        ctx.P_NG = torch.sum(input_label * (1 - target_label), 1)  # FP
        ctx.NP_G = torch.sum((1 - input_label) * target_label, 1)  # FN

        index = ctx.P_G / (ctx.P_G + _alpha * ctx.P_NG + _beta * ctx.NP_G + _epsilon)
        loss = torch.pow((1 - index), 1 / _gamma)
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if _reduction == 'none':
            loss = loss
        elif _reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

    # @staticmethod
    def backward(ctx, grad_out):
        """
        :param ctx:
        :param grad_out:
        :return:
        d_loss/dT_loss=(1/gamma)*(T_loss)**(1/gamma-1)
        (dT_loss/d_P1)  = 2*P_G*[G*(P_G+alpha*P_NG+beta*NP_G)-(G+alpha*NG)]/[(P_G+alpha*P_NG+beta*NP_G)**2]
                        = 2*P_G
        (dT_loss/d_p0)=
        """
        _alpha = 0.5
        _beta = 0.5
        _gamma = 1.0
        _reduction = 'mean'
        _epsilon = 1e-6

        inputs, target = ctx.saved_tensors
        inputs = inputs.float()
        target = target.float()
        batch_size = inputs.size(0)
        sum = ctx.P_G + _alpha * ctx.P_NG + _beta * ctx.NP_G + _epsilon
        P_G = ctx.P_G.view(batch_size, 1, 1, 1, 1)
        if inputs.dim() == 5:
            sum = sum.view(batch_size, 1, 1, 1, 1)
        elif inputs.dim() == 4:
            sum = sum.view(batch_size, 1, 1, 1)
            P_G = ctx.P_G.view(batch_size, 1, 1, 1)
        sub = (_alpha * (1 - target) + target) * P_G

        dL_dT = (1 / _gamma) * torch.pow((P_G / sum), (1 / _gamma - 1))
        dT_dp0 = -2 * (target / sum - sub / sum / sum)
        dL_dp0 = dL_dT * dT_dp0

        dT_dp1 = _beta * (1 - target) * P_G / sum / sum
        dL_dp1 = dL_dT * dT_dp1
        grad_input = torch.cat((dL_dp1, dL_dp0), dim=1)
        # grad_input = torch.cat((grad_out.item() * dL_dp0, dL_dp0 * grad_out.item()), dim=1)
        return grad_input, None

class BinaryTverskyLossV2(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean'):
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative. Larger beta weigh recall higher
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLossV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = 10
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target, mask=None):
        batch_size = output.size(0)
        bg_target = 1 - target
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)
            bg_target = bg_target.float().mul(valid_mask)

        output = torch.sigmoid(output).view(batch_size, -1)
        target = target.view(batch_size, -1)
        bg_target = bg_target.view(batch_size, -1)

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * bg_target, 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, weights=None):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`

        """
        super(MultiTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights

    def forward(self, inputs, targets):

        num_class = inputs.size(1)
        weight_losses = 0.0
        if self.weights is not None:
            assert len(self.weights) == num_class, 'number of classes should be equal to length of weights '
            weights = self.weights
        else:
            weights = [1.0 / num_class] * num_class
        input_slices = torch.split(inputs, [1] * num_class, dim=1)
        for idx in range(num_class):
            input_idx = input_slices[idx]
            input_idx = torch.cat((1 - input_idx, input_idx), dim=1)
            target_idx = (targets == idx) * 1
            loss_func = FocalBinaryTverskyLoss(self.alpha, self.beta, self.gamma)
            loss_idx = loss_func.apply(input_idx, target_idx)
            weight_losses += loss_idx * weights[idx]
        # loss = torch.Tensor(weight_losses)
        # loss = loss.to(inputs.device)
        # loss = torch.sum(loss)
        return weight_losses

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def forward(self, score, target):
        loss = self.criterion(score, target[:].long())

        return loss

class DistanceLoss(nn.Module):
    def __init__(self, n_classes, softmax=False, loss_function=nn.BCELoss, sigmoid=False, weight=[1., 1., 1.]):
        super(DistanceLoss, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax
        self.sigmoid = sigmoid
        weight = torch.Tensor(weight)
        self.loss = loss_function(weight=weight)

    def forward(self, inputs, target):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        elif self.sigmoid:
            inputs = torch.sigmoid(inputs)
        target = target.long()
        loss = self.loss(inputs, target)
        return loss
    
class MultiChannelLoss(nn.Module):
    def __init__(self):
        super(MultiChannelLoss, self).__init__()
        self.CrossEntropy = nn.BCEWithLogitsLoss()
    def forward(self, scores, targets):
        loss = 0
        for score, target in zip(scores.chunk(scores.size(1), dim=1), targets.chunk(targets.size(1), dim=1)):
            loss += self.CrossEntropy(score, target)
        return loss / scores.size(1)

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
                    (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])

class Weighted_L1_Loss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super(Weighted_L1_Loss, self).__init__()
        self.weights = weights
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, input, target):
        loss = 0
        for i in range(target.shape[1]):
            loss += self.criterion(input[:,i,:,:], target[:,i,:,:]) * self.weights[i]

        return loss        

class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.84,
                 compensation=200.0,
                 num_classes=7,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        self.num_classes = num_classes
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((num_classes*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            for j in range(num_classes):
                g_masks[num_classes*idx+j, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=self.num_classes, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=self.num_classes, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=self.num_classes, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=self.num_classes, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=self.num_classes, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = 1
        for i in range(self.num_classes):
            lM *= l[:, -i-1, :, :]
        # lM = l[:, -1, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.num_classes, length=self.num_classes),
                               groups=self.num_classes, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()

class SelfNCELoss(nn.Module):
    def __init__(self, batch_size=1, num_argu=2, device=None):
        super(SelfNCELoss, self).__init__()
        self.batch_size = batch_size
        self.num_argu = num_argu
        self.device = device
        self.criterison = nn.CrossEntropyLoss()
    def forward(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.num_argu)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).type(torch.cuda.LongTensor)

        logits = logits / 0.5
        
        loss = self.criterison(logits, labels)
        return loss
    
class BCEWithMSELoss(nn.Module):
    def __init__(self, weights, bce_weight, reduction='mean'):
        super(BCEWithMSELoss, self).__init__()
        self.weights = weights
        self.regression_loss = nn.L1Loss(reduction=reduction)
        self.cross_entropy = nn.BCEWithLogitsLoss(weight=bce_weight)
    def forward(self, input, target):
        entropy_loss = self.cross_entropy(input, target)
        input = torch.sigmoid(input)
        regression_loss = self.regression_loss(input, target)
        loss = self.weights*(entropy_loss) + (1-self.weights)*regression_loss
        return loss   

class EntropyWithDistanceLoss(nn.Module):
    def __init__(self, weights, entropy_weight, reduction='mean', sigmoid=False):
        super(EntropyWithDistanceLoss, self).__init__()
        self.weights = weights
        self.sigmoid = sigmoid
        self.regression_loss = nn.L1Loss(reduction=reduction)
        self.cross_entropy = nn.CrossEntropyLoss(weight=entropy_weight)
    def forward(self, input, target):
        entropy_loss = self.cross_entropy(input, target)
        input = torch.sigmoid(input) if self.sigmoid else torch.softmax(input, dim=-1)
        regression_loss = self.regression_loss(input, target)
        loss = self.weights*(entropy_loss) + (1-self.weights)*regression_loss
        return loss   

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha*(1-pt)**self.gamma*BCE_loss
        if self.reduction=='mean':
            return F_loss.mean()
        elif self.reduction=='sum':
            return F_loss.sum()