import torch
import torch.nn as nn

from vit import vit_small, vit_tiny
from utils import init_linear_module


class DINOHead(nn.Module):
    def __init__(self, input_dim, feat_dim, hidden_dim=2048,
                 bottleneck_dim=256, norm_last_layer=True):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, bottleneck_dim)

        self.mlp = nn.Sequential(
            self.linear1, nn.GELU(), self.linear2, 
            nn.GELU(), self.linear3
        )

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, feat_dim, bias=False)
        )

        # 参数初始化
        init_linear_module(self.linear1)
        init_linear_module(self.linear2)
        init_linear_module(self.linear3)
        init_linear_module(self.last_layer)
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad_(False)

    def forward(self, x):
        x = self.mlp(x)

        # 这一步是为了让MLP层数增大时增加训练的稳定性
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)

        return x


class DINOLoss(nn.Module):
    def __init__(self, feat_dim, momentum_center=0.9):
        super().__init__()
        self.momentum_center = momentum_center
        self.register_buffer("center", torch.zeros(1,feat_dim))


    def forward(self, student_outs, teacher_outs, 
                temp_student, temp_teacher):
        """
        Parameters
        ----------
        student_outs : List[Tensor]. 共N+2个List元素, 2代表
            local crops的个数, N代表global crops的个数, 每个Tensor
            是(m,feat_dim)维的.

        teacher_outs : List[Tensor]. 共2个List元素, 2是global crops
            的个数. 每个Tensor是(m,feat_dim)维的.

        temp_student : student的sharpening的参数.

        temp_teacher : 同上.
        """
        # 首先将student和teacher合并起来方便计算
        assert isinstance(student_outs, (tuple, list))
        assert isinstance(teacher_outs, (tuple, list))
        num_crops_std = len(student_outs)
        num_crops_tea = len(teacher_outs)
        student_outs = torch.cat(student_outs, dim=0)
        teacher_outs = torch.cat(teacher_outs, dim=0)
        assert teacher_outs.requires_grad == False

        # 首先对teacher进行centering和sharpening
        new_center = teacher_outs.mean(dim=0, keepdim=True)
        teacher_outs = teacher_outs.sub_(self.center).div_(temp_teacher).softmax(dim=-1)

        # 之后对student进行sharpening
        student_outs = torch.log_softmax(student_outs / temp_student, dim=-1)

        # 将student和teacher拆分为chunk
        student_outs = student_outs.chunk(num_crops_std)
        teacher_outs = teacher_outs.chunk(num_crops_tea)

        # 计算loss
        loss = 0
        for i, std_out in enumerate(student_outs):
            for j, tea_out in enumerate(teacher_outs):
                if i == j: continue
                assert std_out.ndim == 2 and tea_out.ndim == 2
                assert tea_out.min() >= 0 and std_out.max() <= 0
                cross_entropy = tea_out * std_out
                loss -= cross_entropy.sum(dim=-1).mean()
        loss /= num_crops_std * num_crops_tea

        # 更新center
        self.center = (
            self.momentum_center * self.center 
            + (1 - self.momentum_center) * new_center
        )

        return loss


class DINOFeatExtractor(nn.Module):
    def __init__(self, feat_dim, drop_path, norm_last_layer):
        """
        Parameters
        ----------
        feat_dim : Int. 最后输出的特征的维度

        drop_path : Float(0~1). Transformer中drop_path的控制比率
        """
        super().__init__()
        self.backbone = vit_tiny(drop_path)
        self.dinohead = DINOHead(input_dim=self.backbone.embed_dim,
                                 feat_dim=feat_dim, 
                                 norm_last_layer=norm_last_layer)
        

    def forward(self, images):
        """
        Parameters
        ----------
        images : List[Tensor]. 对于 student network, 有 N+2 个元素, 
            前两个元素是 global crops, 后 N 个元素是 local crops. 
            global crops 的图片维度是 m,3,224,224. local crops 的图片维度是
            m,3,96,96.

        Return
        ------
        features : Tensor. (m, feat_dim)
        """
        assert isinstance(images, list)

        # 将images按照尺寸进行分类
        self.indices = self._check(images)

        # 按照chunk进行前向传播
        start = self.indices[0]
        feats = []
        for end in self.indices[1:]:
            chunk = images[start: end]
            chunk = torch.cat(chunk, dim=0)
            assert chunk.ndim == 4  # 图片应该是4维的m,c,h,w
            chunk = self.backbone(chunk)
            feats.append(chunk)
            start = end
        
        return self.dinohead(torch.cat(feats, dim=0)).chunk(len(images))


    def _check(self, images):
        indices = [0,]
        for i, img in enumerate(images[1:], 1):
            that = images[indices[-1]]
            cond1 = img.shape[-1] == that.shape[-1]
            cond2 = img.shape[-2] == that.shape[-2]
            if cond1 and cond2:
                continue
            else:
                indices.append(i)
        indices.append(len(images))
        return indices


class DINO(nn.Module):
    def __init__(self, feat_dim, drop_path=0):
        """
        Parameters
        ----------
        feat_dim : 最终提取出来的特征的维度

        drop_path : 在Transfomer中采用的drop_path比率 
        """
        super().__init__()
        self.feat_dim = feat_dim

        self.student = DINOFeatExtractor(feat_dim=feat_dim, drop_path=drop_path,
                                         norm_last_layer=False)
        self.teacher = DINOFeatExtractor(feat_dim=feat_dim, drop_path=0,
                                         norm_last_layer=True)
        self.teacher.load_state_dict(self.student.state_dict())
        self.dino_loss = DINOLoss(feat_dim=feat_dim)

    def param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.student.named_parameters():
            if not param.requires_grad: continue
            if name.endswith(".bias") or param.ndim == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized,
                                          'weight_decay': 0}]


    def forward(self, images, temp_student, temp_teacher):
        """
        Parameters
        ----------
        images : List[Tensor].每个Tensor是一个m,3,h,w的张量, List中前两个元素代表
            global crops, 后N个元素代表 local crops

        temp_student : 学生的sharpening的温度

        temp_teacher : 教师的sharpending的温度

        Attributes
        ----------
        student_outs : List[Tensor]. List的长度等于local crops的数目加上global crops
            的数目, 每一个Tensor的维度是(m,feat_dim)

        teacher_outs : List[Tensor]. List的长度等于global crops的数目, 每一个Tensor的
            维度是(m,feat_dim)
        """
        assert images[0].ndim == 4

        # 计算学生网络和教师网络的输出
        student_outs = self.student(images) # List[Tensor]
        with torch.no_grad():
            teacher_outs = self.teacher(images[:2]) # List[Tensor]
        assert student_outs[0].ndim == 2 and teacher_outs[0].ndim == 2

        return self.dino_loss(student_outs, teacher_outs, 
                              temp_student, temp_teacher)
