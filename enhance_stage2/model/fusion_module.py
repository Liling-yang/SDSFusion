import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numbers
from einops import rearrange

# Semantic-aware emdedding module语义信息嵌入模块
# input_S语义特征, input_R图像特征
class TransformerBlock(nn.Module):
	def __init__(self, dim_in, dim_out, ffn_expansion_factor=2.66):
		super(TransformerBlock, self).__init__()
		# self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)
		self.conv1 = nn.Conv2d(dim_in, dim_out, (1, 1))
		self.norm1 = LayerNorm(dim_out)
		self.attn = Attention(dim_out)
		self.norm2 = LayerNorm(dim_out)
		self.ffn = FeedForward(dim_out, ffn_expansion_factor)

	def forward(self, input_R, input_S):
		input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]], align_corners=False)
		input_S = self.conv1(input_S)
		input_R = self.norm1(input_R)
		input_S = self.norm1(input_S)
		input_R = input_R + self.attn(input_R, input_S)
		input_R = input_R + self.ffn(self.norm2(input_R))

		return input_R

class Restormer_CNN_block(nn.Module):
	def __init__(self,in_dim,out_dim):
		super(Restormer_CNN_block, self).__init__()
		self.embed = nn.Conv2d(in_dim, out_dim,kernel_size=3,stride=1, padding=1, bias=False)
		self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim)
		self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
		self.FFN = nn.Conv2d(out_dim*2, out_dim,kernel_size=3,stride=1, padding=1, bias=False)          
	def forward(self, x):
		x=self.embed(x)
		x1=self.GlobalFeature(x)
		x2=self.LocalFeature(x)
		out=self.FFN(torch.cat((x1,x2),1))
		return out
		# return nn.ReLU()(out)
		# return F.leaky_relu(out, negative_slope=0.2)
	
	
class Restormer_CNN_block_IR(nn.Module):
	def __init__(self,in_dim,out_dim):
		super(Restormer_CNN_block_IR, self).__init__()
		self.embed = nn.Conv2d(in_dim, out_dim,kernel_size=3,stride=1, padding=1, bias=False)
		self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim)
		self.CNN = nn.Conv2d(out_dim, out_dim,kernel_size=3,stride=1, padding=1, bias=False)
	def forward(self, x):
		x = self.embed(x)
		x1 = self.GlobalFeature(x)
		return self.CNN(x1)
		# return nn.ReLU()(self.CNN(x1))
		# return F.leaky_relu(self.CNN(x1), negative_slope=0.2)

class SEBlock(nn.Module):
	def __init__(self, channels, ratio):
		super(SEBlock, self).__init__()
		self.avg_pooling = nn.AdaptiveAvgPool2d(1)
		self.max_pooling = nn.AdaptiveMaxPool2d(1)
		self.global_pooling = self.max_pooling
		# self.global_pooling = self.avg_pooling
		self.fc_layers = nn.Sequential(
			nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
			nn.ReLU(),
			nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
		)
		self.softmax = nn.Softmax(dim=1)  # Apply softmax over the channel dimension
	 
	def forward(self, x):
		b, c, _, _ = x.shape
		v = self.global_pooling(x).view(b, c)
		v = self.fc_layers(v).view(b, c)
		v = self.softmax(v).view(b, c, 1, 1)  # Reshape to match input dimensions
		return x * v
	
class GlobalFeatureExtraction(nn.Module):
	def __init__(self,
				 dim,
				 type='base',
				 ffn_expansion_factor=1.,  
				 qkv_bias=False,):
		super(GlobalFeatureExtraction, self).__init__()
		self.norm1 = LayerNorm(dim)
		self.attn = AttentionBase(dim)
		self.norm2 = LayerNorm(dim)
		self.mlp = Mlp(in_features=dim,out_fratures=dim,ffn_expansion_factor=ffn_expansion_factor,) # FFN
	def forward(self, x):
		x = x + self.attn(self.norm1(x)) # Restormer的MDTA
		x = x + self.mlp(self.norm2(x)) # FNN+LN(类似于vision transformer,只不过多头注意力换成了Restormer中的MDTA)
		return x

class LocalFeatureExtraction(nn.Module):
	def __init__(self,
				 dim=64,
				 num_blocks=2,
				 ):
		super(LocalFeatureExtraction, self).__init__()
		self.Extraction = nn.Sequential(*[ResBlock(dim,dim) for i in range(num_blocks)])
	def forward(self, x):
		return self.Extraction(x)
	
class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
		)
	def forward(self, x):
		out = self.conv(x)
		return out+x

class AttentionBase(nn.Module):
	def __init__(self,
				 dim,   
				 num_heads=8,
				 qkv_bias=False,):
		super(AttentionBase, self).__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = nn.Parameter(torch.ones(num_heads, 1, 1)) # 注意力得分的动态缩放
		# num_heads 表示注意力机制的头数,即有多少个子空间
		# torch.ones(num_heads, 1, 1) 创建了一个(num_heads, 1, 1) 的全 1 张量
		# 设置为 nn.Parameter，表示一个可学习的参数, 在训练过程中它的值会被自动更新
		# 作用1：在计算注意力得分时, 将注意力得分除以 self.scale 进行缩放。这种缩放操作有助于稳定训练过程, 防止注意力得分过大导致梯度爆炸。
		# 作用2：self.scale 是一个可学习的参数,不同的子空间可以学习到不同的缩放因子,这有助于捕捉不同子空间的重要性。
		self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
		self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
		self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

	def forward(self, x):
		b, c, h, w = x.shape
		qkv = self.qkv2(self.qkv1(x))
		q, k, v = qkv.chunk(3, dim=1)
		q = rearrange(q, 'b (head c) h w -> b head c (h w)',
					  head=self.num_heads)
		k = rearrange(k, 'b (head c) h w -> b head c (h w)',
					  head=self.num_heads)
		v = rearrange(v, 'b (head c) h w -> b head c (h w)',
					  head=self.num_heads)
		q = torch.nn.functional.normalize(q, dim=-1) # l2-norm
		k = torch.nn.functional.normalize(k, dim=-1)
		attn = (q @ k.transpose(-2, -1)) * self.scale # 计算注意力得分
		attn = attn.softmax(dim=-1)
		out = (attn @ v) # 矩阵乘法, torch.mul是逐元素相乘
		out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
		out = self.proj(out)
		return out

class Attention(nn.Module):
	def __init__(self, dim, num_heads=2, bias=False):
		super(Attention, self).__init__()
		self.num_heads = num_heads
		self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) # 可学习的"缩放因子"

		self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
		self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias) # 深度可分离卷积
		self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
		self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
		# q本来就不用dwconv，只是normal conv
		self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

	def forward(self, x, y):
		b, c, h, w = x.shape

		kv = self.kv_dwconv(self.kv(x))
		k, v = kv.chunk(2, dim=1) # 沿着第1个维度(通道维)划分为两个等大的张量
		q = self.q_dwconv(self.q(y))

		q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
		k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
		v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

		q = torch.nn.functional.normalize(q, dim=-1) # L2范数归一化
		k = torch.nn.functional.normalize(k, dim=-1)

		attn = (q @ k.transpose(-2, -1)) * self.temperature
		attn = attn.softmax(dim=-1)

		out = (attn @ v)

		out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

		out = self.project_out(out)
		return out

class FeedForward(nn.Module):
	def __init__(self, dim, ffn_expansion_factor):
		super(FeedForward, self).__init__()
		hidden_features = int(dim * ffn_expansion_factor)
		self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)
		self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,groups=hidden_features * 2, bias=False)
		self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

	def forward(self, x):
		x = self.project_in(x)
		x1, x2 = self.dwconv(x).chunk(2, dim=1)
		x = F.gelu(x1) * x2
		x = self.project_out(x)
		return x

class Mlp(nn.Module):
	"""
	MLP as used in Vision Transformer, MLP-Mixer and related networks
	"""
	def __init__(self, 
				 in_features, 
				 out_fratures,
				 ffn_expansion_factor = 2,
				 bias = False):
		super().__init__()
		hidden_features = int(in_features*ffn_expansion_factor)

		self.project_in = nn.Conv2d(in_features, hidden_features*2, kernel_size=1, bias=bias)

		self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
								stride=1, padding=1, groups=hidden_features, bias=bias)

		self.project_out = nn.Conv2d(hidden_features, out_fratures, kernel_size=1, bias=bias)
	def forward(self, x):
		x = self.project_in(x)
		x1, x2 = self.dwconv(x).chunk(2, dim=1)
		x = F.gelu(x1) * x2
		x = self.project_out(x)
		return x
	
##########################################################################
## Layer Norm
def to_3d(x):
	return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
	return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(BiasFree_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)

		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(WithBias_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)

		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		mu = x.mean(-1, keepdim=True)
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
	def __init__(self, dim):
		super(LayerNorm, self).__init__()
		self.body = WithBias_LayerNorm(dim)

	def forward(self, x):
		h, w = x.shape[-2:]
		return to_4d(self.body(to_3d(x)), h, w)


class MFSM(nn.Module):
	def __init__(self):
		super(MFSM, self).__init__()
		
		channel=[8,16,32,32]
		self.V_en_1 = Restormer_CNN_block(3, channel[0])
		self.V_en_2 = Restormer_CNN_block(channel[0], channel[1])
		self.V_en_3 = Restormer_CNN_block(channel[1], channel[2])
		self.V_en_4 = Restormer_CNN_block(channel[2], channel[3])

		self.V_segembed_1 = TransformerBlock(dim_in=48, dim_out=16)
		self.V_segembed_2 = TransformerBlock(dim_in=96, dim_out=32)
		self.V_segembed_3 = TransformerBlock(dim_in=192, dim_out=32)

		self.I_en_1 = Restormer_CNN_block_IR(3, channel[0])
		self.I_en_2 = nn.Sequential(SEBlock(channels=channel[0],ratio=8),Restormer_CNN_block_IR(channel[0], channel[1]))
		self.I_en_3 = nn.Sequential(SEBlock(channels=channel[1],ratio=8),Restormer_CNN_block_IR(channel[1], channel[2]))
		self.I_en_4 = nn.Sequential(SEBlock(channels=channel[2],ratio=8),Restormer_CNN_block_IR(channel[2], channel[3]))

		self.f_1 = Restormer_CNN_block(channel[0]*2, channel[0]) # fusion layer
		self.f_2 = Restormer_CNN_block(channel[1]*2, channel[1])
		self.f_3 = Restormer_CNN_block(channel[2]*2, channel[2])
		self.f_4 = Restormer_CNN_block(channel[3]*2, channel[3])
		
		# self.f_1 = Restormer_CNN_block(channel[0]*3, channel[0]) # fusion layer
		# self.f_2 = Restormer_CNN_block(channel[1]*3, channel[1])
		# self.f_3 = Restormer_CNN_block(channel[2]*3, channel[2])
		# self.f_4 = Restormer_CNN_block(channel[3]*3, channel[3])

		self.V_down1=nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False)
		self.V_down2=nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False)
		self.V_down3=nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False)
		

		self.I_down1=nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False)
		self.I_down2=nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False)
		self.I_down3=nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False)
		

		self.up4=nn.Sequential(
			nn.ConvTranspose2d(channel[3],channel[2], 4, 2, 1, bias=False),
			nn.ReLU()
		)
		self.up3=nn.Sequential(
			nn.ConvTranspose2d(channel[2],channel[1], 4, 2, 1, bias=False),
			nn.ReLU()
		)
		self.up2=nn.Sequential(
			nn.ConvTranspose2d(channel[1],channel[0], 4, 2, 1, bias=False),
			nn.ReLU()
		)

		self.de_1 = Restormer_CNN_block(channel[0]*2,channel[0])
		self.de_2 = Restormer_CNN_block(channel[1]*2,channel[1])
		self.de_3 = Restormer_CNN_block(channel[2]*2,channel[2])
		self.de_4 = Restormer_CNN_block(channel[3],channel[3])

		# self.de_1 = Restormer_CNN_block_IR(channel[0]*2,channel[0])
		# self.de_2 = Restormer_CNN_block_IR(channel[1]*2,channel[1])
		# self.de_3 = Restormer_CNN_block_IR(channel[2]*2,channel[2])
		# self.de_4 = Restormer_CNN_block_IR(channel[3],channel[3])


		self.last = nn.Sequential(
			nn.Conv2d(channel[0], 3, kernel_size=3, stride=1, padding=1),
			nn.Sigmoid()
		)

	def forward(self, i, v, seg_fea):
		print("Restormer_CNN_block_IR is used!")
		i_1=self.I_en_1(i)
		i_2=self.I_en_2(self.I_down1(i_1))
		i_3=self.I_en_3(self.I_down2(i_2))
		i_4=self.I_en_4(self.I_down3(i_3))

		v_1=self.V_en_1(v)
		v_2=self.V_en_2(self.V_down1(v_1))
		v_3=self.V_en_3(self.V_down2(v_2))
		v_4=self.V_en_4(self.V_down3(v_3))

		v_seg_2 = self.V_segembed_1(v_2, seg_fea[0])
		v_seg_3 = self.V_segembed_2(v_3, seg_fea[1])
		v_seg_4 = self.V_segembed_3(v_4, seg_fea[2])

		f_1=self.f_1(torch.cat((i_1,v_1),1))
		f_2=self.f_2(torch.cat((i_2,v_seg_2),1))
		f_3=self.f_3(torch.cat((i_3,v_seg_3),1))
		f_4=self.f_4(torch.cat((i_4,v_seg_4),1))

		# f_1=self.f_1(torch.cat((i_1,v_1,attention_fusion(i_1,v_1)),1))
		# f_2=self.f_2(torch.cat((i_2,v_seg_2,attention_fusion(i_2,v_seg_2)),1))
		# f_3=self.f_3(torch.cat((i_3,v_seg_3,attention_fusion(i_3,v_seg_3)),1))
		# f_4=self.f_4(torch.cat((i_4,v_seg_4,attention_fusion(i_4,v_seg_4)),1))
	
		out=self.up4(self.de_4(f_4))
		out=self.up3(self.de_3(torch.cat((out,f_3),1)))
		out=self.up2(self.de_2(torch.cat((out,f_2),1)))
		out=self.de_1(torch.cat((out,f_1),1))
		return self.last(out)