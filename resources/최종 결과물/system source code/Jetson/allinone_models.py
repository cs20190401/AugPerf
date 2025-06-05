import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC,  abstractmethod

from typing import Optional, Tuple, Callable
from natten.functional import natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb

class AllInOne(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.cfg = cfg
    self.num_levels = cfg.depth
    self.num_features = int(cfg.dim_embed * 2 ** (self.num_levels - 1))

    self.embeddings = AllInOneEmbeddings(cfg)

    self.encoder = AllInOneEncoder(
      cfg,
      depth=cfg.depth,
    )

    self.norm = nn.LayerNorm(cfg.dim_embed, eps=cfg.layer_norm_eps)

    self.beat_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.05)
    self.downbeat_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.0125)
    self.section_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.001)
    self.function_classifier = Head(num_classes=cfg.data.num_labels, cfg=cfg)

    self.dropout = nn.Dropout(cfg.drop_last)

  def forward(
    self,
    inputs: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    pass
  
class AllInOneTempo(AllInOne):
  def __init__(self, cfg):
    super().__init__(cfg=cfg)
    del self.section_classifier, self.function_classifier
    self.beat_classifier = HeadRT(num_classes=1, cfg=cfg, init_confidence=0.05)
    self.downbeat_classifier = HeadRT(num_classes=1, cfg=cfg, init_confidence=0.0125)
    self.tempo_classfier = HeadRT(num_classes=300, cfg=cfg)
    self.nobeat = False
    if cfg.focal_loss or cfg.dice_loss:
      self.nobeat = True
      self.nobeat_classifier = HeadRT(num_classes=1, cfg=cfg, init_confidence=0.4)

  def forward(
    self,
    inputs: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    N, K, T, F = inputs.shape

    inputs = inputs.reshape(-1, 1, T, F)  # N x K, C=1, T, F=81
    frame_embed = self.embeddings(inputs)  # NK, T, C=16
    
    encoder_outputs = self.encoder(
      frame_embed,
      output_attentions=output_attentions,
    )
    hidden_state_levels = encoder_outputs[0]

    hidden_states = hidden_state_levels[-1].reshape(N, K, T, -1)  # N, K, T, C=16
    hidden_states = self.norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    logits_beat = self.beat_classifier(hidden_states)
    logits_downbeat = self.downbeat_classifier(hidden_states)
    logits_tempo = self.tempo_classfier(hidden_states)
    if self.nobeat:
      logits_nobeat = self.nobeat_classifier(hidden_states)
	  
    return (
      logits_beat,
      logits_downbeat,
      logits_tempo,
      logits_nobeat if self.nobeat else None,
      hidden_states
    )


class AllInOneTempoSum(AllInOne):
  def __init__(self, cfg):
    super().__init__(cfg=cfg)
    del self.section_classifier, self.function_classifier
    self.beat_classifier = HeadRT(num_classes=1, cfg=cfg, init_confidence=0.05)
    self.downbeat_classifier = HeadRT(num_classes=1, cfg=cfg, init_confidence=0.0125)
    self.tempo_classfier = HeadRT(num_classes=300, cfg=cfg)
    self.nobeat = False
    if cfg.focal_loss or cfg.dice_loss:
      self.nobeat = True
      self.nobeat_classifier = HeadRT(num_classes=1, cfg=cfg, init_confidence=0.4)

  def forward(
    self,
    inputs: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    N, K, T, F = inputs.shape

    inputs = inputs.reshape(-1, 1, T, F)  # N x K, C=1, T, F=81
    frame_embed = self.embeddings(inputs)  # NK, T, C=16
    
    encoder_outputs = self.encoder(
      frame_embed,
      output_attentions=output_attentions,
    )
    hidden_state_levels = encoder_outputs[0]

    hidden_states = hidden_state_levels[-1].reshape(N, K, T, -1)  # N, K, T, C=16
    hidden_states = self.norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    logits_beat = self.beat_classifier(hidden_states)
    logits_downbeat = self.downbeat_classifier(hidden_states)
    logits_beat = logits_beat + logits_downbeat
    logits_tempo = self.tempo_classfier(hidden_states)
    if self.nobeat:
      logits_nobeat = self.nobeat_classifier(hidden_states)
	  
    return (
      logits_beat,
      logits_downbeat,
      logits_tempo,
      logits_nobeat if self.nobeat else None,
      hidden_states
    )
  

class AllInOneTempo_with_Head(AllInOne):
  def __init__(self, cfg):
    super().__init__(cfg=cfg)
    del self.section_classifier, self.function_classifier
    self.beat_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.05)
    self.downbeat_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.0125)
    self.tempo_classfier = HeadRT(num_classes=300, cfg=cfg)
    self.nobeat = False
    if cfg.focal_loss or cfg.dice_loss:
      self.nobeat = True
      self.nobeat_classifier = Head(num_classes=1, cfg=cfg, init_confidence=0.4)

  def forward(
    self,
    inputs: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    N, K, T, F = inputs.shape

    inputs = inputs.reshape(-1, 1, T, F)  # N x K, C=1, T, F=81
    frame_embed = self.embeddings(inputs)  # NK, T, C=16
    
    encoder_outputs = self.encoder(
      frame_embed,
      output_attentions=output_attentions,
    )
    hidden_state_levels = encoder_outputs[0]

    hidden_states = hidden_state_levels[-1].reshape(N, K, T, -1)  # N, K, T, C=16
    hidden_states = self.norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    logits_beat = self.beat_classifier(hidden_states) # [Batch, Frame]
    logits_downbeat = self.downbeat_classifier(hidden_states) # [Batch, Frame]
    logits_tempo = self.tempo_classfier(hidden_states) # [Batch, 300]
    if self.nobeat:
      logits_nobeat = self.nobeat_classifier(hidden_states) # [Batch, Frame]
    return (
      logits_beat,
      logits_downbeat,
      logits_tempo,
      logits_nobeat if self.nobeat else None,
      hidden_states
    )


class AllInOneEncoder(nn.Module):
  def __init__(self, cfg, depth: int):
    super().__init__()
    self.cfg = cfg

    drop_path_rates = [x.item() for x in torch.linspace(0, cfg.drop_path, depth)]
    dilations = [
      min(cfg.dilation_factor ** i, cfg.dilation_max)
      for i in range(depth)
    ]
    self.layers = nn.ModuleList(
      [
        AllInOneBlock(
          cfg=cfg,
          dilation=dilations[i],
          drop_path_rate=drop_path_rates[i],
        )
        for i in range(depth)
      ]
    )

  def forward(
    self,
    frame_embed: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    # N: batch size
    # K: instrument
    # T: time
    # C: channel
    # x has shape of: NK, T, C=16

    hidden_state_levels = []
    hidden_states = frame_embed
    for i, layer in enumerate(self.layers):
      layer_outputs = layer(hidden_states, output_attentions)
      hidden_states = layer_outputs[0]
      hidden_state_levels.append(hidden_states)

    outputs = (hidden_state_levels,)
    if output_attentions:
      outputs += layer_outputs[1:]
    return outputs


class AllInOneBlock(nn.Module):
  def __init__(self, cfg, dilation: int, drop_path_rate: float):
    super().__init__()

    self.cfg = cfg
    self.dilation = dilation

    self.timelayer = DinatLayer1d(
      cfg=cfg,
      dim=cfg.dim_embed,
      num_heads=cfg.num_heads,
      kernel_size=cfg.kernel_size,
      dilation=dilation,
      drop_path_rate=drop_path_rate,
      double_attention=cfg.double_attention,
    )

    if cfg.instrument_attention:
      self.instlayer = DinatLayer2d(
        cfg=cfg,
        dim=cfg.dim_embed,
        num_heads=cfg.num_heads,
        kernel_size=5,
        dilation=1,
        drop_path_rate=drop_path_rate,
      )
    else:
      self.instlayer = DinatLayer1d(
        cfg=cfg,
        dim=cfg.dim_embed,
        num_heads=cfg.num_heads,
        kernel_size=5,
        dilation=1,
        drop_path_rate=drop_path_rate,
        double_attention=False,
      )

  def forward(
    self,
    hidden_states: torch.FloatTensor,
    output_attentions: Optional[bool] = None,
  ):
    NK, T, C = hidden_states.shape
    if self.cfg.data.bpfed:
      num_channels = len(self.cfg.data.sources)
    elif self.cfg.data.demixed:
      num_channels = self.cfg.data.num_instruments
    else:
      num_channels = 1
    N, K = NK // num_channels, num_channels

    timelayer_outputs = self.timelayer(hidden_states, output_attentions)
    hidden_states = timelayer_outputs[0]
    if self.cfg.instrument_attention:
      hidden_states = hidden_states.reshape(N, K, T, C)
      instlayer_outputs = self.instlayer(hidden_states, output_attentions)
      hidden_states = instlayer_outputs[0]
      hidden_states = hidden_states.reshape(NK, T, C)
    else:
      instlayer_outputs = self.instlayer(hidden_states, output_attentions)
      hidden_states = instlayer_outputs[0]

    outputs = (hidden_states,)
    if output_attentions:
      outputs += timelayer_outputs[1:]
      if self.instlayer is not None:
        outputs += instlayer_outputs[1:]
    return outputs


class AllInOneEmbeddings(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    dim_input, hidden_size = cfg.dim_input, cfg.dim_embed
    self.dim_input = dim_input
    self.hidden_size = hidden_size
    self.causal = True if cfg.causal and cfg.model=="allinonetempo" else False

    self.act_fn = get_activation_function(cfg.act_conv)
    first_conv_filters = hidden_size if cfg.model == 'tcn' else hidden_size // 2

    self.conv0 = nn.Conv2d(1, first_conv_filters, kernel_size=(3, 3), stride=(1, 1), padding=(0,0) if self.causal else (1, 0))
    self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
    self.drop0 = nn.Dropout(cfg.drop_conv)

    self.conv1 = nn.Conv2d(first_conv_filters, hidden_size, kernel_size=(1, 12), stride=(1, 1), padding=(0, 0))
    self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
    self.drop1 = nn.Dropout(cfg.drop_conv)

    self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(0,0) if self.causal else (1, 0))
    self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))

    self.norm = nn.LayerNorm(cfg.dim_embed)
    self.dropout = nn.Dropout(cfg.drop_conv)

  def forward(self, x: torch.FloatTensor):
    if self.causal:
      x = F.pad(x, (0, 0, 2, 0))
    x = self.conv0(x)  # NK, C=12, T, F=83
    x = self.pool0(x)
    x = self.act_fn(x)
    x = self.drop0(x)

    x = self.conv1(x)
    x = self.pool1(x)
    x = self.act_fn(x)
    x = self.drop1(x)

    if self.causal:
      x = F.pad(x, (0, 0, 2, 0))
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.act_fn(x)

    embeddings = x.squeeze(-1)  # NK, C=24, T
    embeddings = embeddings.permute(0, 2, 1)  # NK, T, C=24
    embeddings = self.norm(embeddings)
    embeddings = self.dropout(embeddings)

    return embeddings


class Head(nn.Module):
  def __init__(self, num_classes: int, cfg, init_confidence: float = None):
    super().__init__()
    if cfg.data.bpfed:
      num_channels = len(cfg.data.sources)
    elif cfg.data.demixed:
      num_channels = cfg.data.num_instruments
    else:
      num_channels = 1
    self.classifier = nn.Linear(num_channels * cfg.dim_embed, num_classes)

    if init_confidence is not None:
      self.reset_parameters(init_confidence)

  def reset_parameters(self, confidence) -> None:
    """
    Initialization following:
    "Focal loss for dense object detection." ICCV. 2017.
    """
    self.classifier.bias.data.fill_(-torch.log(torch.tensor(1 / confidence - 1)))

  def forward(self, x: torch.FloatTensor):
    # x shape: N, K, T, C=24
    batch, inst, frame, embed = x.shape
    x = x.permute(0, 2, 1, 3)  # batch, frame, inst, embed
    x = x.reshape(batch, frame, inst * embed)  # batch, frame, inst x embed
    logits = self.classifier(x)  # batch, frame, class
    logits = logits.permute(0, 2, 1)  # batch, class, frame
    if logits.shape[1] == 1:
      logits = logits.squeeze(1)
    return logits
    

class HeadRT(nn.Module):
  def __init__(self, num_classes: int, cfg, init_confidence: float = None):
    super().__init__()
    if cfg.data.bpfed:
      num_channels = len(cfg.data.sources)
    elif cfg.data.demixed:
      num_channels = cfg.data.num_instruments
    else:
      num_channels = 1
    self.classifier = nn.Linear(num_channels * cfg.dim_embed, num_classes)

    if init_confidence is not None:
      self.reset_parameters(init_confidence)

    T = int(cfg.buffer_length * cfg.sample_rate / cfg.hop_size) # 250 frames
    self.fc = nn.Linear(T,1)

  def reset_parameters(self, confidence) -> None:
    """
    Initialization following:
    "Focal loss for dense object detection." ICCV. 2017.
    """
    self.classifier.bias.data.fill_(-torch.log(torch.tensor(1 / confidence - 1)))

  def forward(self, x: torch.FloatTensor):
    # x shape: N, K, T, C=24
    batch, inst, frame, embed = x.shape
    x = x.permute(0, 2, 1, 3)  # batch, frame, inst, embed
    x = x.reshape(batch, frame, inst * embed)  # batch, frame, inst x embed
    # print("before classifier:", x.shape)
    x = self.classifier(x)  # batch, frame, class
    x = x.permute(0, 2, 1)  # batch, class, frame
    # print("before fc:", x.shape)
    logits = self.fc(x) # batch, class
    if logits.shape[-1] == 1:
      logits = logits.squeeze(-1)
    if logits.shape[-1] == 1:
      logits = logits.squeeze(-1)
    return logits

def get_activation_function(name: str):
  activation_functions = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'softmax': nn.Softmax(dim=1),
    'log_softmax': nn.LogSoftmax(dim=1),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'gelu': nn.GELU(),
    'prelu': nn.PReLU(),
  }
  
  if name in activation_functions:
    return activation_functions[name]
  else:
    raise ValueError(f"Unsupported activation function: {name}")

# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
  if drop_prob == 0.0 or not training:
    return input
  keep_prob = 1 - drop_prob
  shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
  random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
  random_tensor.floor_()  # binarize
  output = input.div(keep_prob) * random_tensor
  return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Dinat
class DinatDropPath(nn.Module):
  def __init__(self, drop_prob: Optional[float] = None) -> None:
    super().__init__()
    self.drop_prob = drop_prob
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return drop_path(hidden_states, self.drop_prob, self.training)
  
  def extra_repr(self) -> str:
    return "p={}".format(self.drop_prob)


class _NeighborhoodAttentionNd(ABC, nn.Module):
  rpb: nn.Parameter
  nattendqkrpb: Callable
  nattendav: Callable
  
  def __init__(
    self,
    cfg,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__()
    if dim % num_heads != 0:
      raise ValueError(
        f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
      )
    
    self.num_attention_heads = num_heads
    self.attention_head_size = int(dim / num_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size
    self.kernel_size = kernel_size
    self.dilation = dilation
    
    self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=cfg.qkv_bias)
    
    self.dropout = nn.Dropout(cfg.drop_attention)
  
  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor]:
    query_layer = self.transpose_for_scores(self.query(hidden_states))
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    
    query_layer = query_layer / math.sqrt(self.attention_head_size)
    
    attention_scores = self.nattendqkrpb(query_layer, key_layer, self.rpb, self.kernel_size, self.dilation)
    
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    
    attention_probs = self.dropout(attention_probs)
    
    context_layer = self.nattendav(attention_probs, value_layer, self.kernel_size, self.dilation)
    if len(context_layer.shape) > 4:  # 2D
      context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
    else:  # 1D
      context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)
    
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    
    return outputs
  
  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(new_x_shape)
    if len(x.shape) > 4:  # 2D
      return x.permute(0, 3, 1, 2, 4)
    else:  # 1D
      return x.permute(0, 2, 1, 3)


class NeighborhoodAttention1d(_NeighborhoodAttentionNd):
  def __init__(
    self,
    cfg,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__(cfg, dim, num_heads, kernel_size, dilation)
    self.rpb = nn.Parameter(
      torch.zeros(num_heads, (2 * self.kernel_size - 1)),
      requires_grad=True,
    )
    self.nattendqkrpb = natten1dqkrpb
    self.nattendav = natten1dav


class NeighborhoodAttention2d(_NeighborhoodAttentionNd):
  def __init__(
    self,
    cfg,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int
  ):
    super().__init__(cfg, dim, num_heads, kernel_size, dilation)
    self.rpb = nn.Parameter(
      torch.zeros(num_heads, (2 * self.kernel_size - 1), (2 * self.kernel_size - 1)),
      requires_grad=True,
    )
    self.nattendqkrpb = natten2dqkrpb
    self.nattendav = natten2dav


# Copied from transformers.models.nat.modeling_nat.NeighborhoodAttentionOutput
class NeighborhoodAttentionOutput(nn.Module):
  def __init__(self, config, dim: int):
    super().__init__()
    self.dense = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(config.drop_attention)
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    
    return hidden_states


class _NeighborhoodAttentionModuleNd(ABC, nn.Module):
  self: _NeighborhoodAttentionNd
  
  def __init__(self, cfg, dim: int):
    super().__init__()
    # self.self = _NeighborhoodAttentionNd(config, dim, num_heads, kernel_size, dilation)
    self.output = NeighborhoodAttentionOutput(cfg, dim)
  
  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor]:
    self_outputs = self.self(hidden_states, output_attentions)
    attention_output = self.output(self_outputs[0])
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


class NeighborhoodAttentionModule1d(_NeighborhoodAttentionModuleNd):
  def __init__(self, cfg, dim: int, num_heads: int, kernel_size: int, dilation: int):
    super().__init__(cfg, dim)
    self.self = NeighborhoodAttention1d(cfg, dim, num_heads, kernel_size, dilation)


class NeighborhoodAttentionModule2d(_NeighborhoodAttentionModuleNd):
  def __init__(self, cfg, dim: int, num_heads: int, kernel_size: int, dilation: int):
    super().__init__(cfg, dim)
    self.self = NeighborhoodAttention2d(cfg, dim, num_heads, kernel_size, dilation)


# Copied from transformers.models.nat.modeling_nat.NatIntermediate with Nat->Dinat
class DinatIntermediate(nn.Module):
  def __init__(self, config, dim_in: int, dim_out: int):
    super().__init__()
    self.dense = nn.Linear(dim_in, dim_out)
    if isinstance(config.act_transformer, str):
      self.intermediate_act_fn = get_activation_function(config.act_transformer)
    else:
      self.intermediate_act_fn = config.act_transformer
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


# Copied from transformers.models.nat.modeling_nat.NatOutput with Nat->Dinat
class DinatOutput(nn.Module):
  def __init__(self, config, dim_in: int, dim_out: int):
    super().__init__()
    self.dense = nn.Linear(dim_in, dim_out)
    self.dropout = nn.Dropout(config.drop_hidden)
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


class _DinatLayerNd(ABC, nn.Module):
  attention: _NeighborhoodAttentionModuleNd
  attention2: Optional[_NeighborhoodAttentionModuleNd]
  
  def __init__(
    self,
    cfg,
    dim: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float,
    double_attention: bool,
  ):
    super().__init__()
    self.double_attention = double_attention
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.window_size = self.kernel_size * self.dilation
    if double_attention:
      self.window_size *= 2
    self.layernorm_before = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
    self.drop_path = DinatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    dim_after = dim * 2 if double_attention else dim
    self.layernorm_after = nn.LayerNorm(dim_after, eps=cfg.layer_norm_eps)
    self.intermediate = DinatIntermediate(cfg, dim_after, int(dim_after * cfg.mlp_ratio))
    self.output = DinatOutput(cfg, int(dim_after * cfg.mlp_ratio), dim)
  
  @abstractmethod
  def maybe_pad(self, *args, **kwargs):
    raise NotImplementedError
  
  def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: Optional[bool] = False,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(hidden_states.shape) > 3:
      is_2d = True
      N, K, T, C = hidden_states.size()
    else:
      is_2d = False
      N, T, C = hidden_states.shape
    shortcut = hidden_states
    
    hidden_states = self.layernorm_before(hidden_states)
    # pad hidden_states if they are smaller than kernel size x dilation
    if is_2d:
      hidden_states, pad_values = self.maybe_pad(hidden_states, K, T)
      _, height_pad, width_pad, _ = hidden_states.shape
    else:
      hidden_states, pad_values = self.maybe_pad(hidden_states, T)
    
    attention_inputs = hidden_states
    hidden_states_list = []
    for attention in [self.attention, self.attention2]:
      if attention is None:
        continue
      
      attention_output = attention(attention_inputs, output_attentions=output_attentions)
      attention_output = attention_output[0]
      
      if is_2d:
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
          attention_output = attention_output[:, :K, :T, :].contiguous()
      else:
        was_padded = pad_values[3] > 0
        if was_padded:
          attention_output = attention_output[:, :T, :].contiguous()
      
      hidden_states = shortcut + self.drop_path(attention_output)
      hidden_states_list.append(hidden_states)
    
    if self.double_attention:
      hidden_states = torch.cat(hidden_states_list, dim=-1)
      shortcut = torch.stack(hidden_states_list).sum(dim=0) / 2.
    else:
      shortcut = hidden_states
    layer_output = self.layernorm_after(hidden_states)
    layer_output = self.output(self.intermediate(layer_output))
    
    layer_output = shortcut + self.drop_path(layer_output)
    
    # layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
    layer_outputs = (layer_output,)
    return layer_outputs


class DinatLayer1d(_DinatLayerNd):
  def __init__(
    self,
    cfg,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float,
    double_attention: bool,
  ):
    super().__init__(cfg, dim, kernel_size, dilation, drop_path_rate, double_attention)
    self.attention = NeighborhoodAttentionModule1d(cfg, dim, num_heads, kernel_size, dilation)
    if double_attention:
      self.attention2 = NeighborhoodAttentionModule1d(cfg, dim, num_heads, kernel_size, dilation * 2)
    else:
      self.attention2 = None
  
  def maybe_pad(self, hidden_states, frames):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0)
    if frames < window_size:
      pad_l = 0
      pad_r = max(0, window_size - frames)
      pad_values = (0, 0, pad_l, pad_r)
      hidden_states = nn.functional.pad(hidden_states, pad_values)
    return hidden_states, pad_values


class DinatLayer2d(_DinatLayerNd):
  def __init__(
    self,
    cfg,
    dim: int,
    num_heads: int,
    kernel_size: int,
    dilation: int,
    drop_path_rate: float
  ):
    super().__init__(cfg, dim, kernel_size, dilation, drop_path_rate, double_attention=False)
    self.attention = NeighborhoodAttentionModule2d(cfg, dim, num_heads, kernel_size, dilation)
    self.attention2 = None
  
  def maybe_pad(self, hidden_states, height, width):
    window_size = self.window_size
    pad_values = (0, 0, 0, 0, 0, 0)
    if height < window_size or width < window_size:
      pad_l = pad_t = 0
      pad_r = max(0, window_size - width)
      pad_b = max(0, window_size - height)
      pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
      hidden_states = nn.functional.pad(hidden_states, pad_values)
    return hidden_states, pad_values