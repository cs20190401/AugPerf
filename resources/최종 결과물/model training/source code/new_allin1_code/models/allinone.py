import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from .dinat import DinatLayer1d, DinatLayer2d
from .utils import get_activation_function
from ..config import Config
from ..typings import AllInOneOutput


class AllInOne(nn.Module):
  def __init__(self, cfg: Config):
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
    # N: batch size
    # K: instrument
    # C: channel
    # T: time
    # F: frequency
    # x has shape of: N, K, T, F
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
    logits_section = self.section_classifier(hidden_states)
    logits_function = self.function_classifier(hidden_states)

    return AllInOneOutput(
      logits_beat=logits_beat,
      logits_downbeat=logits_downbeat,
      logits_section=logits_section,
      logits_function=logits_function,
      embeddings=hidden_states,
    )
  
class AllInOneTempo(AllInOne):
  def __init__(self, cfg: Config):
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
    # N: batch size
    # K: instrument
    # C: channel
    # T: time
    # F: frequency
    # x has shape of: N, K, T, F
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

    return AllInOneOutput(
      logits_beat=logits_beat,
      logits_downbeat=logits_downbeat,
      logits_tempo=logits_tempo,
      logits_nobeat=logits_nobeat if self.nobeat else None,
      embeddings=hidden_states,
    )


class AllInOneTempo_with_Head(AllInOne):
  def __init__(self, cfg: Config):
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
    # N: batch size
    # K: instrument
    # C: channel
    # T: time
    # F: frequency
    # x has shape of: N, K, T, F
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
    logits_beat = logits_beat + logits_downbeat
    logits_tempo = self.tempo_classfier(hidden_states) # [Batch, 300]
    if self.nobeat:
      logits_nobeat = self.nobeat_classifier(hidden_states) # [Batch, Frame]

    return AllInOneOutput(
      logits_beat=logits_beat,
      logits_downbeat=logits_downbeat,
      logits_tempo=logits_tempo,
      logits_nobeat=logits_nobeat if self.nobeat else None,
      embeddings=hidden_states,
    )


class RealTimeAllInOne(AllInOneTempo):
  def __init__(self, cfg: Config):
    super().__init__(cfg)
    self.mag_classifier = HeadRT(num_classes=1, cfg=cfg)

  def forward(
      self,
      inputs: torch.FloatTensor,
      output_attentions: bool | None = None):
    
    output = super().forward(inputs, output_attentions)

    hidden_states = output.embeddings
    logits_mag = self.mag_classifier(hidden_states)

    return AllInOneOutput(
      logits_beat=output.logits_beat,
      logits_downbeat=output.logits_downbeat,
      logits_tempo=output.logits_tempo,
      logits_nobeat=output.logits_nobeat,
      logits_mag = logits_mag,
      embeddings=hidden_states,
    )


class AllInOneEncoder(nn.Module):
  def __init__(self, cfg: Config, depth: int):
    super().__init__()
    self.cfg = cfg

    drop_path_rates = [x.item() for x in torch.linspace(0, cfg.drop_path, depth)]
    dilations = [
      min(cfg.dilation_factor ** i, cfg.dilation_max)
      for i in range(depth)
    ] # dilations = [1, 2, 4, 8, 16, 32, ... , 2^(d-1)]
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
  def __init__(self, cfg: Config, dilation: int, drop_path_rate: float):
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
    # N: batch size
    # K: instrument
    # T: time
    # C: channel
    # x has shape of: NK, T, C=16
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
  def __init__(self, cfg: Config):
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
    # NK: batch x inst
    # C: channel
    # T: time
    # F: frequency
    # x has shape of: NK, C=1, T, F
    # x = x.unsqueeze(1)  # NK, C=1, T, F=81
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
  def __init__(self, num_classes: int, cfg: Config, init_confidence: float = None):
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
  def __init__(self, num_classes: int, cfg: Config, init_confidence: float = None):
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


class DepthwiseSeparableConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, causal:bool=True, padding=0):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.dilation = dilation
    self.causal = causal
    
    # Depthwise Convolution (causal in time dimension)
    self.depthwise = nn.Conv2d(
      in_channels, 
      in_channels, 
      kernel_size=kernel_size, 
      stride=stride, 
      dilation=dilation, 
      groups=in_channels, 
      padding=0 if self.causal else padding,  # No padding here, handled manually for causality
      bias=False
    )
    
    # Pointwise Convolution (1x1)
    self.pointwise = nn.Conv2d(
      in_channels, 
      out_channels, 
      kernel_size=1,
      stride=1,
      padding=0,
      bias=False
    )

  def forward(self, x):
    # Apply causal padding in the time dimension
    if self.causal:
      pad = (self.kernel_size[0] - 1) * self.dilation
      x = F.pad(x, (0, 0, pad, 0))  # Only pad the time dimension causally

    # Depthwise Convolution
    x = self.depthwise(x)

    # Pointwise Convolution
    x = self.pointwise(x)

    return x


class DepthSepEmbeddings(AllInOneEmbeddings):
  def __init__(self, cfg: Config):
    super().__init__(cfg=cfg)
    first_conv_filters = self.hidden_size if cfg.model == 'tcn' else self.hidden_size // 2
    causal = cfg.causal
    self.conv0 = DepthwiseSeparableConv2d(in_channels=1, out_channels=first_conv_filters, kernel_size=(3, 3), stride=(1, 1), padding=0 if causal else (1,0), causal=causal)
    self.conv1 = DepthwiseSeparableConv2d(in_channels=first_conv_filters, out_channels=self.hidden_size, kernel_size=(1, 12), stride=(1, 1), padding=0 if causal else (0,0), causal=causal)
    self.conv2 = DepthwiseSeparableConv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=0 if causal else (1,0), causal=causal)

class AllInOneTempoDepthSep(AllInOneTempo):
  def __init__(self, cfg: Config):
    super().__init__(cfg=cfg)
    self.embeddings = DepthSepEmbeddings(cfg)


if __name__ == "__main__":
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  example_input = torch.rand((1,250,24)).to(device)
  cfg = Config
  encoder = AllInOneEncoder(cfg=cfg, depth=10).to(device)
  example_output = encoder(example_input)
  print(example_output[0][-1].shape)