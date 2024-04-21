import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
from source.base import BaseModel
from source.base import BaseModel


class BatchNormReluRNN(nn.Module):
    """
    A class to combine the Pipeline: [BatchNorm, ReLU, RNN]
    """
    rnn_name_to_class = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(self, input_size: int, hidden_size: int, rnn_type: str, dropout_p: float, debug: bool):
        super().__init__()
        self.debug = debug
        # BatchNorm
        self.batch_norm = nn.BatchNorm1d(input_size)
        # RNN
        rnn_class = self.rnn_name_to_class[rnn_type]
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        # outputs: (batch, time, features)
        if self.debug:
            print(f'{inputs.size()} - BatchNormReluRNN input')

        # BatchNorm + ReLU
        inputs = F.relu(self.batch_norm(inputs.transpose(1, 2))).transpose(1, 2)
        if self.debug:
            print(f'{inputs.size()} - BatchNormReluRNN input after BatchNorm')
        # outputs: (batch, time, features)

        # Apply RNN
        outputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs: (batch, time, features * 2)
        if self.debug:
            print(f'{outputs.size()} - BatchNormReluRNN output')
        return outputs


class MaskCNN(nn.Module):
    """
    The module to compute convolutions using masks (so that inference is consistent among different batch size)
    Example: padded values after first convolution become non-zero, so the second convolution will be biased
    """

    def __init__(self, convolution_modules: nn.Sequential, debug: bool) -> None:
        super().__init__()
        self.convolution_modules = convolution_modules
        self.debug = debug

    def forward(self, inputs: torch.Tensor, seq_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (batch, n_channels, time, features)
        if self.debug:
            print(f'{inputs.size()} - MaskCNN input')

        output = None
        for module in self.convolution_modules:
            if self.debug:
                print(f'{inputs.size()}')
                print(module.__class__.__name__)
            # Apply Convolution
            output = module(inputs)

            # Create Mask
            mask = torch.BoolTensor(output.size()).fill_(0)
            # Put mask to device
            if output.is_cuda:
                mask = mask.cuda()

            # Find the length of sequence after Convolution
            seq_lengths = self._get_sequence_lengths(module, seq_lengths, dim=1)

            # Iterate over samples from batch
            for idx, length in enumerate(seq_lengths):
                length = length.item()

                # Mask the padded values
                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            # Apply mask to output
            output = output.masked_fill(mask, 0)
            inputs = output
        if self.debug:
            print(f'{output.size()} - MaskCNN output')

        return output, seq_lengths

    def get_output_size(self, input_size: int):
        """
        Compute the output size along known dimension (along the frequency dimension of the spectrogram)
        """
        size = torch.Tensor([input_size]).int()
        for module in self.convolution_modules:
            size = self._get_sequence_lengths(module, size, dim=0)
        return size.item()

    def transform_input_lengths(self, input_size: torch.Tensor):
        """
        Transform spectrogram length to sequence length
        """
        for module in self.convolution_modules:
            input_size = self._get_sequence_lengths(module, input_size, dim=1)
        return input_size

    def _get_sequence_lengths(self, convolution: nn.Module, input_size: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Get the size of convolution output given input_size input
        """
        # Output size = [(input_size + 2 * padding_size - (dilation_size * (kernel_size - 1))) / stride_size] + 1
        if isinstance(convolution, nn.Conv2d):
            # Take time size (index=1) because this dimension is different for each element in batch
            kernel_size = convolution.kernel_size[dim]
            dilation_size = convolution.dilation[dim]
            padding_size = convolution.padding[dim]
            stride_size = convolution.stride[dim]
            input_size = (input_size + 2 * padding_size - dilation_size * (kernel_size - 1) - 1) / stride_size + 1

        return input_size.int()


class ConvolutionsModule(nn.Module):
    activation_name_to_class = {
        'relu': nn.ReLU,
        'hardtanh': nn.Hardtanh,
    }

    def __init__(
            self,
            n_feats: int,
            in_channels: int,
            out_channels: int,
            activation: str,
            debug: bool
    ) -> None:
        super().__init__()
        self.debug = debug
        self.activation = self.activation_name_to_class[activation]()
        self.mask_conv = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,
            ),
            debug=debug
        )
        self.output_size = self.mask_conv.get_output_size(n_feats)

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (batch, features, time)
        if self.debug:
            print(f'{spectrogram.size()} - ConvolutionsModule input')
        outputs, output_lengths = self.mask_conv(spectrogram.unsqueeze(1), spectrogram_length)
        # outputs: (batch, channels, features, time)
        batch_size, channels, features, time = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        # outputs: (batch, time, channels, features)
        outputs = outputs.view(batch_size, time, channels * features)
        # outputs: (batch, time, channels * features)
        if self.debug:
            print(f'{outputs.size()} - ConvolutionsModule output')

        return outputs, output_lengths


class DeepSpeech2(BaseModel):
    def __init__(
            self,
            n_feats: int,
            n_class: int,
            rnn_type='gru',
            n_rnn_layers: int = 5,
            conv_out_channels: int = 32,
            rnn_hidden_size: int = 512,
            dropout_p: float = 0.1,
            activation: str = 'relu',
            debug: bool = True
    ):
        super().__init__(n_feats=n_feats, n_class=n_class)
        self.debug = debug
        self.conv = ConvolutionsModule(n_feats=n_feats, in_channels=1, out_channels=conv_out_channels, activation=activation, debug=debug)

        rnn_output_size = rnn_hidden_size * 2
        self.rnn_layers = nn.ModuleList([
            BatchNormReluRNN(
                input_size=self.conv.mask_conv.get_output_size(n_feats) * conv_out_channels if idx == 0 else rnn_output_size,
                hidden_size=rnn_hidden_size,
                rnn_type=rnn_type,
                dropout_p=dropout_p,
                debug=debug
            ) for idx in range(n_rnn_layers)
        ])

        self.batch_norm = nn.BatchNorm1d(rnn_output_size)
        self.fc = nn.Linear(rnn_output_size, n_class, bias=False)

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (batch, features, time)
        if self.debug:
            print(f'{spectrogram.size()} - initial input')
        outputs, output_lengths = self.conv(spectrogram, spectrogram_length)

        # outputs: (batch, time, features * channels)
        outputs = outputs.permute(1, 0, 2).contiguous()
        # outputs: (time, batch, features * channels)
        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs.transpose(0, 1), output_lengths)

        # outputs: (time, batch, rnn_output_size)
        if self.debug:
            print(f'{outputs.size()} - after RNN layers')

        outputs = self.batch_norm(outputs.permute(1, 2, 0))

        # outputs: (batch, rnn_output_size, time)
        outputs = self.fc(outputs.transpose(1, 2))

        return {"logits": outputs}

    def transform_input_lengths(self, input_lengths):
        return self.conv.mask_conv.transform_input_lengths(input_lengths)
