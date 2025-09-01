import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    """
    Our spectrograms are padded, so different spectrograms will have
    a different length. We need to make sure we dont include any padding information
    in our convolution, and update padding masks for the next convolution in the stack!

    Args:

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        bias=True,
        **kwargs,
    ):
        super(MaskedConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs,
        )

    def forward(self, x: torch.FloatTensor, seq_lens: torch.LongTensor):
        """
        Updates convolution forward to zero out padding regions after convolution
        """

        ### Compute Output Seq Lengths of Each Sample After Convolution ###
        output_seq_lens = self._compute_output_seq_len(seq_lens)

        ### Pass Data Through Convolution ###
        conv_out = super().forward(x)

        ### Zero Out Any Values In The Padding Region (After Convolution) So they Dont Contribute ###
        max_len = output_seq_lens.max()
        seq_range = torch.arange(max_len, device=x.device)  # [max_len]

        # Compare each position against lengths
        mask = seq_range.unsqueeze(0) < output_seq_lens.to(x.device).unsqueeze(1)

        ### Unsqueeze mask to match image shape ###
        mask = mask.unsqueeze(1).unsqueeze(1)

        ### Apply Mask ###
        conv_out = conv_out.masked_fill(~mask, 0.0)

        return conv_out, output_seq_lens

    def _compute_output_seq_len(self, seq_lens: torch.LongTensor):
        """
        To perform masking AFTER the encoding 2D Convolutions, we need to
        compute what the shape of the output tensor is after each successive convolutions
        is applied.

        Convolution formula can be found in PyTorch Docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        """

        return (
            torch.floor(
                (seq_lens + (2 * self.padding[1]) - (self.kernel_size[1] - 1) - 1)
                // self.stride[1]
            )
            + 1
        )


class ConvolutionFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(ConvolutionFeatureExtractor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = MaskedConv2d(
            in_channels,
            out_channels,
            kernel_size=(11, 41),
            stride=(2, 2),
            padding=(5, 20),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = MaskedConv2d(
            out_channels,
            out_channels,
            kernel_size=(11, 21),
            stride=(2, 1),
            padding=(5, 10),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.output_feature_dim = 20

        ### Compute Final Output Features ###
        self.conv_output_features = self.output_feature_dim * self.out_channels

    def forward(self, x: torch.FloatTensor, seq_lens: torch.LongTensor):
        x, seq_lens = self.conv1(x, seq_lens)
        x = self.bn1(x)
        x = torch.nn.functional.hardtanh(x)

        x, seq_lens = self.conv2(x, seq_lens)
        x = self.bn2(x)
        x = torch.nn.functional.hardtanh(x)

        x = x.permute(0, 3, 1, 2).flatten(2)

        return x, seq_lens


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(RNNLayer, self).__init__()

        self.hidden_dim = hidden_size
        self.input_size = input_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.layernorm = nn.LayerNorm(2 * hidden_size)

    def forward(self, x: torch.FloatTensor, seq_lens: torch.LongTensor):
        seq_len = x.shape[1]

        ### Pack Sequence (For efficient computation that ignores padding) ###
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)

        ### Pass Packed Sequence through RNN ###
        out, ot = self.rnn(packed_x)

        ### Unpack (and repad) sequence ###
        x, y = nn.utils.rnn.pad_packed_sequence(
            out, total_length=seq_len, batch_first=True
        )

        ### Normalize ###
        x = self.layernorm(x)

        return x


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        conv_in_channels=1,
        conv_out_channels=32,
        rnn_hidden_size=512,
        rnn_depth=5,
        vocab_size=32,
    ):
        super(DeepSpeech2, self).__init__()

        self.feature_extractor = ConvolutionFeatureExtractor(
            conv_in_channels, conv_out_channels
        )

        self.output_hidden_features = self.feature_extractor.conv_output_features

        self.vocab_size = vocab_size

        ### Stack Together RNN Layers ###
        ### First Layer has 640 inputs, everything after has 2 * 512 inputs ###
        self.rnns = nn.ModuleList(
            [
                RNNLayer(
                    input_size=self.output_hidden_features
                    if i == 0
                    else 2 * rnn_hidden_size,
                    hidden_size=rnn_hidden_size,
                )
                for i in range(rnn_depth)
            ]
        )

        ### Classification Head ###
        self.head = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, rnn_hidden_size),
            nn.Hardtanh(),
            nn.Linear(rnn_hidden_size, self.vocab_size),
        )

    def forward(self, x: torch.FloatTensor, seq_lens: torch.LongTensor):
        ### Extract Features ###
        x, final_seq_lens = self.feature_extractor(x, seq_lens)

        ### Pass To RNN Layers ###
        for rnn in self.rnns:
            x = rnn(x, final_seq_lens)

        ### Classification Head ###
        x = self.head(x)

        return x, final_seq_lens


if __name__ == "__main__":
    m = DeepSpeech2()
    print(m)
    x = torch.randn(5, 1, 80, 1234)
    lens = torch.tensor([1234, 1230, 1200, 1000, 596])
    co, lo = m(x, lens)
    print(co.shape)
    print(lo)
