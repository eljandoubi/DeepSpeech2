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
        mask = seq_range.unsqueeze(0) < output_seq_lens.unsqueeze(1)

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


if __name__ == "__main__":
    m = MaskedConv2d(
        1, 32, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20), bias=False
    )
    print(m)
    x = torch.randn(5, 1, 80, 1234)
    lens = torch.tensor([1234, 1230, 1200, 1000, 596])
    co, lo = m(x, lens)
    print(co.shape)
    print(lo)
