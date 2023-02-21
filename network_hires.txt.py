ConvolutionalOccupancyNetwork(
  (decoder_qual): LocalDecoder(
    (fc_c): ModuleList(
      (0): Linear(in_features=96, out_features=32, bias=True)
      (1): Linear(in_features=96, out_features=32, bias=True)
      (2): Linear(in_features=96, out_features=32, bias=True)
      (3): Linear(in_features=96, out_features=32, bias=True)
      (4): Linear(in_features=96, out_features=32, bias=True)
    )
    (fc_p): Linear(in_features=7, out_features=32, bias=True)
    (blocks): ModuleList(
      (0): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (1): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (2): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (3): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (4): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
    )
    (fc_out): Linear(in_features=32, out_features=1, bias=True)
  )
  (decoder_width): LocalDecoder(
    (fc_c): ModuleList(
      (0): Linear(in_features=96, out_features=32, bias=True)
      (1): Linear(in_features=96, out_features=32, bias=True)
      (2): Linear(in_features=96, out_features=32, bias=True)
      (3): Linear(in_features=96, out_features=32, bias=True)
      (4): Linear(in_features=96, out_features=32, bias=True)
    )
    (fc_p): Linear(in_features=7, out_features=32, bias=True)
    (blocks): ModuleList(
      (0): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (1): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (2): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (3): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (4): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
    )
    (fc_out): Linear(in_features=32, out_features=1, bias=True)
  )
  (decoder_tsdf): LocalDecoder(
    (fc_c): ModuleList(
      (0): Linear(in_features=96, out_features=32, bias=True)
      (1): Linear(in_features=96, out_features=32, bias=True)
      (2): Linear(in_features=96, out_features=32, bias=True)
      (3): Linear(in_features=96, out_features=32, bias=True)
      (4): Linear(in_features=96, out_features=32, bias=True)
    )
    (fc_p): Linear(in_features=3, out_features=32, bias=True)
    (blocks): ModuleList(
      (0): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (1): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (2): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (3): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
      (4): ResnetBlockFC(
        (fc_0): Linear(in_features=32, out_features=32, bias=True)
        (fc_1): Linear(in_features=32, out_features=32, bias=True)
        (actvn): ReLU()
      )
    )
    (fc_out): Linear(in_features=32, out_features=1, bias=True)
  )
  (encoder): LocalVoxelEncoder(
    (conv_in): Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (unet): UNet(
      (down_convs): ModuleList(
        (0): DownConv(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (1): DownConv(
          (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (2): DownConv(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (up_convs): ModuleList(
        (0): UpConv(
          (upconv): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
          (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1): UpConv(
          (upconv): ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
          (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (conv_final): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)