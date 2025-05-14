class ResidualBlock(nn.moduke):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input_activity):
        # This helps to deal with vanishing gradients; If a^[l+2] would initially be vanishing, the additon of a^[l] activity addresses the vanishing gradient problem.
        out = self.relu(self.bn1(self.conv1(input_activity)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += input_activity # Skip connection
        out = self.relu(out) # Produces input activity for layer [l+2]
        return out