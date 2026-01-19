import torch # type: ignore
import torch.nn as nn # type: ignore

HIDDEN_SIZE = 64
DROPOUT_RATE = 0.2
LEAKY_SLOPE = 0.2

class Discriminator(nn.Module):
    def __init__(self, input_size=15):
        super(Discriminator, self).__init__()
    
        self.model = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.LeakyReLU(LEAKY_SLOPE),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_SIZE // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    disc = Discriminator(input_size=15)
    test_input = torch.randn(1, 15)
    validity = disc(test_input)
    
    print("--- Discriminator Initialized ---")
    print(f"Input Shape: {test_input.shape}")
    print(f"Validity Score (0=Fake, 1=Real): {validity.item():.4f}")