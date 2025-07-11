import torch
from torch import nn
from model.backbone import Patchify, Reconstruct


def main():
    # Assuming Patchify and Reconstruct are already defined above
    # Example input: batch of images (B, C, H, W)
    x = torch.randn(2, 3, 8, 8)  # shape: (batch_size=2, channels=3, height=8, width=8)
    patch_size = (3, 2, 2)
    dim_index = [1, 2, 3]  # Apply patching on height and width

    # Instantiate modules
    patchify = Patchify(patch_size, dim_index)
    reconstruct = Reconstruct([1] + list(x.shape)[1:], dim_index)

    # Remove the 'exit()'s in the Patchify and Reconstruct forward methods before testing
    patches = patchify(x)
    print("Patched shape:", patches.shape)

    reconstructed = reconstruct(patches)
    print("Reconstructed shape:", reconstructed.shape)

    # Check correctness
    if torch.allclose(x, reconstructed, atol=1e-5):
        print("✅ Reconstruction successful, outputs match.")
    else:
        print("❌ Reconstruction failed, outputs do not match.")
    return



if __name__ == "__main__":
    main()