import torch


def transform_text(
    text: str, context_window: int = 1024, dtype=torch.bfloat16, device="cuda"
) -> torch.Tensor:
    ascii_bytes = text.encode("ascii", errors="ignore")

    ascii_codes = torch.tensor(list(ascii_bytes), dtype=torch.uint8)

    ascii_codes = ascii_codes.to(dtype=torch.float32)
    ascii_codes.div_(63.5)
    ascii_codes.sub_(1.0)
    ascii_codes = ascii_codes.to(dtype=dtype)

    n = ascii_codes.size(0)
    if n == 0:
        return ascii_codes.to(device)

    remainder = n % context_window
    if remainder != 0:
        padding_size = context_window - remainder
        padding = torch.zeros(padding_size, dtype=dtype)
        ascii_codes = torch.cat([ascii_codes, padding], dim=0)

    ascii_codes = ascii_codes.view(-1, context_window)
    return ascii_codes.to(device)


if __name__ == "__main__":
    sample_text = (
        "Hello, World! This is a sample document to test the preprocessing function."
    )
    tensor = transform_text(sample_text, context_window=1024)
    print(f"Shape of the tensor: {tensor.shape}")
    print(f"Tensor content:\n{tensor}")
