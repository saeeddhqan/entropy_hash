import torch


def bin_hash(vector: torch.Tensor):
    binary_hashes = vector > 0
    return binary_hashes


def hamming_distance(hash1, hash2):
    x = hash1 ^ hash2
    return bin(x).count("1")
