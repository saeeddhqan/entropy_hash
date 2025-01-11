import hashlib
import re
from typing import List
from typing import Tuple


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens


def char_tokenizer(text):
    return [c for c in text]


def hash_token(token: str):
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def compute_simhash(text: str, hash_bits: int = 64, char_tokenize: bool = True):
    tokens = char_tokenizer(text) if char_tokenize else tokenize(text)
    v = [0] * hash_bits

    for token in tokens:
        token_hash = hash_token(token)
        for i in range(hash_bits):
            bitmask = 1 << i
            if token_hash & bitmask:
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hash_bits):
        if v[i] >= 0:
            fingerprint |= 1 << i
    return fingerprint


def simhash(pair: Tuple[str, str], hash_bits: int = 64, char_tokenize: bool = True):
    first = compute_simhash(pair[0], hash_bits, char_tokenize)
    second = compute_simhash(pair[1], hash_bits, char_tokenize)
    return first, second


def simhash_batch(
    docs: List[str],
    char_tokenize: bool = True,
    hash_bits: int = 64,
    inner_parallel: int = 0,
):
    return [compute_simhash(x, hash_bits, char_tokenize) for x in docs]


def hamming_distance(hash1, hash2):
    x = hash1 ^ hash2
    return bin(x).count("1")


if __name__ == "__main__":
    doc1 = "The quick brown fox jumps over the lazy dog"
    doc2 = "The quick brown fox jumps over the lazy dog"
    doc3 = "A fast brown fox leaps over the lazy dog"
    doc4 = "scotland"

    simhash1 = compute_simhash(doc1, char_tokenize=False)
    simhash2 = compute_simhash(doc2, char_tokenize=False)
    simhash3 = compute_simhash(doc3, char_tokenize=False)
    simhash4 = compute_simhash(doc4, char_tokenize=False)
    simhash5 = compute_simhash(doc4, hash_bits=32, char_tokenize=False)

    print(f"SimHash 1: {simhash1:064b}")
    print(f"SimHash 2: {simhash2:064b}")
    print(f"SimHash 3: {simhash3:064b}")
    print(f"SimHash 4: {simhash4:064b}")
    print(f"SimHash 5: {simhash5:032b}")

    print(
        f"Hamming Distance between Doc1 and Doc2: {hamming_distance(simhash1, simhash2)}"
    )
    print(
        f"Hamming Distance between Doc1 and Doc3: {hamming_distance(simhash1, simhash3)}"
    )
    out = simhash_batch(["scotland", "doc2", "doc3"], char_tokenize=False)
    print([f"{x:064b}" for x in out])
