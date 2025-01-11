from ctypes import *
from typing import List
from typing import Tuple

_lib = CDLL("entropy_hash/simhash/simhash/libsimhash_parallel.so")

_lib.compute_simhash.argtypes = [
    c_char_p,
    c_int,
    c_int,
    c_int,
]
_lib.compute_simhash.restype = c_uint64

_lib.hamming_distance.argtypes = [c_uint64, c_uint64]
_lib.hamming_distance.restype = c_int

_lib.compute_simhash_batch.argtypes = [POINTER(c_char_p), c_int, c_int, c_int, c_int]
_lib.compute_simhash_batch.restype = POINTER(c_uint64)
_lib.free_simhash_batch_results.argtypes = [POINTER(c_uint64)]
_lib.free_simhash_batch_results.restype = None


def compute_simhash(text: str, hash_bits: int = 64, char_tokenize: bool = True):
    return _lib.compute_simhash(text.encode("utf-8"), int(char_tokenize), hash_bits, 1)


def simhash(pair: Tuple[str, str], hash_bits: int = 64, char_tokenize: bool = True):
    first = compute_simhash(pair[0], hash_bits, int(char_tokenize))
    second = compute_simhash(pair[1], hash_bits, int(char_tokenize))
    return first, second


def simhash_batch(
    docs: List[str],
    char_tokenize: bool = True,
    hash_bits: int = 64,
    inner_parallel: int = 0,
):
    num_texts = len(docs)
    c_array_type = c_char_p * num_texts
    c_texts = c_array_type(*[s.encode("utf-8") for s in docs])

    result_ptr = _lib.compute_simhash_batch(
        c_texts, num_texts, int(char_tokenize), hash_bits, inner_parallel
    )
    results = [result_ptr[i] for i in range(num_texts)]

    _lib.free_simhash_batch_results(result_ptr)

    return results


def hamming_distance(hash1, hash2):
    return _lib.hamming_distance(hash1, hash2)


if __name__ == "__main__":
    doc1 = "The quick brown fox jumps over the lazy dog"
    doc2 = "The quick brown fox jumps over the lazy dog"
    doc3 = "A fast brown fox leaps over the lazy dog"
    doc4 = "scotland"

    simhash1 = compute_simhash(doc1, char_tokenize=False)
    simhash2 = compute_simhash(doc2, char_tokenize=False)
    simhash3 = compute_simhash(doc3, char_tokenize=False)
    simhash4 = compute_simhash(doc4, char_tokenize=False)
    simhash5 = compute_simhash(doc4, hash_bits=128, char_tokenize=False)

    print(f"SimHash 1: {simhash1:064b}")
    print(f"SimHash 2: {simhash2:064b}")
    print(f"SimHash 3: {simhash3:064b}")
    print(f"SimHash 4: {simhash4:064b}")
    print(f"SimHash 5: {simhash5:0128b}")

    print(
        f"Hamming Distance between Doc1 and Doc2: {hamming_distance(simhash1, simhash2)}"
    )
    print(
        f"Hamming Distance between Doc1 and Doc3: {hamming_distance(simhash1, simhash3)}"
    )
    out = simhash_batch(["scotland", "doc2", "doc3"], char_tokenize=False)
    print([f"{x:064b}" for x in out])
