# Entropy Hash

**Entropy Hash** is a high-performance algorithm for near-duplicate detection in text. It serves as a fast and more accurate alternative to SimHash.

- ✅ **23% more accurate** than SimHash
- ⚡ **6.6× faster** on synthetic benchmarks
- 🚀 Built on PyTorch for maximum speed and flexibility

---

## Installation

You can install Entropy Hash using:

```bash
pip install entropy-hash
````

If installing locally from source:

```bash
git clone https://github.com/saeeddhqan/entropy_hash.git
cd entropy_hash
pip install -e .
```

---

## Usage

Here's a quick example to get started:

```python
from entropy_hash.pipeline.pipeline import EntropyHash

# Initialize the model
entropy_hash = EntropyHash(device="cuda", num_bits=64)

# Example input
docs = [
    "Deep learning is a subset of machine learning.",
    "Machine learning includes deep learning.",
    "Quantum computing is a different field."
]

# Get raw vectors (PyTorch tensors). Set binarization to True to receive hashed vectors.
vectors = entropy_hash.batch(docs, binarization=False)

# Example: compute cosine similarity
import torch.nn.functional as F

similarity = F.cosine_similarity(vectors[0], vectors[1], dim=0)
print("Similarity:", similarity.item())
```


### Reproduce results

```bash
git clone https://github.com/saeeddhqan/entropy_hash
cd entropy_hash
apt install libssl-dev
gcc -shared -o entropy_hash/simhash/simhash/libsimhash_parallel.so -fPIC -fopenmp entropy_hash/simhash/simhash/simhash_parallel.c -lcrypto
pip install -r requirements.txt
python -m entropy_hash.benchmark.synthetic_bench
```



## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---

## Citation

If you use this library in your research, please cite it as:

```
@misc{entropyhash2025,
  title={EntropyHash: near duplicate detection algorithm},
  author={Saeed Dehqan},
  year={2025},
  howpublished={\url{https://github.com/saeeddhqan/entropy_hash}},
}
```

