import random
import string
from typing import List
from typing import Tuple

import numpy
import torch

from entropy_hash.model import model
from entropy_hash.util.hashing import bin_hash
from entropy_hash.util.rand_docs import generate_similar_documents
from entropy_hash.util.transform import transform_text

seed = 1234


def set_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)


class EntropyHash:
    def __init__(
        self,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
        context_window: int = 8192,
        num_bits: int = 64,
    ):
        self.device = device
        self.dtype = dtype
        self.context_window = context_window
        self.num_bits = num_bits
        self.network = model.initialize_network(device, context_window, num_bits)
        self.max_doc_shard = 64

    def pair(self, pair: Tuple):
        preprocessed_tensors = []
        for doc in pair:
            tensor = transform_text(
                doc,
                context_window=self.context_window,
                dtype=self.dtype,
                device=self.device,
            )
            preprocessed_tensors.append(tensor)
        batched_tensors = torch.cat(preprocessed_tensors, dim=0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device, dtype=self.dtype
        ):
            output_vectors = self.network(batched_tensors, 2)
            output_vectors = bin_hash(output_vectors)

        return output_vectors

    def batch(self, docs: List[str]):
        preprocessed_tensors = [
            transform_text(doc, context_window=self.context_window, dtype=self.dtype)
            for doc in docs
        ]
        outputs = []
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device, dtype=self.dtype
        ):
            for i in range(0, len(preprocessed_tensors), self.max_doc_shard):
                batch_tensors = preprocessed_tensors[i : i + self.max_doc_shard]
                batch = torch.cat(batch_tensors, dim=0)
                batch = self.network(batch, len(batch_tensors))
                batch = bin_hash(batch)
                outputs.append(batch)

        return [vec for shard in outputs for vec in shard]


if __name__ == "__main__":
    base_length = 10
    num_documents = 10
    similarities = 0.8
    device = "cuda"
    context_window = 1024
    num_bits = 64
    documents = generate_similar_documents(
        base_length=base_length,
        num_documents=num_documents,
        similarity=similarities,
        allowed_chars=string.ascii_letters + string.digits + " ",
        random_seed=seed,
    )
    network = model.initialize_network(device, context_window, num_bits)

    preprocessed_tensors = []
    for doc in documents:
        tensor = transform_text(
            doc, context_window=context_window, normalization="-1-1"
        )
        preprocessed_tensors.append(tensor)

    batched_tensors = torch.cat(
        preprocessed_tensors, dim=0
    )  # Shape: (num_documents, 1024)

    batched_tensors = batched_tensors.to(device)

    with torch.no_grad():
        output_vectors = network(batched_tensors, num_documents)
    # Display the document vectors
    for idx in range(num_documents):
        print(f"Document {idx + 1}")
        print(f"\ttext: {documents[idx]}")
        print(
            f"\tsimilarity:",
            [
                round(
                    hamming_distance_tensor(
                        output_vectors[idx], output_vectors[x]
                    ).item(),
                    4,
                )
                for x in range(num_documents)
            ],
        )
        print(f"\tvector:", output_vectors[idx][:5])
        print("-" * 80)
