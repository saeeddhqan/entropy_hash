import random
import string
import time

import numpy
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from entropy_hash.pipeline.pipeline import EntropyHash
from entropy_hash.simhash.simhash_c import hamming_distance
from entropy_hash.simhash.simhash_c import simhash
from entropy_hash.simhash.simhash_c import simhash_batch
from entropy_hash.util.rand_docs import generate_similar_documents

seed = 1234


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)

device = "cuda"
bit = 64
base_length = 8192
similarities = torch.linspace(0, 1, steps=100)
pairs = {x: [] for x in similarities}
entropy_hash = EntropyHash(device, num_bits=bit)

# Generate pairs of documents
for similarity in similarities:
    similarity = similarity.item()
    pairs[similarity] = tuple(
        generate_similar_documents(
            base_length=base_length,
            num_documents=2,
            similarity=similarity,
            allowed_chars=string.ascii_letters + string.digits + " ",
            random_seed=seed,
        )
    )

overall_simhash = 0.0
overall_entropy_hash = 0.0
true_similarities = []
pred_simhash_similarities = []
pred_entropy_similarities = []

for similarity in similarities:
    similarity = similarity.item()

    num_pairs = 1
    pair = pairs[similarity]

    out = simhash(pair, bit)
    dist_simhash = hamming_distance(out[0], out[1])
    pred_simhash = 1 - (dist_simhash / bit)
    error_simhash = abs(pred_simhash - similarity)
    pred_simhash_similarities.append(pred_simhash)

    out = entropy_hash.pair(pair)
    dist_entropy = (out[0] != out[1]).sum()
    pred_entropy = 1 - (dist_entropy / bit)
    error_entropy_hash = abs(pred_entropy - similarity)
    pred_entropy_similarities.append(pred_entropy.item())

    overall_simhash += 1 - error_simhash
    overall_entropy_hash += 1 - error_entropy_hash
    true_similarities.append(similarity)

mae_simhash = mean_absolute_error(true_similarities, pred_simhash_similarities)
mae_entropy = mean_absolute_error(true_similarities, pred_entropy_similarities)

mse_simhash = mean_squared_error(true_similarities, pred_simhash_similarities)
mse_entropy = mean_squared_error(true_similarities, pred_entropy_similarities)

print(f"SimHash MAE: {mae_simhash}, MSE: {mse_simhash}")
print(f"EntropyHash MAE: {mae_entropy}, MSE: {mse_entropy}")
print("overall accuracy simhash:", overall_simhash / len(similarities))
print("overall accuracy entropy hash:", (overall_entropy_hash / len(similarities)))


iterate = 16
simhash_log = []
entrohash_log = []

for num_docs in (16, 32, 64, 128, 1024, 2048):
    documents = generate_similar_documents(
        base_length=base_length,
        num_documents=num_docs,
        similarity=0.8,
        allowed_chars=string.ascii_letters + string.digits + " ",
        random_seed=seed,
    )
    simhash_time = 0.0
    entropy_hash_time = 0.0
    # warmup
    simhash(documents)
    entropy_hash.batch(documents)
    for _ in range(iterate):
        start_time = time.perf_counter()
        simhash_batch(documents)
        simhash_time += time.perf_counter() - start_time

        start_time = time.perf_counter()
        entropy_hash.batch(documents)
        entropy_hash_time += time.perf_counter() - start_time

    simhash_log.append(simhash_time / iterate)
    entrohash_log.append(entropy_hash_time / iterate)
    print(f"num_docs = {num_docs}, speedup = {simhash_log[-1] / entrohash_log[-1]}")
simhash_sum = sum(simhash_log) / len(simhash_log)
entrohash_sum = sum(entrohash_log) / len(entrohash_log)
print(f"speedup = {simhash_sum / entrohash_sum}")
