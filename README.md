
### Reproduce results

```bash
git clone https://github.com/saeeddhqan/entropy_hash
cd entropy_hash
apt install libssl-dev
gcc -shared -o entropy_hash/simhash/simhash/libsimhash_parallel.so -fPIC -fopenmp entropy_hash/simhash/simhash/simhash_parallel.c -lcrypto
pip install -r requirements.txt
python -m entropy_hash.benchmark.synthetic_bench
```

