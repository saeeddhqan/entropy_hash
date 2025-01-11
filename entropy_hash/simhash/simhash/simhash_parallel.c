#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define HASH_BITS 64
#define MAX_TOKEN_LENGTH 256
#define MAX_TOKENS 100000

// Structure to hold the MD5 hash as an integer
#include <openssl/evp.h>

typedef unsigned long long uint64;

uint64_t hash_token(const char* token) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);
    EVP_DigestUpdate(ctx, token, strlen(token));
    EVP_DigestFinal_ex(ctx, digest, &digest_len);
    EVP_MD_CTX_free(ctx);

    uint64 hash = 0;
    memcpy(&hash, digest, sizeof(uint64));
    return hash;
}

// Function to tokenize text
char** tokenize(const char* text, int* num_tokens) {
    char** tokens = malloc(MAX_TOKENS * sizeof(char*));
    char* text_copy = strdup(text);
    char* token = strtok(text_copy, " \t\n\r");
    *num_tokens = 0;
    
    while (token != NULL && *num_tokens < MAX_TOKENS) {
        // Convert to lowercase and remove non-alphanumeric characters
        char* clean_token = malloc(MAX_TOKEN_LENGTH);
        int j = 0;
        for (int i = 0; token[i]; i++) {
            if (isalnum(token[i])) {
                clean_token[j++] = tolower(token[i]);
            }
        }
        clean_token[j] = '\0';
        
        if (j > 0) {
            tokens[*num_tokens] = clean_token;
            (*num_tokens)++;
        } else {
            free(clean_token);
        }
        token = strtok(NULL, " \t\n\r");
    }
    
    free(text_copy);
    return tokens;
}

char** char_tokenizer(const char* text, int* num_tokens) {
    // Count the length of the input string
    int length = strlen(text);

    // Allocate an array of char* (one for each character)
    // Here, each token is effectively a 1-character "string" + null terminator
    char** tokens = (char**)malloc(length * sizeof(char*));
    if (!tokens) {
        *num_tokens = 0;
        return NULL;
    }

    for (int i = 0; i < length; i++) {
        // Allocate space for 2 chars: the character + '\0'
        tokens[i] = (char*)malloc(2 * sizeof(char));
        if (!tokens[i]) {
            // If allocation fails, free everything allocated so far
            for (int j = 0; j < i; j++) {
                free(tokens[j]);
            }
            free(tokens);
            *num_tokens = 0;
            return NULL;
        }
        tokens[i][0] = text[i];
        tokens[i][1] = '\0';
    }

    *num_tokens = length;
    return tokens;
}

// Main SimHash computation function
uint64_t compute_simhash(const char* text, int char_tokenize, int hash_bits, int inner_parallel) {
    // Dynamically allocate array for bit counts
    int* v = (int*)calloc(hash_bits, sizeof(int));
    if (!v) {
        fprintf(stderr, "Allocation failure for v.\n");
        return 0ULL;
    }

    int num_tokens;
    char** tokens;
    if (char_tokenize == 0) {
        tokens = tokenize(text, &num_tokens);
    } else {
        tokens = char_tokenizer(text, &num_tokens);
    }

    if(inner_parallel == 1){
        // Parallel loop over tokens
        #pragma omp parallel for schedule(dynamic) reduction(+:v[:hash_bits])
        for (int i = 0; i < num_tokens; i++) {
            uint64_t token_hash = hash_token(tokens[i]);
            
            // For each bit, increment or decrement
            for (int j = 0; j < hash_bits; j++) {
                if (token_hash & ((uint64_t)1 << j)) {
                    v[j]++;
                } else {
                    v[j]--;
                }
            }
            // Each thread frees its own token
            free(tokens[i]);
        }
    } else {
        for (int i = 0; i < num_tokens; i++) {
            uint64_t token_hash = hash_token(tokens[i]);
            
            // For each bit, increment or decrement
            for (int j = 0; j < hash_bits; j++) {
                if (token_hash & ((uint64_t)1 << j)) {
                    v[j]++;
                } else {
                    v[j]--;
                }
            }
            // Each thread frees its own token
            free(tokens[i]);
        }
    }
    // Free the array of token pointers
    free(tokens);

    // Compute final fingerprint
    uint64_t fingerprint = 0ULL;
    for (int i = 0; i < hash_bits; i++) {
        if (v[i] >= 0) {
            fingerprint |= ((uint64_t)1 << i);
        }
    }

    free(v);
    return fingerprint;
}

uint64_t* compute_simhash_batch(
    char** texts,
    int num_texts,
    int char_tokenize,
    int hash_bits,
    int inner_parallel
) {
    uint64_t* results = (uint64_t*)malloc(num_texts * sizeof(uint64_t));
    if (!results) {
        fprintf(stderr, "Failed to allocate memory for results array\n");
        return NULL; 
    }

    #pragma omp parallel for
    for (int i = 0; i < num_texts; i++) {
        results[i] = compute_simhash(texts[i], char_tokenize, hash_bits, inner_parallel);
    }

    return results;
}


void free_simhash_batch_results(uint64_t* ptr) {
    free(ptr);
}

// Function to compute Hamming distance
int hamming_distance(uint64 hash1, uint64 hash2) {
    uint64 x = hash1 ^ hash2;
    int distance = 0;
    while (x) {
        distance += x & 1;
        x >>= 1;
    }
    return distance;
}
