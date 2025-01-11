import random
import string
from typing import List


def generate_similar_documents(
    base_length: int,
    num_documents: int,
    similarity: float,
    allowed_chars: str = string.printable.strip(),
    random_seed: int = 1245,
) -> List[str]:

    if not (0.0 <= similarity <= 1.0):
        raise ValueError("Similarity must be between 0.0 and 1.0.")

    if base_length <= 0:
        raise ValueError("Base length must be a positive integer.")

    if num_documents <= 0:
        raise ValueError("Number of documents must be a positive integer.")

    if random_seed is not None:
        random.seed(random_seed)

    if not allowed_chars:
        raise ValueError("Allowed characters set cannot be empty.")

    base_document = "".join(random.choices(allowed_chars, k=base_length))

    documents = [base_document]

    num_variants = num_documents - 1
    if num_variants < 1:
        return documents[:num_documents]

    for _ in range(num_variants):
        if similarity == 1.0:
            variant = base_document
        elif similarity == 0.0:
            variant = "".join(random.choices(allowed_chars, k=base_length))
        else:
            variant_chars = list(base_document)
            num_substitutions = int(round((1 - similarity) * base_length, 15))
            substitution_indices = random.sample(range(base_length), num_substitutions)

            for idx in substitution_indices:
                original_char = variant_chars[idx]
                possible_replacements = allowed_chars.replace(original_char, "")
                if not possible_replacements:
                    continue
                variant_chars[idx] = random.choice(possible_replacements)

            variant = "".join(variant_chars)

        documents.append(variant)

    return documents


if __name__ == "__main__":
    base_length = 10
    num_documents = 3
    similarity = 0.9
    random_seed = 1245

    documents = generate_similar_documents(
        base_length=base_length,
        num_documents=num_documents,
        similarity=similarity,
        allowed_chars=string.ascii_letters + string.digits + " ",
        random_seed=random_seed,
    )

    for idx, doc in enumerate(documents):
        print(f"Document {idx + 1} (Similarity to base: {similarity * 100}%):")
        print(doc)
        print("-" * 80)
