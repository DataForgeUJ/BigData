"""Build Exact, LSH, and HNSW retrieval indexes."""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hnswlib
import numpy as np
import pandas as pd


# Configuration
EMBEDDING_MODE = "frozen"

EMBEDDING_DIR = (
    Path("data/embeddings") /
    EMBEDDING_MODE
)

INDEX_DIR = (
    Path("artifacts/indexes") /
    EMBEDDING_MODE
)

GALLERY_FRACTIONS = [
    0.25,
    0.50,
    0.75,
    1.00
]

SEED = 42

# LSH configuration
LSH_NUM_BITS = 16
LSH_NUM_TABLES = 10

# HNSW configuration
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100


# Utility functions
def load_embeddings():
    """Load train embeddings and metadata."""

    embeddings_file = (
        EMBEDDING_DIR /
        "train_embeddings.npy"
    )

    metadata_file = (
        EMBEDDING_DIR /
        "train_metadata.csv"
    )

    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Embedding file not found:\n{embeddings_file}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_file}"
        )

    embeddings = np.load(
        embeddings_file
    ).astype(
        np.float32
    )

    metadata = pd.read_csv(
        metadata_file
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Number of embeddings does not match "
            "number of metadata rows."
        )

    return embeddings, metadata


def normalize_embeddings(embeddings):
    """Ensure embeddings have unit L2 norm."""

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(
        norms,
        1e-12
    )

    return embeddings / norms


def create_gallery_subset(
    embeddings,
    metadata,
    fraction,
    seed
):
    """
    Create a deterministic nested gallery subset.

    The 25%, 50%, 75% and 100% galleries are nested
    inside one another.
    """

    rng = np.random.default_rng(seed)

    total = len(embeddings)

    # Create one deterministic ordering.
    ordering = rng.permutation(total)

    gallery_size = int(
        total * fraction
    )

    selected_indices = ordering[:gallery_size]

    selected_embeddings = embeddings[
        selected_indices
    ]

    selected_metadata = metadata.iloc[
        selected_indices
    ].reset_index(drop=True)

    return (
        selected_embeddings,
        selected_metadata,
        selected_indices
    )


# Exact Search
def save_exact_gallery(
    embeddings,
    metadata,
    output_dir
):
    """
    Save the gallery used by Exact Search.

    Exact Search does not require a specialized index.
    Step 05 will directly compare query embeddings
    against this gallery.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_dir / "gallery_embeddings.npy",
        embeddings
    )

    metadata.to_csv(
        output_dir / "gallery_metadata.csv",
        index=False
    )


# LSH
class RandomHyperplaneLSH:
    """
    Random-hyperplane locality-sensitive hashing.

    Each hash table uses random hyperplanes.
    Similar vectors are more likely to produce
    the same hash bucket.
    """

    def __init__(
        self,
        dimension,
        num_bits=16,
        num_tables=10,
        seed=42
    ):

        self.dimension = dimension
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.seed = seed

        rng = np.random.default_rng(seed)

        self.planes = []

        for _ in range(num_tables):

            # Use sparse random hyperplanes.
            #
            # Each bit uses only a small subset
            # of dimensions. This keeps index
            # construction practical for a
            # 2048-dimensional dataset.

            table_planes = np.zeros(
                (
                    num_bits,
                    dimension
                ),
                dtype=np.float32
            )

            dimensions_per_bit = min(
                32,
                dimension
            )

            for bit in range(num_bits):

                selected_dimensions = (
                    rng.choice(
                        dimension,
                        size=dimensions_per_bit,
                        replace=False
                    )
                )

                table_planes[
                    bit,
                    selected_dimensions
                ] = rng.normal(
                    0,
                    1,
                    size=dimensions_per_bit
                )

            self.planes.append(
                table_planes
            )

        self.buckets = []


    def _hash_vectors(
        self,
        vectors,
        planes
    ):
        """Generate integer hash codes."""

        projections = (
            vectors @ planes.T
        )

        bits = projections >= 0

        hash_codes = np.zeros(
            len(vectors),
            dtype=np.uint32
        )

        for bit in range(self.num_bits):

            hash_codes |= (
                bits[:, bit].astype(
                    np.uint32
                )
                << bit
            )

        return hash_codes


    def fit(self, embeddings):
        """Build all LSH hash tables."""

        self.buckets = []

        for table_number, planes in enumerate(
            self.planes
        ):

            hash_codes = self._hash_vectors(
                embeddings,
                planes
            )

            table = {}

            for index, hash_code in enumerate(
                hash_codes
            ):

                key = int(hash_code)

                if key not in table:
                    table[key] = []

                table[key].append(index)

            self.buckets.append(table)

            print(
                "LSH table",
                table_number + 1,
                "/",
                self.num_tables,
                "- buckets:",
                len(table)
            )

        return self


    def query_candidates(
        self,
        query
    ):
        """Return candidate gallery indices."""

        candidates = set()

        query = np.asarray(
            query,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )

        for planes, table in zip(
            self.planes,
            self.buckets
        ):

            hash_code = self._hash_vectors(
                query,
                planes
            )[0]

            bucket = table.get(
                int(hash_code),
                []
            )

            candidates.update(
                bucket
            )

        return sorted(candidates)


# HNSW
def build_hnsw_index(
    embeddings,
    output_file
):
    """Build and save an HNSW index."""

    dimension = embeddings.shape[1]

    index = hnswlib.Index(
        space="cosine",
        dim=dimension
    )

    index.init_index(
        max_elements=len(embeddings),
        M=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION
    )

    index.add_items(
        embeddings,
        np.arange(
            len(embeddings)
        )
    )

    index.set_ef(
        HNSW_EF_SEARCH
    )

    index.save_index(
        str(output_file)
    )

    return index


# Main
def main():

    print()
    print("========================================")
    print("Step 04 - Build Retrieval Indexes")
    print("========================================")

    print("Embedding mode:", EMBEDDING_MODE)
    print("Embedding directory:", EMBEDDING_DIR)
    print("Index directory:", INDEX_DIR)

    # Load embeddings
    print()
    print("Loading frozen train embeddings...")

    embeddings, metadata = load_embeddings()

    print(
        "Embeddings shape:",
        embeddings.shape
    )

    print(
        "Metadata rows:",
        len(metadata)
    )

    # Ensure embeddings are normalized.
    embeddings = normalize_embeddings(
        embeddings
    )

    print(
        "Embedding dimension:",
        embeddings.shape[1]
    )

    # Build each gallery size
    for fraction in GALLERY_FRACTIONS:

        percentage = int(
            fraction * 100
        )

        print()
        print("========================================")
        print(
            f"Building {percentage}% gallery"
        )
        print("========================================")

        gallery_embeddings, gallery_metadata, selected_indices = (
            create_gallery_subset(
                embeddings,
                metadata,
                fraction,
                SEED
            )
        )

        print(
            "Gallery size:",
            len(gallery_embeddings)
        )

        gallery_dir = (
            INDEX_DIR /
            f"gallery_{percentage}"
        )

        exact_dir = (
            gallery_dir /
            "exact"
        )

        lsh_dir = (
            gallery_dir /
            "lsh"
        )

        hnsw_dir = (
            gallery_dir /
            "hnsw"
        )

        exact_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        lsh_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        hnsw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save gallery
        print()
        print("Saving Exact Search gallery...")

        save_exact_gallery(
            gallery_embeddings,
            gallery_metadata,
            exact_dir
        )

        np.save(
            gallery_dir /
            "selected_indices.npy",
            selected_indices
        )

        # Build LSH
        print()
        print("Building LSH index...")

        lsh = RandomHyperplaneLSH(
            dimension=gallery_embeddings.shape[1],
            num_bits=LSH_NUM_BITS,
            num_tables=LSH_NUM_TABLES,
            seed=SEED
        )

        lsh.fit(
            gallery_embeddings
        )

        lsh_file = (
            lsh_dir /
            "lsh_index.pkl"
        )

        with open(
            lsh_file,
            "wb"
        ) as file:

            pickle.dump(
                lsh,
                file
            )

        print(
            "LSH index saved:",
            lsh_file
        )

        # Build HNSW
        print()
        print("Building HNSW index...")

        hnsw_file = (
            hnsw_dir /
            "hnsw_index.bin"
        )

        build_hnsw_index(
            gallery_embeddings,
            hnsw_file
        )

        print(
            "HNSW index saved:",
            hnsw_file
        )

        # Save configuration
        configuration = {
            "embedding_mode": EMBEDDING_MODE,
            "embedding_dimension": int(
                gallery_embeddings.shape[1]
            ),
            "gallery_fraction": fraction,
            "gallery_size": len(
                gallery_embeddings
            ),
            "seed": SEED,
            "lsh": {
                "num_bits": LSH_NUM_BITS,
                "num_tables": LSH_NUM_TABLES
            },
            "hnsw": {
                "space": "cosine",
                "M": HNSW_M,
                "ef_construction": HNSW_EF_CONSTRUCTION,
                "ef_search": HNSW_EF_SEARCH
            }
        }

        with open(
            gallery_dir /
            "configuration.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                configuration,
                file,
                indent=4
            )

        print()
        print(
            f"{percentage}% gallery complete."
        )

    print()
    print("========================================")
    print("Step 04 complete")
    print("========================================")

    print(
        "Indexes saved to:",
        INDEX_DIR
    )


if __name__ == "__main__":
    main()
