import numpy as np

from rag_pipeline.embeddings import blob_to_vector, vector_to_blob


def test_vector_blob_round_trip():
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    restored = blob_to_vector(vector_to_blob(vector))
    np.testing.assert_allclose(restored, vector)
