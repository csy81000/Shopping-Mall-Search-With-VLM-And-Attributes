import numpy as np

from shopping_search.indexing import fuse_embeddings


def test_fused_embeddings_have_unit_norm() -> None:
    image = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    text = np.array([[0.0, 4.0], [5.0, 0.0]], dtype=np.float32)

    combined = fuse_embeddings(image, text, has_text=[True, False])

    assert np.allclose(np.linalg.norm(combined, axis=1), 1.0)
    assert np.allclose(combined[1], [0.0, 1.0])
