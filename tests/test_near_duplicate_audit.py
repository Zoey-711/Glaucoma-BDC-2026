import numpy as np
from PIL import Image

from scripts.audit_near_duplicates import candidates, phash


def test_hash_deterministic_and_radius_search_matches_bruteforce():
    rng = np.random.default_rng(42)
    h = phash(Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)))
    assert h == phash(Image.fromarray(np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)))
    values = [h, h ^ 0b111111, h ^ 0b1111111, 0, 2**64 - 1]
    rows = [dict(phash=f'{v:016x}', mirror_phash=f'{v:016x}') for v in values]
    actual = {(i,j) for i,j,_,_ in candidates(rows)}
    expected = {(i,j) for j in range(len(values)) for i in range(j) if (values[i]^values[j]).bit_count() <= 6}
    assert actual == expected


def test_mirror_candidate_is_detected():
    rows = [dict(phash='aaaaaaaaaaaaaaaa', mirror_phash='1111111111111111'),
            dict(phash='0000000000000000', mirror_phash='aaaaaaaaaaaaaaaa')]
    assert list(candidates(rows)) == [(0, 1, 0, 'mirror')]
