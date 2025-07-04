# tests/test_polarization.py

import numpy as np
from nlp_core.polarization import js_divergence

def test_js_divergence_simple():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert js_divergence(p, q) == pytest.approx(0.0)
