import numpy as np
import pytest

from bvaluepg.polyagammadensity import sigmoid


def test_sigmoid_scalar_midpoint():
    # Spezialfall: sigmoid(0) = 0.5
    assert sigmoid(0.0) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("x", [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
def test_sigmoid_bounds_scalar(x):
    y = sigmoid(x)
    assert np.isfinite(y)
    assert 0.0 <= y <= 1.0


def test_sigmoid_vector_shape_and_bounds():
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = sigmoid(x)

    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_sigmoid_monotonic_increasing():
    # sigmoid ist monoton steigend
    xs = np.array([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])
    ys = sigmoid(xs)

    # Differenzen müssen >= 0 sein (kleine numerische Toleranz)
    assert np.all(np.diff(ys) >= -1e-15)


def test_sigmoid_extreme_values_no_nan():
    # Extremwerte: sollte nicht NaN werden; Ergebnis sollte sinnvoll saturieren
    y_pos = sigmoid(1000.0)
    y_neg = sigmoid(-1000.0)

    assert np.isfinite(y_pos)
    assert np.isfinite(y_neg)

    # Erwartung: sehr nahe an 1 bzw. 0 (bei Overflow evtl. exakt 1/0 – ist okay)
    assert y_pos >= 1.0 - 1e-12
    assert y_neg <= 1e-12