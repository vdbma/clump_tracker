from clump_tracker.clumps import Clump


def test_clump():
    c = Clump(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7, 8.0, 9.0)
    assert c.coords == (0.0, 1.0, 2.0)
