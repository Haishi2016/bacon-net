from nets.poly2 import poly2


def test_poly2():
    n = poly2("WHAT")
    assert n.hey() == "hey"
