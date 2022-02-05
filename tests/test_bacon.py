from bacon import baconNet


def test_explain():
    n = baconNet("hey")
    assert n.explain() == "explained!"
