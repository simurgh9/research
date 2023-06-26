from net.mnsit import MNSIT

PATH = '/Users/tfn/Library/CloudStorage/Dropbox/grad/ou/research/code/ann/test/'
mn = MNSIT(path=PATH + 'mnsit_data/')


def test_plot_six():
    assert mn.plot_image(999, source='training') is None


def test_data_shapes():
    X, X_test, y, y_test = mn.get_data()
    assert (60000, 28, 28) == X.shape
    assert (60000, ) == y.shape
    assert (10000, 28, 28) == X_test.shape
    assert (10000, ) == y_test.shape


def test_labels_edges():
    X, X_test, y, y_test = mn.get_data()
    assert (y[:10] == [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]).all()
    assert (y[-10:] == [9, 2, 9, 5, 1, 8, 3, 5, 6, 8]).all()
    assert (y_test[:10] == [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]).all()
    assert (y_test[-10:] == [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]).all()
