from chainer import Chain
from chainer import links as L
from chainer import functions as F

class CombNet(Chain):
    def __init__(self):
        super(CombNet, self).__init__()
        with self.init_scope():
            self.nn1 = L.Linear(1024, 1024)
            self.nn2 = L.Linear(1024, 512)
            self.nn3 = L.Linear(512, 256)
            self.