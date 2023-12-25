import unittest
import numpy as np
import mlp
from numpy.testing import assert_allclose
import pickle as pk

seed = 10417617

with open("tests.pk","rb") as f: tests = pk.load(f)

TOLERANCE = 1e-5

# to run one test: python -m unittest tests.TestLinearMap
# to run all tests: python -m unittest tests

class TestLinearMap(unittest.TestCase):
  def test(self):
    weights,bias,result = tests[0]
    sl = mlp.LinearMap(18,100,alpha=0)
    sl.loadparams(np.array(weights),np.array(bias))

    test1 = np.arange(18).reshape((18,1))
    assert_allclose(sl.forward(test1),result,atol=TOLERANCE)

    test2 = np.arange(100).reshape((100,1))
    result = tests[1][2]
    assert_allclose(sl.backward(test2),result,atol=TOLERANCE)
    sl.zerograd()

    test3 = np.arange(36).reshape((18,2))
    result = tests[2][2]
    assert_allclose(sl.forward(test3),result,atol=TOLERANCE)

    test4 = np.arange(200).reshape((100,2))
    assert_allclose(sl.backward(test4),tests[3][2],atol=TOLERANCE)

class TestReLUDropout(unittest.TestCase):
  def test(self):
    idstr = 'reludropout'
    sl = mlp.ReLU(dropout_probability=0.5)

    test9 = (np.arange(36).reshape((18,2))-18).astype('float64')
    assert_allclose(sl.forward(test9,train=False),tests[idstr+"test7"],atol=TOLERANCE)
    np.random.seed(seed)
    assert_allclose(sl.forward(test9,train=True),tests[idstr+"test7.5"],atol=TOLERANCE)

    test8=(np.arange(36).reshape((18,2))-22).astype('float64')
    assert_allclose(sl.backward(test8),tests[idstr+"test8"])

class TestMomentum(unittest.TestCase):
  def test(self):
    idstr = 'momentum'
    sl = mlp.LinearMap(18,100,alpha=0.5,lr=0.01)
    weights,bias = tests[idstr+"wb"]
    
    sl.loadparams(np.array(weights),np.array(bias))
    test5 = np.arange(36).reshape((18,2))
    
    assert_allclose(sl.forward(test5),tests[idstr+"test5"],atol=TOLERANCE)
    
    test6 = np.arange(200).reshape((100,2))
    sl.backward(test6)
    sl.step()
    sl.zerograd()
    sl.backward(test6)
    sl.step()
    assert_allclose(sl.getW(),tests[idstr+"test6"],atol=TOLERANCE)
    assert_allclose(sl.getb(),tests[idstr+"test7"],atol=TOLERANCE)
    
class TestReLU(unittest.TestCase):
  def test(self):
    idstr = 'relu'
    sl = mlp.ReLU(dropout_probability=0)
    
    test7 = (np.arange(36).reshape((18,2))-18).astype('float64')
    assert_allclose(sl.forward(test7,train=False),tests[idstr+"test7"])
    assert_allclose(sl.forward(test7),tests[idstr+"test7.5"])
    
    test8 = np.arange(36).reshape((18,2))-22
    assert_allclose(sl.backward(test8),tests[idstr+"test8"])
    

class TestLoss(unittest.TestCase):
  def test(self):
    sl = mlp.SoftmaxCrossEntropyLoss()

    np.random.seed(1)
    logits = np.random.uniform(-1,1,[18,2])
    labels = np.zeros(logits.shape)
    labels[3,0], labels[15,1] = 1, 1

    tests[8] = 3.341601237187909
    tests[9] = np.array([[ 0.02616557,  0.03451958],
                        [ 0.01136603,  0.01496244],
                        [ 0.01523983,  0.00983114],
                        [-0.48350725,  0.0163136 ],
                        [ 0.02512681,  0.02401098],
                        [ 0.02627951,  0.03217908],
                        [ 0.01710387,  0.0473285 ],
                        [ 0.01200323,  0.03124354],
                        [ 0.02618037,  0.02498454],
                        [ 0.01504693,  0.01214698],
                        [ 0.05636732,  0.05667883],
                        [ 0.02126896,  0.03263949],
                        [ 0.06557396,  0.04891534],
                        [ 0.01347032,  0.00883735],
                        [ 0.01595961,  0.04733087],
                        [ 0.01383351, -0.48102556],
                        [ 0.07718302,  0.02374111],
                        [ 0.04533841,  0.01536218]])
    assert_allclose(sl.forward(logits, labels),tests[8],atol=TOLERANCE)
    assert_allclose(sl.backward(),tests[9],atol=TOLERANCE)



class TestSingleLayerMLP(unittest.TestCase):
  def test(self):
    idstr = 'slst'
    data = [np.arange(20).reshape((20,1)),np.arange(20).reshape((20,1))-1]
    ann = mlp.SingleLayerMLP(20,19,hiddenlayer=100,alpha=0.1,dropout_probability=0.5)

    Ws,bs = tests[idstr+"wb"]
    Ws = [np.array(W) for W in Ws]
    bs = [np.array(b) for b in bs]
    ann.loadparams(Ws,bs)

    np.random.seed(seed)
    ann.forward(data[0])
    ann.backward(np.arange(19).reshape((19,1)))
    ann.step()
    for ind,W in enumerate(ann.getWs()):
      assert_allclose(tests[idstr+"0resultW"+str(ind)],W)
    for ind,b in enumerate(ann.getbs()):
      assert_allclose(tests[idstr+"0resultb"+str(ind)],b)
    ann.zerograd()
    ann.backward(np.arange(19).reshape((19,1))+1)
    ann.step()
    for ind,W in enumerate(ann.getWs()):
      assert_allclose(tests[idstr+"1resultW"+str(ind)],W)
    for ind,b in enumerate(ann.getbs()):
      assert_allclose(tests[idstr+"1resultb"+str(ind)],b)

class TestTwoLayerMLP(unittest.TestCase):
  def test(self):
    idstr = 'tlst'
    data = [np.arange(20).reshape((20,1)),np.arange(20).reshape((20,1))-1]
    ann = mlp.TwoLayerMLP(20,19,hiddenlayers=[100,100],alpha=0.1,dropout_probability=0.5)
    Ws,bs = tests[idstr+"wb"]
    ann.loadparams(Ws,bs)

    np.random.seed(seed)
    ann.forward(data[0])
    ann.backward(np.arange(19).reshape((19,1)))
    ann.step()
    for ind,W in enumerate(ann.getWs()):
      assert_allclose(tests[idstr+"0resultW"+str(ind)],W)
    for ind,b in enumerate(ann.getbs()):
      assert_allclose(tests[idstr+"0resultb"+str(ind)],b)

    ann.zerograd()
    ann.backward(np.arange(19).reshape((19,1))+1)
    ann.step()
    for ind,W in enumerate(ann.getWs()):
      assert_allclose(tests[idstr+"1resultW"+str(ind)],W)
    for ind,b in enumerate(ann.getbs()):
      assert_allclose(tests[idstr+"1resultb"+str(ind)],b)