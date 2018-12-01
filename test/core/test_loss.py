import math
import unittest

import torch as tc
import torch.nn.functional as F

import fastNLP.core.losses as loss


class TestLoss(unittest.TestCase):

	def test_case_1(self):
		#验证nllloss的原理

		print (".----------------------------------")

		# loss_func = loss.Loss("nll")
		print(callable(tc.nn.NLLLoss))
		loss_func = loss.NewLoss(F.nll_loss)

		nll_loss = loss.NLLLoss()

		#pdb.set_trace()

		y = tc.Tensor(
				[
					[.3,.4,.3],
					[.5,.3,.2],
					[.3,.6,.1],
				]
			)

		gy = tc.LongTensor(
				[
					0,
					1,
					2,
				]
			)


		y = tc.log(y)
		los = loss_func({'input': y}, {'target': gy})
		losses = nll_loss({'input': y}, {'target': gy})

		r = -math.log(.3) - math.log(.3) - math.log(.1)
		r /= 3
		print ("loss = %f" % (los))
		print ("r = %f" % (r))
		print ("nll_loss = %f" % (losses))

		self.assertEqual(int(los * 1000), int(r * 1000))

	def _test_case_2(self):
		#验证squash()的正确性
		print ("----------------------------------")

		log = math.log

		loss_func = loss.Loss("nll")

		#pdb.set_trace()

		y = tc.Tensor(
				[
					[[.3,.4,.3],[.3,.4,.3],],
					[[.5,.3,.2],[.1,.2,.7],],
					[[.3,.6,.1],[.2,.1,.7],],
				]
			)

		gy = tc.LongTensor(
				[
					[0,2],
					[1,2],
					[2,1],
				]
			)


		#pdb.set_trace()

		y = tc.log(y)
		#los = loss_func({'input': y}, {'target': gy})
		los = loss_func(y, gy)
		print ("loss = %f" % (los))

		r = -log(.3) - log(.3) - log(.1) - log(.3) - log(.7) - log(.1)
		r /= 6
		print ("r = %f" % (r))

		self.assertEqual(int(los * 1000), int(r * 1000))

	def test_case_3(self):
		#验证pack_padded_sequence()的正确性
		print ("----------------------------------")

		log = math.log

		#loss_func = loss.Loss("nll")
		loss_func = loss.NLLLoss()

		#pdb.set_trace()

		y = tc.Tensor(
				[
					[[.3,.4,.3],[.3,.2,.5],[.4,.5,.1,],],
					[[.5,.3,.2],[.1,.2,.7],[.0,.0,.0,],],
					[[.3,.6,.1],[.0,.0,.0],[.0,.0,.0,],],
				]
			)

		gy = tc.LongTensor(
				[
					[0,2,1,],
					[1,2,0,],
					[2,0,0,],
				]
			)

		lens = [3,2,1]

		#pdb.set_trace()

		y = tc.log(y)

		yy = tc.nn.utils.rnn.pack_padded_sequence(y , lens , batch_first = True).data
		gyy = tc.nn.utils.rnn.pack_padded_sequence(gy , lens , batch_first = True).data
		los = loss_func({'input': yy}, {'target': gyy})
		print ("loss = %f" % (los))


		r = -log(.3) - log(.5) - log(.5) - log(.3) - log(.7) - log(.1)
		r /= 6
		print ("r = %f" % (r))

		self.assertEqual(int(los * 1000), int(r * 1000))

	def test_case_4(self):
		#验证unpad()的正确性
		print ("----------------------------------")

		log = math.log

		#pdb.set_trace()

		y = tc.Tensor(
				[
					[[.3,.4,.3],[.3,.2,.5],[.4,.5,.1,],[.6,.3,.1,],],
					[[.5,.3,.2],[.1,.2,.7],[.0,.0,.0,],[.0,.0,.0,],],
					[[.3,.6,.1],[.0,.0,.0],[.0,.0,.0,],[.0,.0,.0,],],
				]
			)

		gy = tc.LongTensor(
				[
					[0,2,1,2,],
					[1,2,0,0,],
					[2,0,0,0,],
				]
			)

		lens = [4,2,1]

		#pdb.set_trace()

		y = tc.log(y)

		loss_func = loss.Loss("nll" , pre_pro = ["unpad"])
		los = loss_func(y , gy , lens = lens)
		print ("loss = %f" % (los))


		r = -log(.1) -log(.3) - log(.5) - log(.5) - log(.3) - log(.7) - log(.1)
		r /= 7
		print ("r = %f" % (r))


		self.assertEqual(int(los * 1000), int(r * 1000))

	def test_case_5(self):
		#验证mask()和make_mask()的正确性
		print ("----------------------------------")

		log = math.log

		#pdb.set_trace()

		y = tc.Tensor(
				[
					[[.5,.3,.2],[.1,.2,.7],[.0,.0,.0,],[.0,.0,.0,],],
					[[.5,.4,.1],[.3,.2,.5],[.4,.5,.1,],[.6,.1,.3,],],
					[[.3,.6,.1],[.3,.2,.5],[.0,.0,.0,],[.0,.0,.0,],],
				]
			)

		gy = tc.LongTensor(
				[
					[1,2,0,0,],
					[0,2,1,2,],
					[2,1,0,0,],
				]
			)

		mask = tc.ByteTensor(
				[
					[1,1,0,0,],
					[1,1,1,1,],
					[1,1,0,0,],
				]
			)

		y = tc.log(y)

		lens = [2,4,2]

		loss_func = loss.Loss("nll" , pre_pro = ["mask"])
		los = loss_func(y , gy , mask = mask)
		print ("loss = %f" % (los))

		los2 = loss_func(y , gy , mask = loss.make_mask(lens,gy.size()[-1]))
		print ("loss2 = %f" % (los2))


		r = -log(.3) -log(.7) - log(.5) - log(.5) - log(.5) - log(.3) - log(.1) - log(.2)
		r /= 8
		print ("r = %f" % (r))


		self.assertEqual(int(los * 1000), int(r * 1000))
		self.assertEqual(int(los2 * 1000), int(r * 1000))

	def test_case_6(self):
		#验证unpad_mask()的正确性
		print ("----------------------------------")

		log = math.log

		#pdb.set_trace()

		y = tc.Tensor(
				[
					[[.3,.4,.3],[.3,.2,.5],[.4,.5,.1,],[.6,.3,.1,],],
					[[.5,.3,.2],[.1,.2,.7],[.0,.0,.0,],[.0,.0,.0,],],
					[[.3,.6,.1],[.0,.0,.0],[.0,.0,.0,],[.0,.0,.0,],],
				]
			)

		gy = tc.LongTensor(
				[
					[0,2,1,2,],
					[1,2,0,0,],
					[2,0,0,0,],
				]
			)

		lens = [4,2,1]

		#pdb.set_trace()

		y = tc.log(y)

		loss_func = loss.Loss("nll" , pre_pro = ["unpad_mask"])
		los = loss_func(y , gy , lens = lens)
		print ("loss = %f" % (los))


		r = -log(.1) -log(.3) - log(.5) - log(.5) - log(.3) - log(.7) - log(.1)
		r /= 7
		print ("r = %f" % (r))

		self.assertEqual(int(los * 1000), int(r * 1000))

	def test_case_7(self):
		#验证一些其他东西
		print ("----------------------------------")

		log = math.log

		#pdb.set_trace()

		y = tc.Tensor(
				[
					[[.3,.4,.3],[.3,.2,.5],[.4,.5,.1,],[.6,.3,.1,],],
					[[.5,.3,.2],[.1,.2,.7],[.0,.0,.0,],[.0,.0,.0,],],
					[[.3,.6,.1],[.0,.0,.0],[.0,.0,.0,],[.0,.0,.0,],],
				]
			)

		gy = tc.LongTensor(
				[
					[0,2,1,2,],
					[1,2,0,0,],
					[2,0,0,0,],
				]
			)

		lens = [4,2,1]

		#pdb.set_trace()

		y = tc.log(y)

		loss_func = loss.Loss("nll" , pre_pro = [] , weight = tc.Tensor([1,1,0]))
		loss_func.add_pre_pro("unpad_mask")
		los = loss_func(y , gy , lens = lens)
		print ("loss = %f" % (los))


		r = - log(.3) - log(.5) - log(.3)
		r /= 3
		print ("r = %f" % (r))
		self.assertEqual(int(los * 1000), int(r * 1000))

	def test_case_8(self):
		def func(a, b):
			import torch.nn.functional as F
			return F.cross_entropy(a, b)

		def func2(a, truth):
			return func(a, truth)

		def func3(predict, truth):
			return func(predict, truth)

		def func4(a, b, c=2):
			return (a + b) * c

		def func6(a, b, **kwargs):
			c = kwargs['c']
			return (a + b) * c

		import torch
		from fastNLP.core.losses import LossBase, NewLoss

		get_loss = NewLoss(func, {'a': 'predict', 'b': 'truth'})
		predict = torch.randn(5, 3)
		truth = torch.LongTensor([1, 0, 1, 2, 1])
		loss1 = get_loss({'predict': predict}, {'truth': truth})
		get_loss_2 = NewLoss(func2, {'a': 'predict'})
		loss2 = get_loss_2({'predict': predict}, {'truth': truth})
		get_loss_3 = NewLoss(func3)
		loss3 = get_loss_3({'predict': predict}, {'truth': truth})
		print(loss1, loss2, loss3)
		assert loss1 == loss2 and loss1 == loss3

		get_loss_4 = NewLoss(func4)
		loss4 = get_loss_4({'a': 1, 'b': 3}, {})
		print(loss4)
		assert loss4 == (1 + 3) * 2

		get_loss_5 = NewLoss(func4)
		loss5 = get_loss_5({'a': 1, 'b': 3}, {'c': 4})
		print(loss5)
		assert loss5 == (1 + 3) * 4

		get_loss_6 = NewLoss(func6)
		loss6 = get_loss_6({'a': 1, 'b': 3}, {'c': 4})
		print(loss6)
		assert loss6 == (1 + 3) * 4

		get_loss_7 = NewLoss(func6, c='cc')
		loss7 = get_loss_7({'a': 1, 'b': 3}, {'cc': 4})
		print(loss7)
		assert loss7 == (1 + 3) * 4


if __name__ == "__main__":
	unittest.main()
