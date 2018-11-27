import os
import unittest

from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics import SeqLabelEvaluator
from fastNLP.core.field import TextField, LabelField
from fastNLP.core.instance import Instance
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.models.sequence_modeling import SeqLabeling

import fastNLP.core.loss as loss
import math
import torch as tc
import pdb

class TestLoss(unittest.TestCase):

	def test_case_1(self):
		#验证nllloss的原理

		print (".----------------------------------")

		loss_func = loss.Loss("nll")

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
		los = loss_func(y , gy)

		r = -math.log(.3) - math.log(.3) - math.log(.1)
		r /= 3
		print ("loss = %f" % (los))
		print ("r = %f" % (r))

		self.assertEqual(int(los * 1000), int(r * 1000))

	def test_case_2(self):
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
		los = loss_func(y , gy)
		print ("loss = %f" % (los))

		r = -log(.3) - log(.3) - log(.1) - log(.3) - log(.7) - log(.1)
		r /= 6
		print ("r = %f" % (r))

		self.assertEqual(int(los * 1000), int(r * 1000))

	def test_case_3(self):
		#验证pack_padded_sequence()的正确性
		print ("----------------------------------")

		log = math.log

		loss_func = loss.Loss("nll")

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
		los = loss_func(yy , gyy)
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

if __name__ == "__main__":
	unittest.main()
