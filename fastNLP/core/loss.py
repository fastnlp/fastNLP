import torch

def squash(predict , truth , **kwargs):
	'''To reshape tensors in order to fit Loss functions in pytorch

	:param predict	: Tensor, model output
	:param truth	: Tensor, truth from dataset
	:param **kwargs : extra arguments

	:return predict , truth: predict & truth after processing
	'''
	return predict.view(-1 , predict.size()[-1]) , truth.view(-1,)

def unpad(predict , truth , **kwargs):
	'''To process padded sequence output to get true loss
	Using pack_padded_sequence() method
	This method contains squash()

	:param predict	: Tensor, [batch_size , max_len , tag_size]
	:param truth	: Tensor, [batch_size , max_len]
	:param **kwargs : extra arguments, kwargs["lens"] is expected to be exsist
		kwargs["lens"] : list or LongTensor, [batch_size]
					  the i-th element is true lengths of i-th sequence  
	
	:return predict , truth: predict & truth after processing
	'''
	if kwargs.get("lens") is None:
		return predict , truth
	lens = torch.LongTensor(kwargs["lens"])
	lens , idx = torch.sort(lens , descending = True)
	predict = torch.nn.utils.rnn.pack_padded_sequence(predict[idx] , lens , batch_first = True).data
	truth = torch.nn.utils.rnn.pack_padded_sequence(truth[idx] , lens , batch_first = True).data
	return predict , truth

def unpad_mask(predict , truth , **kwargs):
	'''To process padded sequence output to get true loss
	Using mask() method
	This method contains squash()

	:param predict	: Tensor, [batch_size , max_len , tag_size]
	:param truth	: Tensor, [batch_size , max_len]
	:param **kwargs : extra arguments, kwargs["lens"] is expected to be exsist
		kwargs["lens"] : list or LongTensor, [batch_size]
					  the i-th element is true lengths of i-th sequence  
	
	:return predict , truth: predict & truth after processing
	'''
	if kwargs.get("lens") is None:
		return predict , truth
	mas = make_mask(kwargs["lens"] , truth.size()[1])
	return mask(predict , truth , mask = mas)

def mask(predict , truth , **kwargs):
	'''To select specific elements from Tensor
	This method contains squash()

	:param predict	: Tensor, [batch_size , max_len , tag_size]
	:param truth	: Tensor, [batch_size , max_len]
	:param **kwargs : extra arguments, kwargs["mask"] is expected to be exsist
		kwargs["mask"] : ByteTensor, [batch_size , max_len]
					  the mask Tensor , the position that is 1 will be selected
	
	:return predict , truth: predict & truth after processing
	'''
	if kwargs.get("mask") is None:
		return predict , truth
	mask = kwargs["mask"]
	
	predict , truth = squash(predict , truth)
	mask = mask.view(-1,)

	predict = torch.masked_select(predict.permute(1,0) , mask).view(predict.size()[-1] , -1).permute(1,0)
	truth = torch.masked_select(truth , mask)

	return predict , truth

def make_mask(lens , tar_len):
	'''to generate a mask that select [:lens[i]] for i-th element
	embezzle from fastNLP.models.sequence_modeling.seq_mask

	:param lens		: list or LongTensor, [batch_size]
	:param tar_len	: int
	
	:return mask 	: ByteTensor
	'''
	lens = torch.LongTensor(lens)
	mask = [torch.ge(lens, i + 1) for i in range(tar_len)]
	mask = torch.stack(mask, 1)
	return mask

#map string to function. Just for more elegant using
method_dict = {
	"squash" : squash,
	"unpad" : unpad,
	"unpad_mask" : unpad_mask,
	"mask" : mask,
}

loss_function_name = {
	"L1Loss".lower() : torch.nn.L1Loss,
	"BCELoss".lower() : torch.nn.BCELoss,
	"MSELoss".lower() : torch.nn.MSELoss,
	"NLLLoss".lower() : torch.nn.NLLLoss,
	"KLDivLoss".lower() : torch.nn.KLDivLoss,
	"NLLLoss2dLoss".lower() : torch.nn.NLLLoss2d,	#every name should end with "loss"
	"SmoothL1Loss".lower() : torch.nn.SmoothL1Loss,
	"SoftMarginLoss".lower() : torch.nn.SoftMarginLoss,
	"PoissonNLLLoss".lower() : torch.nn.PoissonNLLLoss,
	"MultiMarginLoss".lower() : torch.nn.MultiMarginLoss,
	"CrossEntropyLoss".lower() : torch.nn.CrossEntropyLoss,
	"BCEWithLogitsLoss".lower() : torch.nn.BCEWithLogitsLoss,
	"MarginRankingLoss".lower() : torch.nn.MarginRankingLoss,
	"TripletMarginLoss".lower() : torch.nn.TripletMarginLoss,
	"HingeEmbeddingLoss".lower() : torch.nn.HingeEmbeddingLoss,
	"CosineEmbeddingLoss".lower() : torch.nn.CosineEmbeddingLoss,
	"MultiLabelMarginLoss".lower() : torch.nn.MultiLabelMarginLoss,
	"MultiLabelSoftMarginLoss".lower() : torch.nn.MultiLabelSoftMarginLoss,
}

class Loss(object):
	'''a Loss object is a callable object represents loss functions
	'''

	def __init__(self , loss_name , pre_pro = [squash], **kwargs):
		'''

		:param loss_name: str or None , the name of loss function
		:param pre_pro 	: list of function or str, methods to reform parameters before calculating loss
			the strings will be auto translated to pre-defined functions
		:param **kwargs: kwargs for torch loss function

		pre_pro funcsions should have three arguments: predict, truth, **arg
			predict and truth is the necessary parameters in loss function
			kwargs is the extra parameters passed-in when calling loss function
		pre_pro functions should return two objects, respectively predict and truth that after processed

		'''

		if loss_name is None:
			# this is useful when Trainer.__init__ performs type check
			self._loss = None
		else:
			if not isinstance(loss_name, str):
				raise NotImplementedError
			else:
				self._loss = self._get_loss(loss_name , **kwargs)

		self.pre_pro = [f if callable(f) else method_dict.get(f) for f in pre_pro]

	def add_pre_pro(self , func):
		'''add a pre_pro function

		:param func: a function or str, methods to reform parameters before calculating loss
			the strings will be auto translated to pre-defined functions
		'''
		if not callable(func):
			func = method_dict.get(func)
			if func is None:
				return
		self.pre_pro.append(func)

	@staticmethod
	def _get_loss(loss_name , **kwargs):
		'''Get loss function from torch

		:param loss_name: str, the name of loss function
		:param **kwargs: kwargs for torch loss function
		:return: A callable loss function object
		'''
		loss_name = loss_name.strip().lower()
		loss_name = "".join(loss_name.split("_"))

		if len(loss_name) < 4 or loss_name[-4 : ] != "loss":
			loss_name += "loss"
		return loss_function_name[loss_name](**kwargs)

	def get(self):
		'''This method exists just for make some existing codes run error-freely
		'''
		return self

	def __call__(self , predict , truth , **kwargs):
		'''call a loss function
		predict and truth will be processed by pre_pro methods in order of addition 

		:param predict	: Tensor, model output
		:param truth 	: Tensor, truth from dataset 
		:param **kwargs : extra arguments, pass to pre_pro functions
			for example, if used unpad_mask() in pre_pro, there should be a kwarg named lens
		'''
		for f in self.pre_pro:
			if f is None:
				continue
			predict , truth = f(predict , truth , **kwargs)

		return self._loss(predict , truth)
