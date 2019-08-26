'''
@author: https://github.com/ahmedbesbes/character-based-cnn
这里借鉴了上述链接中char-cnn model的代码，改动主要为将其改动为符合fastnlp的pipline
'''
import torch
import torch.nn as nn
from fastNLP.core.const import Const as C

class CharacterLevelCNN(nn.Module):
    def __init__(self, args,embedding):
        super(CharacterLevelCNN, self).__init__()

        self.config=args.char_cnn_config
        self.embedding=embedding

        conv_layers = []
        for i, conv_layer_parameter in enumerate(self.config['model_parameters'][args.model_size]['conv']):
            if i == 0:
                #in_channels = args.number_of_characters + len(args.extra_characters)
                in_channels = args.embedding_dim
                out_channels = conv_layer_parameter[0]
            else:
                in_channels, out_channels = conv_layer_parameter[0], conv_layer_parameter[0]

            if conv_layer_parameter[2] != -1:
                conv_layer = nn.Sequential(nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=conv_layer_parameter[1], padding=0),
                                           nn.ReLU(),
                                           nn.MaxPool1d(conv_layer_parameter[2]))
            else:
                conv_layer = nn.Sequential(nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=conv_layer_parameter[1], padding=0),
                                           nn.ReLU())
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        input_shape = (args.batch_size, args.max_length,
                       args.number_of_characters + len(args.extra_characters))
        dimension = self._get_conv_output(input_shape)

        print('dimension :', dimension)

        fc_layer_parameter = self.config['model_parameters'][args.model_size]['fc'][0]
        fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension, fc_layer_parameter), nn.Dropout(0.5)),
            nn.Sequential(nn.Linear(fc_layer_parameter,
                                    fc_layer_parameter), nn.Dropout(0.5)),
            nn.Linear(fc_layer_parameter, args.num_classes),
        ])

        self.fc_layers = fc_layers

        if args.model_size == 'small':
            self._create_weights(mean=0.0, std=0.05)
        elif args.model_size == 'large':
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        input = torch.rand(shape)
        output = input.transpose(1, 2)
        # forward pass through conv layers
        for i in range(len(self.conv_layers)):
            output = self.conv_layers[i](output)

        output = output.view(output.size(0), -1)
        n_size = output.size(1)
        return n_size

    def forward(self, chars):
        input=self.embedding(chars)
        output = input.transpose(1, 2)
        # forward pass through conv layers
        for i in range(len(self.conv_layers)):
            output = self.conv_layers[i](output)

        output = output.view(output.size(0), -1)

        # forward pass through fc layers
        for i in range(len(self.fc_layers)):
            output = self.fc_layers[i](output)

        return {C.OUTPUT: output}