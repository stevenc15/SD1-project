
#IMPORTS




import torch.nn as nn #neural network construction


from model import Encoder_GRU, Encoder_CNN_1D, Encoder_CNN_2D

class teacher(nn.Module):
    def __init__(self, input_shape_1D,input_shape_2D, w):
        super(teacher, self).__init__()
        self.w = w

        # 1D Models

        self.model_1=Encoder_GRU(input_shape_1D,input_shape_2D,0.40)
        self.cnn_1D = Encoder_CNN_1D(input_shape_1D,input_shape_2D,0.40)
        self.cnn_2D = Encoder_CNN_2D(input_shape_1D,input_shape_2D,0.40)

        self.BN_1D= nn.BatchNorm1d(input_shape_1D, affine=False)
        self.BN_2D= nn.BatchNorm2d(input_shape_2D, affine=False)


    def forward(self, inputs_1D_N, inputs_2D_N):

        # input_1D_N_1=inputs_1D_N.transpose(1,2)
        # input_1D_N_1=self.BN_1D(input_1D_N_1)
        # input_1D_N=input_1D_N_1.transpose(1,2)

        # inputs_2D_N=inputs_2D_N.transpose(1,3)
        # inputs_2D_N=self.BN_2D(inputs_2D_N)
        # inputs_2D_N=inputs_2D_N.transpose(1,3)

        output_GRU = self.model_1(inputs_1D_N)
        output_C2 = self.cnn_1D(inputs_1D_N,inputs_1D_N)
        output_C1 = self.cnn_2D(inputs_1D_N,inputs_2D_N)


        output = (output_GRU +output_C2+output_C1)/3

        return output_GRU, output_C1, output_C2, output