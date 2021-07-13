import torch
import torch.nn as nn
import torch.nn.functional as F
import High_deg_ConvGRU_model as gru_model


class multilayers_Painter_model(nn.Module):
    def __init__(self, in_length, out_length, mode='train'):
        super(multilayers_Painter_model, self).__init__()
        self.mode = mode
        self.decoder_GRU_out_channel = 32
        self.static_encoder_channel = 128
        self.out_len = out_length
        self.encoder_GRU = Encoder(hidden_channel1=64, hidden_channel2=128,
                                   hidden_channel3=256, input_channel=1)
        self.decoder_GRU = Decoder(hidden_channel1=64, hidden_channel2=128,
                                   hidden_channel3=256, out_channel=self.decoder_GRU_out_channel, out_len=out_length)
        # standard_GRU
        self.decoder = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.out_len = out_length
        print('0.25deg_U_Wind_ConvGRU_Encoder_Decoder')

    def forward(self, input):
        input_decode = self.encoder_GRU.forward(input)
        decod_out = self.decoder_GRU.forward(input_decode)

        preds = []
        for j in range(self.out_len):
            # standard_gru
            pred_t = self.decoder(decod_out[:, j, :, :, :])
            preds.append(pred_t)
        outs = torch.stack(preds, 1)
        return outs


class Encoder(nn.Module):
    def __init__(self,hidden_channel1 ,
                      hidden_channel2 ,
                      hidden_channel3 ,
                      input_channel):
        super(Encoder, self).__init__()
        self.hidden1 = hidden_channel1
        self.hidden2 = hidden_channel2
        self.hidden3 = hidden_channel3
        self.con_layer1 = nn.Conv2d(in_channels=input_channel, out_channels=self.hidden1,
                                    kernel_size=4, stride=2, padding=1)
        self.gru_layer1 = gru_model.W_ConvGRU(hidden=self.hidden1, conv_kernel_size=5, input_channle=self.hidden1,)

        self.con_layer2 = nn.Conv2d(in_channels=self.hidden1, out_channels=self.hidden2,
                                    kernel_size=4, stride=2, padding=1)
        self.gru_layer2 = gru_model.W_ConvGRU(hidden=self.hidden2, conv_kernel_size=5, input_channle=self.hidden2,)

        self.con_layer3 = nn.Conv2d(in_channels=self.hidden2, out_channels=self.hidden3,
                                    kernel_size=4, stride=2, padding=1)
        self.gru_layer3 = gru_model.W_ConvGRU(hidden=self.hidden3, conv_kernel_size=5, input_channle=self.hidden3,)

    def forward(self, input):
        H = input.size()[-2]
        W = input.size()[-1]
        decode_gru_input = []
        # grulayer_1 & convlayer_1
        hidden_size1 = (input.size()[0], self.hidden1, int(H / 2), int(W / 2))
        state1 = torch.zeros(hidden_size1).cuda()
        n_step = input.size()[1]
        layer2_input = []
        for i in range(n_step):
            x_t = input[:, i, :, :, :]
            x_t = self.con_layer1(x_t)
            gru_state = self.gru_layer1.forward(x_t, state1)
            state1 = gru_state
            layer2_input.append(state1)
        decode_gru_input.append(state1)
        # grulayer_2&convlayer_2
        hidden_size2 = (input.size()[0], self.hidden2, int(H / 4), int(W / 4))
        state2 = torch.zeros(hidden_size2).cuda()
        layer3_input = []
        for i in range(n_step):
            x_t = layer2_input[i]
            x_t = self.con_layer2(x_t)
            gru_state = self.gru_layer2.forward(x_t, state2)
            state2 = gru_state
            layer3_input.append(state2)
        decode_gru_input.append(state2)
        # grulayer_3&convlayer_3
        hidden_size3 = (input.size()[0], self.hidden3, int(H / 8), int(W / 8))
        state3 = torch.zeros(hidden_size3).cuda()
        for i in range(n_step):
            x_t = layer3_input[i]
            x_t = self.con_layer3(x_t)
            gru_state = self.gru_layer3.forward(x_t, state3)
            state3 = gru_state
        decode_gru_input.append(state3)
        return decode_gru_input


class Decoder(nn.Module):
    def __init__(self,hidden_channel1,
                      hidden_channel2,
                      hidden_channel3,
                      out_channel,
                      out_len):
        super(Decoder, self).__init__()
        self.out_len = out_len

        self.gru_layer3 = gru_model.W_ConvGRU(hidden=hidden_channel3, conv_kernel_size=5, input_channle=hidden_channel3,
                                             )
        self.conv_layer3 = nn.ConvTranspose2d(in_channels=hidden_channel3, out_channels=hidden_channel2,
                                              kernel_size=4, stride=2, padding=1,output_padding=(1,0))
        self.gru_layer2 = gru_model.W_ConvGRU(hidden=hidden_channel2, conv_kernel_size=5, input_channle=hidden_channel2,
                                              )
        self.conv_layer2 = nn.ConvTranspose2d(in_channels=hidden_channel2, out_channels=hidden_channel1,
                                              kernel_size=4, stride=2, padding=1)
        self.gru_layer1 = gru_model.W_ConvGRU(hidden=hidden_channel1, conv_kernel_size=5, input_channle=hidden_channel1,
                                              )
        self.conv_layer1 = nn.ConvTranspose2d(in_channels=hidden_channel1, out_channels=out_channel,
                                              kernel_size=4, stride=2, padding=1, output_padding=1)

    def forward(self, states):
        state3 = states[-1]
        layer2_input = []
        for i in range(self.out_len):
            paper_t = torch.zeros(state3.size()[0], state3.size()[1],
                                 state3.size()[-2], state3.size()[-1])
            paper_t = paper_t.cuda()
            h3 = self.gru_layer3.forward(x_t=paper_t, hidden=state3)
            state3 = h3
            hidden3_t = self.conv_layer3(h3)
            layer2_input.append(hidden3_t)
        state2 = states[-2]
        layer1_input = []
        for i in range(self.out_len):
            h2 = self.gru_layer2.forward(x_t=layer2_input[i], hidden=state2)
            state2 = h2
            hidden2_t = self.conv_layer2(h2)
            layer1_input.append(hidden2_t)
        state1 = states[0]
        out = []
        for i in range(self.out_len):
            h1 = self.gru_layer1.forward(x_t=layer1_input[i], hidden=state1)
            state1 = h1
            hidden1_t = self.conv_layer1(h1)
            out.append(hidden1_t)
        outs = torch.stack(out, 1)
        return outs

