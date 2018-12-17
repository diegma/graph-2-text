import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class GCNLayer(nn.Module):
    """ Graph convolutional neural network encoder.

    """
    def __init__(self,
                 num_inputs, num_units,
                 num_labels,
                 in_arcs=True,
                 out_arcs=True,
                 batch_first=False,
                 use_gates=True,
                 use_glus=False):
        super(GCNLayer, self).__init__()

        self.in_arcs = in_arcs
        self.out_arcs = out_arcs

        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_labels = num_labels
        self.batch_first = batch_first

        self.glu = nn.GLU(3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_gates = use_gates
        self.use_glus = use_glus
        #https://www.cs.toronto.edu/~yujiali/files/talks/iclr16_ggnn_talk.pdf
        #https://arxiv.org/pdf/1612.08083.pdf

        if in_arcs:
            self.V_in = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal(self.V_in)

            self.b_in = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant(self.b_in, 0)

            if self.use_gates:
                self.V_in_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal(self.V_in_gate)
                self.b_in_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant(self.b_in_gate, 1)

        if out_arcs:
            self.V_out = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal(self.V_out)

            self.b_out = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant(self.b_out, 0)

            if self.use_gates:
                self.V_out_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal(self.V_out_gate)
                self.b_out_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant(self.b_out_gate, 1)

        self.W_self_loop = Parameter(torch.Tensor(self.num_inputs, self.num_units))

        nn.init.xavier_normal(self.W_self_loop)

        if self.use_gates:
            self.W_self_loop_gate = Parameter(torch.Tensor(self.num_inputs, 1))
            nn.init.xavier_normal(self.W_self_loop_gate)

    def forward(self, src, lengths=None, arc_tensor_in=None, arc_tensor_out=None,
                label_tensor_in=None, label_tensor_out=None,
                mask_in=None, mask_out=None,  # batch* t, degree
                mask_loop=None, sent_mask=None):

        if not self.batch_first:
            encoder_outputs = src.permute(1, 0, 2).contiguous()
        else:
            encoder_outputs = src.contiguous()
        batch_size = encoder_outputs.size()[0]
        seq_len = encoder_outputs.size()[1]
        max_degree = 1
        input_ = encoder_outputs.view((batch_size * seq_len, self.num_inputs))  # [b* t, h]

        if self.in_arcs:
            input_in = torch.mm(input_, self.V_in)  # [b* t, h] * [h,h] = [b*t, h]
            first_in = input_in.index_select(0, arc_tensor_in[0] * seq_len + arc_tensor_in[1])  # [b* t* degr, h]
            second_in = self.b_in.index_select(0, label_tensor_in[0])  # [b* t* degr, h]
            in_ = first_in + second_in
            degr = int(first_in.size()[0] / batch_size / seq_len)

            in_ = in_.view((batch_size, seq_len, degr, self.num_units))

            if self.use_glus:
                # gate the information of each neighbour, self nodes are in here too.
                in_ = torch.cat((in_, in_), 3)
                in_ = self.glu(in_)

            if self.use_gates:
                # compute gate weights
                input_in_gate = torch.mm(input_, self.V_in_gate)  # [b* t, h] * [h,h] = [b*t, h]
                first_in_gate = input_in_gate.index_select(0, arc_tensor_in[0] * seq_len + arc_tensor_in[1])  # [b* t* mxdeg, h]
                second_in_gate = self.b_in_gate.index_select(0, label_tensor_in[0])
                in_gate = (first_in_gate + second_in_gate).view((batch_size, seq_len, degr))

            max_degree += degr

        if self.out_arcs:
            input_out = torch.mm(input_, self.V_out)  # [b* t, h] * [h,h] = [b* t, h]
            first_out = input_out.index_select(0, arc_tensor_out[0] * seq_len + arc_tensor_out[1])  # [b* t* mxdeg, h]
            second_out = self.b_out.index_select(0, label_tensor_out[0])

            degr = int(first_out.size()[0] / batch_size / seq_len)
            max_degree += degr

            out_ = (first_out + second_out).view((batch_size, seq_len, degr, self.num_units))


            if self.use_glus:
                # gate the information of each neighbour, self nodes are in here too.
                out_ = torch.cat((out_, out_), 3)
                out_ = self.glu(out_)

            if self.use_gates:
                # compute gate weights
                input_out_gate = torch.mm(input_, self.V_out_gate)  # [b* t, h] * [h,h] = [b* t, h]
                first_out_gate = input_out_gate.index_select(0, arc_tensor_out[0] * seq_len + arc_tensor_out[1])  # [b* t* mxdeg, h]
                second_out_gate = self.b_out_gate.index_select(0, label_tensor_out[0])
                out_gate = (first_out_gate + second_out_gate).view((batch_size, seq_len, degr))


        same_input = torch.mm(encoder_outputs.view(-1, encoder_outputs.size(2)), self.W_self_loop). \
            view(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        same_input = same_input.view(encoder_outputs.size(0), encoder_outputs.size(1), 1, self.W_self_loop.size(1))
        if self.use_gates:
            same_input_gate = torch.mm(encoder_outputs.view(-1, encoder_outputs.size(2)), self.W_self_loop_gate) \
                .view(encoder_outputs.size(0), encoder_outputs.size(1), -1)

        if self.in_arcs and self.out_arcs:
            potentials = torch.cat((in_, out_, same_input), dim=2)  # [b, t,  mxdeg, h]
            if self.use_gates:
                potentials_gate = torch.cat((in_gate, out_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_in, mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
        elif self.out_arcs:
            potentials = torch.cat((out_, same_input), dim=2)  # [b, t,  2*mxdeg+1, h]
            if self.use_gates:
                potentials_gate = torch.cat((out_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
        elif self.in_arcs:
            potentials = torch.cat((in_, same_input), dim=2)  # [b, t,  2*mxdeg+1, h]
            if self.use_gates:
                potentials_gate = torch.cat((in_gate, same_input_gate), dim=2)  # [b, t,  mxdeg, h]
            mask_soft = torch.cat((mask_in, mask_loop), dim=1)  # [b* t, mxdeg]
        else:
            potentials = same_input  # [b, t,  2*mxdeg+1, h]
            if self.use_gates:
                potentials_gate = same_input_gate # [b, t,  mxdeg, h]
            mask_soft = mask_loop  # [b* t, mxdeg]

        potentials_resh = potentials.view((batch_size * seq_len,
                                           max_degree, self.num_units,))  # [h, b * t, mxdeg]


        if self.use_gates:
            potentials_r = potentials_gate.view((batch_size * seq_len,
                                                 max_degree))  # [b * t, mxdeg]

            probs_det_ = (self.sigmoid(potentials_r) * mask_soft).unsqueeze(2)  # [b * t, mxdeg]
            potentials_masked = potentials_resh * probs_det_  # [b * t, mxdeg,h]
        else:
            # NO Gates
            potentials_masked = potentials_resh * mask_soft.unsqueeze(2)



        potentials_masked_ = potentials_masked.sum(dim=1)  # [b * t, h]
        potentials_masked_ = self.relu(potentials_masked_)  # [b * t, h]

        result_ = potentials_masked_.view((batch_size, seq_len, self.num_units))  # [ b, t, h]

        result_ = result_ * sent_mask.permute(1, 0).contiguous().unsqueeze(2)  # [b, t, h]

        memory_bank = result_.permute(1, 0, 2).contiguous()  # [t, b, h]

        return memory_bank
