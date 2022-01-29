# Seq2Seq with Attention (Bahdanau and Luong)

import random

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import  pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, padded_idx, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(vocab_size , emb_dim, padding_idx = padded_idx)
        self.encoder   = nn.GRU      (emb_dim, enc_hid_dim, batch_first = True)
        self.transform = nn.Linear   (enc_hid_dim, dec_hid_dim)

    def forward(self, src, src_len):
        '''
        Params:
            src    : Torch LongTensor (batch_size, src_seq_len)
            src_len: Torch LongTensor (batch_size) # descending order, max_value equals src_seq_len
        Return:
            outputs: Torch LongTensor (batch_size, src_seq_len, enc_hid_dim)
            hiddens: Torch LongTensor (batch_size, dec_hid_dim)
        '''
        embedded = self.dropout(self.embedding(src))
        embedded = pack_padded_sequence(embedded, src_len.to('cpu'), batch_first = True, enforce_sorted = False)
        outputs, hiddens = self.encoder(embedded)
        outputs, lengths = pad_packed_sequence(outputs, batch_first = True)
        hiddens = torch.tanh(self.transform(torch.cat([hiddens[i] for i in range(hiddens.shape[0])], dim = -1)))
        return outputs, hiddens

class AttentionBahdanau(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(AttentionBahdanau, self).__init__()
        self.a = nn.Linear(enc_hid_dim + dec_hid_dim , dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim , 1 , bias = False)

    def forward(self, hidden_states, encoder_outputs, encoder_masks):
        '''
        Params:
            hidden_states  : Torch LongTensor (batch_size, dec_hid_dim)
            encoder_outputs: Torch LongTensor (batch_size, src_seq_len, enc_hid_dim)
            encoder_masks  : Torch LongTensor (batch_size, src_seq_len)
        Return:
            energy         : Torch LongTensor (batch_size, src_seq_len)
            weighted       : Torch LongTensor (batch_size, enc_hid_dim)
        '''
        hidden_states = hidden_states.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)

        energy = self.a(torch.cat((hidden_states, encoder_outputs), dim = -1)).tanh()
        energy = self.v(energy).squeeze(-1).masked_fill(encoder_masks == 0, -1e10).softmax(dim = -1)

        weighted = torch.bmm(energy.unsqueeze(1), encoder_outputs).squeeze(1)

        return weighted#, energy

class AttentionLuong(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, align_method = 'concat'):
        super(AttentionLuong, self).__init__()

        self.align_method = align_method
        assert align_method in [ 'dot' , 'general' , 'concat' ]

        if  align_method == 'dot':
            assert enc_hid_dim == dec_hid_dim

        if  align_method == 'general':
            self.a = nn.Linear(dec_hid_dim , enc_hid_dim)

        if  align_method == 'concat': # the same as Bahdanau Attention
            self.a = nn.Linear(enc_hid_dim + dec_hid_dim , dec_hid_dim)
            self.v = nn.Linear(dec_hid_dim , 1 , bias = False)

    def forward(self, hidden_states, encoder_outputs, encoder_masks):
        '''
        Params:
            hidden_states  : Torch LongTensor (batch_size, dec_hid_dim)
            encoder_outputs: Torch LongTensor (batch_size, src_seq_len, enc_hid_dim)
            encoder_masks  : Torch LongTensor (batch_size, src_seq_len)
        Return:
            energy         : Torch LongTensor (batch_size, src_seq_len)
            weighted       : Torch LongTensor (batch_size, enc_hid_dim)
        '''

        if  self.align_method == 'dot':
            energy = torch.bmm(
                hidden_states.unsqueeze(1),
                encoder_outputs.permute(0, 2, 1)
            )
            energy = energy.squeeze(1).masked_fill(encoder_masks == 0, -1e10).softmax(dim = -1)

            weighted = torch.bmm(energy.unsqueeze(1), encoder_outputs).squeeze(1)

        if  self.align_method == 'general':
            energy = torch.bmm(
                self.a(hidden_states.unsqueeze(1)),
                encoder_outputs.permute(0, 2, 1)
            )
            energy = energy.squeeze(1).masked_fill(encoder_masks == 0, -1e10).softmax(dim = -1)

            weighted = torch.bmm(energy.unsqueeze(1), encoder_outputs).squeeze(1)

        if  self.align_method == 'concat':

            hidden_states = hidden_states.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)

            energy = self.a(torch.cat((hidden_states, encoder_outputs), dim = -1)).tanh()
            energy = self.v(energy).squeeze(-1).masked_fill(encoder_masks == 0, -1e10).softmax(dim = -1)

            weighted = torch.bmm(energy.unsqueeze(1), encoder_outputs).squeeze(1)

        return weighted#, energy

class Decoder(nn.Module): # With Attention
    def __init__(self, vocab_size, padded_idx, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention_type = 'luong', align_method = 'concat'):
        super(Decoder, self).__init__()
        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx = padded_idx)

        self.attention_type = attention_type
        assert attention_type in [ 'bahdanau' , 'luong' ]

        if  attention_type == 'bahdanau':
            self.attention = AttentionBahdanau(enc_hid_dim , dec_hid_dim)
            self.decoder   = nn.GRU           (emb_dim + enc_hid_dim , dec_hid_dim, batch_first = True)
            self.transform = nn.Linear        (emb_dim + enc_hid_dim + dec_hid_dim, vocab_size)

        if  attention_type == 'luong':
            self.decoder   = nn.GRU        (emb_dim , dec_hid_dim, batch_first = True)
            self.attention = AttentionLuong(enc_hid_dim , dec_hid_dim, align_method)
            self.transform = nn.Linear     (enc_hid_dim + dec_hid_dim, vocab_size)

    def forward(self, src, dec_hiddens, enc_outputs, enc_masks):
        '''
        Params:
            src        : Torch LongTensor (batch_size)
            dec_hiddens: Torch LongTensor (batch_size, dec_hid_dim)
            enc_outouts: Torch LongTensor (batch_size, src_seq_len, enc_hid_dim)
            enc_masks  : Torch LongTensor (batch_size, src_seq_len)
        Return:
            predictions: Torch LongTensor (batch_size, trg_vocab_size)
        '''

        if  self.attention_type == 'bahdanau':

            embedded = self.dropout(self.embedding(src.unsqueeze(1)))
            weighted = self.attention(dec_hiddens, enc_outputs, enc_masks).unsqueeze(1)
            concated = torch.cat([embedded, weighted], dim = 2)

            dec_hiddens = dec_hiddens.unsqueeze(0)
            dec_outputs , dec_hiddens = self.decoder(concated, dec_hiddens)
            dec_hiddens = dec_hiddens.squeeze(0)

            predictions = self.transform(
                torch.cat([embedded.squeeze(1), weighted.squeeze(1), dec_outputs.squeeze(1)], dim = -1)
            )

        if  self.attention_type == 'luong':

            embedded = self.dropout(self.embedding(src.unsqueeze(1)))

            dec_hiddens = dec_hiddens.unsqueeze(0)
            dec_outputs , dec_hiddens = self.decoder(embedded, dec_hiddens)
            dec_hiddens = dec_hiddens.squeeze(0)

            weighted = self.attention(dec_hiddens, enc_outputs, enc_masks)

            predictions = self.transform(
                torch.cat([dec_outputs.squeeze(1), weighted.squeeze(1)], dim = -1)
            )

        return predictions, dec_hiddens

class BeamSearchNode():
    def __init__(self, idx, pro, hid, prev_node):
        self.idx = idx
        self.pro = pro
        self.hid = hid
        self.prev_node = prev_node

        if  prev_node is None:
            self.sent_pro = pro
            self.sent_len = 1
        else:
            self.sent_pro = prev_node.sent_pro + pro
            self.sent_len = prev_node.sent_len + 1
        
        self.sent_matrix = self.sent_pro / self.sent_len

class Seq2Seq(nn.Module):
    def __init__( self, 
        src_vocab_size, src_padded_idx, enc_emb_dim, enc_hid_dim, enc_dropout,
        trg_vocab_size, trg_padded_idx, dec_emb_dim, dec_hid_dim, dec_dropout,
        attention_type, align_method
    ):
        super(Seq2Seq, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_padded_idx = src_padded_idx
        self.trg_padded_idx = trg_padded_idx

        self.encoder = Encoder(
            src_vocab_size, src_padded_idx, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout
        )
        self.decoder = Decoder(
            trg_vocab_size, trg_padded_idx, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attention_type, align_method
        )

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        '''
        Params:
            src    : Torch LongTensor (batch_size, src_seq_len)
            src_len: Torch LongTensor (batch_size)
            trg    : Torch LongTensor (batch_size, trg_seq_len)
            teacher_forcing_ratio: Float between 0 and 1, probability of whether to use guide
        '''
        encoder_outputs, encoder_hiddens = self.encoder(src, src_len)

        masked = src != self.src_padded_idx
        inputs = trg [:, 0]
        hidden = encoder_hiddens

        batch_size, trg_seq_len = trg.shape[0], trg.shape[1]
        outputs = torch.zeros((batch_size, trg_seq_len, self.trg_vocab_size), device = src.device)

        for t in range(1, trg_seq_len):
            output , hidden = self.decoder(inputs , hidden , encoder_outputs , masked)
            inputs = trg[:, t] if random.random() < teacher_forcing_ratio else output.argmax(-1)
            outputs[:, t, :] = output

        return outputs

    def predict(self, src, src_len, max_len, trg_bos_idx, trg_eos_idx, decoding_strategy, beam_width = 1):
        '''
        Params:
            src    : Torch LongTensor (batch_size = 1, src_seq_len)
            src_len: Torch LongTensor (batch_size = 1)
            max_len: Int , maximum length of generated sentence
            trg_bos_idx: Int , index of bos tag in trg vocab
            trg_eos_idx: Int , index of eos tag in trg vocab
            decoding_strategy: String, chosen between GreedySearch and BeamSearch
        Return:
            outputs: List<List<Int>>
        '''
        assert src.shape[0] == 1 ,  'predict method only support `batch_size == 1`'
        assert decoding_strategy == 'greedysearch' or decoding_strategy == 'beamsearch'

        encoder_outputs, encoder_hiddens = self.encoder(src, src_len)

        masked = src != self.src_padded_idx
        inputs = torch.ones(src.shape[0], dtype = src.dtype, device = src.device) * trg_bos_idx
        hidden = encoder_hiddens

        if  decoding_strategy == 'greedysearch':
            outputs = []

            for _ in range(max_len):
                output , hidden = self.decoder(inputs, hidden, encoder_outputs, masked)
                inputs = output.argmax(-1)
                if  trg_eos_idx != inputs.item():
                    outputs.append(inputs.item())
                else:
                    break

            return [outputs]

        if  decoding_strategy == 'beamsearch':
            bst_nodes = []
            end_nodes = []
            decoding_step = 0

            output , hidden = self.decoder(inputs, hidden, encoder_outputs, masked)
            output = output.squeeze(0).log_softmax(0)
            topk_p , topk_i = torch.topk(output, beam_width)
            for i, p in zip(topk_i, topk_p):
                bst_nodes.append(BeamSearchNode(i, p, hidden, None))

            while len(end_nodes) < beam_width:
                decoding_step += 1
                tmp_nodes = []
                for node in bst_nodes:
                    output , hidden = self.decoder(node.idx.reshape(-1), node.hid, encoder_outputs, masked)
                    output = output.squeeze(0).log_softmax(0)
                    topk_p , topk_i = torch.topk(output, beam_width - len(end_nodes))
                    for i, p in zip(topk_i, topk_p):
                        tmp_nodes.append(BeamSearchNode(i, p, hidden, node))

                tmp_nodes = sorted(tmp_nodes, key = lambda x: x.sent_matrix, reverse = True)[:beam_width - len(end_nodes)]

                if  decoding_step == max_len:
                    end_nodes.extend(tmp_nodes)
                else:
                    bst_nodes.clear()
                    for node in tmp_nodes:
                        if  node.idx == trg_eos_idx:
                            end_nodes.append(node)
                        else:
                            bst_nodes.append(node)

            outputs = []
            for node in end_nodes:
                tmp_outputs = []
                while node is not None:
                    tmp_outputs.append(node.idx.item())
                    node = node.prev_node
                outputs.append(tmp_outputs[::-1])

            return outputs

def get_module(
    option, src_vocab_size, trg_vocab_size, src_padded_idx, trg_padded_idx
):
    return Seq2Seq(
        src_vocab_size, src_padded_idx, option.enc_emb_dim, option.enc_hid_dim, option.enc_dropout,
        trg_vocab_size, trg_padded_idx, option.dec_emb_dim, option.dec_hid_dim, option.dec_dropout,
        option.attention_type, option.align_method
    )

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    batch_size = 16

    src_vocab_size = 13000
    trg_vocab_size = 17000
    src_padded_idx = 0
    trg_padded_idx = 0

    src_seq_len = 18
    trg_seq_len = 20

    module = get_module (
        option, src_vocab_size, trg_vocab_size, src_padded_idx, trg_padded_idx
    )

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    trg = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_len))

    # src_len must be in descending order, and the max value should be equal to src_seq_len
    src_len = torch.flip(torch.arange(3, 19), dims = [0])

    outputs = module(src, src_len, trg)
    print(outputs.shape) # (batch_size, trg_seq_len, trg_vocab_size)

    # predict
    g_result = module.predict(src[0].unsqueeze(0), src_len[0].unsqueeze(0), trg_seq_len, 0, 1, 'greedysearch' )
    b_result = module.predict(src[0].unsqueeze(0), src_len[0].unsqueeze(0), trg_seq_len, 0, 1, 'beamsearch', 2)
    print(g_result) # Nested List: List<List<Int>>
    print(b_result) # Nested List: List<List<Int>>
