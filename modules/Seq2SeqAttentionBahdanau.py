import random

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, padded_idx, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(vocab_size , emb_dim, padding_idx = padded_idx)
        self.encoder   = nn.GRU      (emb_dim, enc_hid_dim, batch_first = True, bidirectional = True)
        self.transform = nn.Linear   (enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src, src_len):
        '''
        Params:
            src    : Torch LongTensor (batch_size, src_seq_len)
            src_len: Torch LongTensor (batch_size)              # descending order
        Return:
            outputs: Torch LongTensor (batch_size, src_seq_len, enc_n_directions * enc_hid_dim)
            hiddens: Torch LongTensor (batch_size, dec_hid_dim)
        '''
        embedded = self.dropout(self.embedding(src))
        embedded = pack_padded_sequence(embedded, src_len.to('cpu'), batch_first = True, enforce_sorted = False)
        outputs, hiddens = self.encoder(embedded)
        outputs, lengths = pad_packed_sequence(outputs, batch_first = True)
        hiddens = torch.tanh(self.transform(torch.cat([hiddens[i] for i in range(hiddens.shape[0])], dim = -1)))
        return outputs, hiddens

class Attention(nn.Module): # Bahdanau Attention
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
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

class Decoder(nn.Module): # With Attention
    def __init__(self, vocab_size, padded_idx, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Decoder, self).__init__()
        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx = padded_idx)
        self.attention = Attention   (enc_hid_dim * 2, dec_hid_dim)
        self.decoder   = nn.GRU      (emb_dim + enc_hid_dim * 2 , dec_hid_dim, batch_first = True)
        self.transform = nn.Linear   (emb_dim + enc_hid_dim * 2 + dec_hid_dim, vocab_size)

    def forward(self, src, dec_hiddens, enc_outputs, enc_masks):
        '''
        Params:
            src        : Torch LongTensor (batch_size)
            dec_hiddens: Torch LongTensor (batch_size, dec_hid_dim)
            enc_outouts: Torch LongTensor (batch_size, src_seq_len, enc_n_directions * enc_hid_dim)
            enc_masks  : Torch LongTensor (batch_size, src_seq_len)
        Return:
            predictions: Torch LongTensor (batch_size, trg_vocab_size)
        '''
        embedded = self.dropout(self.embedding(src.unsqueeze(1)))
        weighted = self.attention(dec_hiddens, enc_outputs, enc_masks).unsqueeze(1)
        concated = torch.cat([embedded, weighted], dim = 2)

        dec_hiddens = dec_hiddens.unsqueeze(0)
        dec_outputs , dec_hiddens = self.decoder(concated, dec_hiddens)
        dec_hiddens = dec_hiddens.squeeze(0)

        predictions = self.transform(
            torch.cat([embedded.squeeze(1), weighted.squeeze(1), dec_outputs.squeeze(1)], dim = -1)
        )
        return predictions, dec_hiddens

class Seq2Seq(nn.Module):
    def __init__( self, 
        src_vocab_size, src_padded_idx, enc_emb_dim, enc_hid_dim, enc_dropout,
        trg_vocab_size, trg_padded_idx, dec_emb_dim, dec_hid_dim, dec_dropout
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
            trg_vocab_size, trg_padded_idx, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout
        )

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        encoder_outputs, encoder_hiddens = self.encoder(src, src_len)

        masked = src != self.src_padded_idx
        inputs = trg [:, 0]
        hidden = encoder_hiddens

        batch_size, trg_seq_len = trg.shape[0], trg.shape[1]
        outputs = torch.zeros((batch_size, trg_seq_len, self.trg_vocab_size), device = src.device)

        for t in range(1, trg_seq_len):
            output , hidden = self.decoder(inputs , hidden , encoder_outputs , masked)
            inputs = trg[:, t] if random.random() < teacher_forcing_ratio else output.argmax(1)
            outputs[:, t, :] = output

        return outputs

    def predict(self):
        pass

def get_module(
    option, src_vocab_size, trg_vocab_size, src_padded_idx, trg_padded_idx
):
    return Seq2Seq(
        src_vocab_size, src_padded_idx, option.enc_emb_dim, option.enc_hid_dim, option.enc_dropout,
        trg_vocab_size, trg_padded_idx, option.dec_emb_dim, option.dec_hid_dim, option.dec_dropout
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
