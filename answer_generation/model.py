import torch
import torch.nn as nn
from torch.nn import AvgPool2d
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, embedding_size, LSTM_units, LSTM_layers, feat_size,
                 batch_size, global_avg_pool_size, 
                 dropout = 0.3, mfb_output_dim = 5000):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.mfb_output_dim = mfb_output_dim
        self.feat_size = feat_size
        self.mfb_out = 1000
        self.mfb_factor = 5
        self.channel_size = global_avg_pool_size
        self.num_ques_glimpse = 2
        self.num_img_glimpse = 2
        
        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=LSTM_units, 
                            num_layers=LSTM_layers, batch_first=False)
        self.Dropout = nn.Dropout(p=dropout, )
        self.Softmax = nn.Softmax()
        
        self.Linear1_q_proj = nn.Linear(LSTM_units*self.num_ques_glimpse, self.mfb_output_dim)
        self.Conv_i_proj = nn.Conv2d(self.feat_size, self.mfb_output_dim, 1)

        self.Dropout_L = nn.Dropout(p=0.2)
        self.Dropout_M = nn.Dropout(p=0.2)
        self.Conv1_Qatt = nn.Conv2d(LSTM_units, 512, 1)
        self.Conv2_Qatt = nn.Conv2d(512, self.num_ques_glimpse, 1)
        self.Conv1_Iatt = nn.Conv2d(1000, 512, 1)
        self.Conv2_Iatt = nn.Conv2d(512, self.num_img_glimpse, 1)
        
        self.qatt_maps = None
        self.iatt_maps = None
        
    def forward(self, ques_embed, img_feat):
        
        # preparing image features
        img_feat_resh = img_feat.permute(0, 3, 1, 2).contiguous()         # N x w x w x C -> N x C x w x w
        img_feat_resh = img_feat_resh.reshape(img_feat_resh.shape[0], img_feat_resh.shape[1], 
                                              self.channel_size*self.channel_size)      # N x C x w*w
        
        # ques_embed                                         N x T x embedding_size
        ques_embed_resh = ques_embed.permute(1, 0, 2).contiguous()        #T x N x embedding_size
        lstm_out, (hn, cn) = self.LSTM(ques_embed_resh)
        lstm1_droped = self.Dropout_L(lstm_out)
        lstm1_resh = lstm1_droped.permute(1, 2, 0).contiguous()                   # N x 1024 x T
        lstm1_resh2 = torch.unsqueeze(lstm1_resh, 3)                # N x 1024 x T x 1
        
        '''
        Question Attention
        '''        
        qatt_conv1 = self.Conv1_Qatt(lstm1_resh2)                   # N x 512 x T x 1
        qatt_relu = F.relu(qatt_conv1)
        qatt_conv2 = self.Conv2_Qatt(qatt_relu)                     # N x 2 x T x 1
#        print(qatt_conv2.shape)
        qatt_conv2 = qatt_conv2.reshape(qatt_conv2.shape[0]*self.num_ques_glimpse,-1)
        qatt_softmax = self.Softmax(qatt_conv2)
        qatt_softmax = qatt_softmax.view(qatt_conv1.shape[0], self.num_ques_glimpse, -1, 1)
        self.qatt_maps = qatt_softmax
        qatt_feature_list = []
        for i in range(self.num_ques_glimpse):
            t_qatt_mask = qatt_softmax.narrow(1, i, 1)              # N x 1 x T x 1
#            print(t_qatt_mask.shape, lstm1_resh2.shape)
            t_qatt_mask = t_qatt_mask * lstm1_resh2                 # N x 1024 x T x 1
            t_qatt_mask = torch.sum(t_qatt_mask, 2, keepdim=True)   # N x 1024 x 1 x 1
            qatt_feature_list.append(t_qatt_mask)
        qatt_feature_concat = torch.cat(qatt_feature_list, 1)       # N x 2048 x 1 x 1
        
        '''
        Image Attention with MFB
        '''
        q_feat_resh = torch.squeeze(qatt_feature_concat)                                # N x 2048
        i_feat_resh = torch.unsqueeze(img_feat_resh, 3)                                   # N x 2048 x w*w x 1
        iatt_q_proj = self.Linear1_q_proj(q_feat_resh)                                  # N x 5000
        iatt_q_resh = iatt_q_proj.view(iatt_q_proj.shape[0], self.mfb_output_dim, 1, 1)      # N x 5000 x 1 x 1
        iatt_i_conv = self.Conv_i_proj(i_feat_resh)                                     # N x 5000 x w*w x 1
        iatt_iq_eltwise = iatt_q_resh * iatt_i_conv
        iatt_iq_droped = self.Dropout_M(iatt_iq_eltwise)                                # N x 5000 x w*w x 1
        iatt_iq_permute1 = iatt_iq_droped.permute(0,2,1,3).contiguous()                              # N x w*w x 5000 x 1
        iatt_iq_resh = iatt_iq_permute1.view(iatt_iq_permute1.shape[0], self.channel_size*self.channel_size, 
                                             self.mfb_out, self.mfb_factor)
        iatt_iq_sumpool = torch.sum(iatt_iq_resh, 3, keepdim=True)                      # N x w*w x 1000 x 1 
        iatt_iq_permute2 = iatt_iq_sumpool.permute(0,2,1,3).contiguous()                            # N x 1000 x w*w x 1
        iatt_iq_sqrt = torch.sqrt(F.relu(iatt_iq_permute2)) - torch.sqrt(F.relu(-iatt_iq_permute2))
        iatt_iq_sqrt = torch.squeeze(iatt_iq_sqrt)
        iatt_iq_sqrt = iatt_iq_sqrt.reshape(iatt_iq_sqrt.shape[0], -1)                           # N x 1000*w*w
        iatt_iq_l2 = F.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(iatt_iq_l2.shape[0], self.mfb_out, self.channel_size*self.channel_size, 1)  # N x 1000 x w*w x 1
        
        ## 2 conv layers 1000 -> 512 -> 2
        iatt_conv1 = self.Conv1_Iatt(iatt_iq_l2)                    # N x 512 x w*w x 1
        iatt_relu = F.relu(iatt_conv1)
        iatt_conv2 = self.Conv2_Iatt(iatt_relu)                     # N x 2 x w*w x 1
        iatt_conv2 = iatt_conv2.view(iatt_conv2.shape[0]*self.num_img_glimpse, -1)
        iatt_softmax = self.Softmax(iatt_conv2)
        iatt_softmax = iatt_softmax.view(iatt_conv1.shape[0], self.num_img_glimpse, -1, 1)
        self.iatt_maps = iatt_softmax.view(iatt_conv1.shape[0], self.num_img_glimpse, self.channel_size, self.channel_size)
        iatt_feature_list = []
        for i in range(self.num_img_glimpse):
            t_iatt_mask = iatt_softmax.narrow(1, i, 1)              # N x 1 x w*w x 1
            t_iatt_mask = t_iatt_mask * i_feat_resh                 # N x 2048 x w*w x 1
            iatt_feature_list.append(t_iatt_mask)
        iatt_feature_concat = torch.mean(iatt_feature_list, 3)       # N x 2048 x w*w x 1
        iatt_feature_resh = iatt_feature_concat.view(iatt_iq_permute1.shape[0], self.channel_size, 
                                                        self.channel_size, iatt_iq_permute1.shape[1])           # N x w x w x 2048
        
        return iatt_feature_resh

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind