import pickle
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
from collections import Counter

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam

class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        # raise NotImplementedError()
        batch_size, sequence_length, embedding_dim = X.shape
        pos_enc = torch.zeros((sequence_length, embedding_dim))
        div_term = torch.pow(10000, torch.arange(0, embedding_dim, 2, dtype=torch.float)/ self.embedding_dim)
        pos = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        
        pos_enc[:, 0::2] = torch.sin(pos/div_term)
        pos_enc[:, 1::2] = torch.cos(pos/div_term)
        return X + pos_enc


class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """
        # raise NotImplementedError()
        
        Q = self.linear_Q(query_X)
        K = self.linear_K(key_X)
        V = self.linear_V(value_X)

        scores = torch.matmul(Q, torch.transpose(K, 1, 2))

        if torch.is_tensor(mask):
            
            masked_scores = scores.masked_fill((~(mask.bool())), -1e32)

            attention_weights = self.softmax(masked_scores/np.sqrt(self.out_dim))

            output = torch.bmm(attention_weights, V)
            return output, attention_weights
        
        else:
            attention_weights = self.softmax(scores/np.sqrt(self.out_dim))

            output = torch.bmm(attention_weights, V)
            return output, attention_weights


class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return self.linear(outputs), attention_weights
        
class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """  
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)

class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """

        padding_locations = torch.where(X == self.vocab_size, torch.zeros_like(X, dtype=torch.float64),
                                        torch.ones_like(X, dtype=torch.float64))
        padding_mask = torch.einsum("bi,bj->bij", (padding_locations, padding_locations))

        X = self.embedding_layer(X)
        X = self.position_encoding(X)
        for block in self.blocks:
            X = block(X, padding_mask)
        return X, padding_locations

class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """  
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)
        
        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)

        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights

class Decoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()
        
        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1)

        self.vocab_size = vocab_size

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        # raise NotImplementedError()
        return torch.tril(torch.ones(size=(seq_length, seq_length)))


    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """
        
        # Lookahead mask
        seq_length = target.shape[1]
        mask = self._lookahead_mask(seq_length)

        # Padding masks
        target_padding = torch.where(target == self.vocab_size, torch.zeros_like(target, dtype=torch.float64), 
                                     torch.ones_like(target, dtype=torch.float64))
        target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, target_padding))
        mask1 = torch.multiply(mask, target_padding_mask)

        source_target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, source_padding))

        target = self.embedding_layer(target)
        target = self.position_encoding(target)

        att_weights = None
        for block in self.blocks:
            target, att = block(encoded_source, target, mask1, source_target_padding_mask)
            if att_weights is None:
                att_weights = att

        y = self.linear(target)
        return y, att_weights

class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)

        self.target_vocab_size = target_vocab_size


    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)

    def predict(self, source: List[int], beam_size=1, max_length=12) -> List[int]:
        """
        Given a sentence in the source language, you should output a sentence in the target
        language of length at most `max_length` that you generate using a beam search with
        the given `beam_size`.

        Note that the start of sentence token is 0 and the end of sentence token is 1.

        Return the final top beam (decided using average log-likelihood) and its average
        log-likelihood.

        Hint: The follow functions may be useful:
            - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
            - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
        """
        self.eval() # Set the PyTorch Module to inference mode (this affects things like dropout)
        
        if not isinstance(source, torch.Tensor):
            source_input = torch.tensor(source).view(1, -1)
        else:
            source_input = source.view(1, -1)

        # # TODO: Implement beam search.
        # # raise NotImplementedError()
        # beam_size = 2

        targets = torch.tensor(0).view(1,-1)
        source_input = source_input
        # print(targets)
        # print(source_input)
        output, _ = self.forward(source_input, targets)
        logits_next_pred = output[:,-1,:]
        # print(logits.shape)
        log_probs = torch.log_softmax(logits_next_pred, dim=-1)

        # get B candidates to start the beam search
        topk = torch.topk(log_probs.view(1, -1), k=beam_size)
        # print(topk)
        curr_candidates, curr_logsum = topk.indices, topk.values
        # print(curr_logsum.flatten().tolist())
        # print(curr_candidates)
        targets = torch.cat((targets.repeat((1,beam_size)).view(beam_size, -1), curr_candidates.view(beam_size, -1) % (self.target_vocab_size+1)), 1).tolist()
        # curr_logsum = curr_logsum.flatten().tolist()
        curr_logsum = curr_logsum.view(-1, 1).tolist()
        
        beam = [{'tokens': targets[i], 'log_prob': curr_logsum[i], 'next_log_prob': 0, 'done': False, 'length': 2} for i in range(beam_size)]
        # print(beam)
        # print("############################")

        final_candidates = []

        # Write code to get the first predcitions, these have to be unique which is not the case if we implement it as below

        for i in range(max_length):

            next_beam = []

            for candidate in beam:
                if not candidate['done']:
                    target_input = torch.tensor(candidate['tokens']).view(1,-1)
                    target_input = target_input.to(source_input.device)
                    # print(target_input.shape)
                    output, _ = self.forward(source_input, target_input)
                    # print(output.shape)
                    output_probs = torch.log_softmax(output[:,-1,:], dim = -1)
                    # print(output_probs.shape)
                    candidate['next_log_prob'] = np.array(output_probs.flatten().tolist()) + candidate['log_prob'][-1] 
                    # print(len(candidate))
                    # print(candidate['tokens']," - ",torch.topk(output_probs, beam_size).indices, torch.topk(output_probs, beam_size).values)
            
            all_probs = torch.tensor(np.array([candidate['next_log_prob'] for candidate in beam]))
            # all_probs = torch.tensor(list(map(lambda candidate : candidate['next_log_prob'] + candidate['log_prob'], beam)))
            # print(beam_size)
            top_probs, top_indices = torch.topk(all_probs.view(1,-1), beam_size)
            top_probs, top_indices = top_probs.flatten(), top_indices.flatten()
            # beam_indices = (top_indices//self.target_vocab_size).tolist()
            # beam_tokens = (top_indices%self.target_vocab_size).tolist()
            # print(top_indices//(self.target_vocab_size+1))
            # print(top_indices%(self.target_vocab_size+1))
            # print(top_probs)

            for prob, idx in zip(top_probs, top_indices):
                candidate = beam[(idx//(self.target_vocab_size+1)).tolist()]
                new_candidate = {
                    'tokens': candidate['tokens'] + [(idx%(self.target_vocab_size+1)).tolist()],
                    'log_prob': candidate['log_prob'] + [prob.item()],
                    'done': (idx%(self.target_vocab_size+1) == 1) | (candidate['length'] +1 == max_length),  # Check if the next token is the end of sentence token
                    'next_log_prob': 0,
                    'length': candidate['length'] + 1
                }
                next_beam.append(new_candidate)

            next_beam = sorted(next_beam, key=lambda x: x['log_prob'][-1] / x['length'], reverse=True)[:beam_size]
            beam_size -= sum(candidate['done'] for candidate in next_beam)
            next_beam = [candidate for candidate in next_beam if candidate['length'] <= max_length]
            final_candidates.extend([candidate for candidate in next_beam if candidate['done']])
            beam = [candidate for candidate in next_beam if not candidate['done']]
            # beam = next_beam
            # print(final_candidates)
            # print(beam)
            # print("############################")

            if beam_size == 0:
                break
        
        final_translation = max(final_candidates, key=lambda x: np.array(x['log_prob'][-1]) / x['length'])

        return final_translation['tokens'], final_translation['log_prob'][-1]/final_translation['length']
        # return final_translation['tokens'], sum(final_translation['log_prob'])

def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab

def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):
    
    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)

def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)

def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 1:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 1:
        target_length += 1

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    plt.show()
    plt.close()

def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")
    return epoch_train_loss, epoch_test_loss

def ngrams(input, n):
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

def precision(target_seq, predicted_seq, n):

    y_hat = ngrams(predicted_seq, n)
    y = ngrams(target_seq, n)
    unique_ngrams_pred = Counter([tuple(i) for i in y_hat]).keys()
    val = 0
    for x in unique_ngrams_pred:
        # print([y.count(list(x)), y_hat.count(list(x))])
        val += min([y.count(list(x)), y_hat.count(list(x))])
    return val/(len(predicted_seq) - n +1)


def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    *** For students in 10-617 only ***
    (Students in 10-417, you can leave `raise NotImplementedError()`)

    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (0), EOS (1), and padding (anything after EOS)
    from the predicted and target sentences.
    
    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """
    # raise NotImplementedError()
    try:
        peos_index = predicted.index(1)
        psos_index = predicted.index(0)
        predicted_seq = predicted[psos_index+1:peos_index]
    except ValueError:
        psos_index = predicted.index(0)
        predicted_seq = predicted[psos_index+1:]
    try:
        teos_index = target.index(1)
        tsos_index = target.index(0)
        target_seq = target[tsos_index+1:teos_index]
    except ValueError:
        tsos_index = target.index(0)
        target_seq = target[tsos_index+1:]


    BLEU = []
    for i in range(1,N+1):
        if (len(predicted_seq)<=i) | (len(target_seq)<=i):
            BLEU.append(0)
            break
        else:
            p = precision(target_seq=target_seq, predicted_seq=predicted_seq, n=i)
            BLEU.append(np.power(p, 1/N))

    brevity_penalty = min([1, np.exp(1-(len(target_seq)/len(predicted_seq)))])

    bleu_k = np.prod(BLEU)

    return brevity_penalty*bleu_k

if __name__ == "__main__":
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()
    
    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), 12)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), 12)

    #2a
    model_2a = Transformer(len(source_vocab), len(target_vocab), 256, 1, 1, 1)

    epoch_train_loss, epoch_test_loss = train(model_2a, train_source, train_target, test_source, test_target, len(target_vocab))
    torch.save(model_2a.state_dict(), 'model_2a.pkl')
    plt.plot(epoch_train_loss, label = 'train_loss')
    plt.plot(epoch_test_loss, label = 'test_loss')
    plt.xlabel("epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Train and Test loss for (1,1,1) config")
    plt.legend()
    # plt.savefig('/Users/gauravmolugu/Downloads/10617/10617_F23_HW3_Part1/Figure_2a.png',)
    plt.show()

    #2b
    model_2b = Transformer(len(source_vocab), len(target_vocab), 256, 1, 1, 4)

    epoch_train_loss, epoch_test_loss = train(model_2b, train_source, train_target, test_source, test_target, len(target_vocab))
    torch.save(model_2b.state_dict(), 'model_2b.pkl')
    plt.plot(epoch_train_loss, label = 'train_loss')
    plt.plot(epoch_test_loss, label = 'test_loss')
    plt.xlabel("epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Train and Test loss for (1,1,4) config")
    plt.legend()
    # plt.savefig('/Users/gauravmolugu/Downloads/10617/10617_F23_HW3_Part1/Figure_2b.png')
    plt.show()

    #2c
    model_2c = Transformer(len(source_vocab), len(target_vocab), 256, 2, 2, 1)

    epoch_train_loss, epoch_test_loss = train(model_2c, train_source, train_target, test_source, test_target, len(target_vocab))
    torch.save(model_2c.state_dict(), 'model_2c.pkl')
    plt.plot(epoch_train_loss, label = 'train_loss')
    plt.plot(epoch_test_loss, label = 'test_loss')
    plt.xlabel("epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Train and Test loss for (2,2,1) config")
    plt.legend()
    # plt.savefig('/Users/gauravmolugu/Downloads/10617/10617_F23_HW3_Part1/Figure_2c.png')
    plt.show()

    #2d
    model_2d = Transformer(len(source_vocab), len(target_vocab), 256, 2, 2, 4)

    epoch_train_loss, epoch_test_loss = train(model_2d, train_source, train_target, test_source, test_target, len(target_vocab))
    torch.save(model_2d.state_dict(), 'model_2d.pkl')
    plt.plot(epoch_train_loss, label = 'train_loss')
    plt.plot(epoch_test_loss, label = 'test_loss')
    plt.xlabel("epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Train and Test loss for (2,2,4) config")
    plt.legend()
    # plt.savefig('/Users/gauravmolugu/Downloads/10617/10617_F23_HW3_Part1/Figure_2d.png')
    plt.show()

    #2e
    model_2e = Transformer(len(source_vocab), len(target_vocab), 256, 2, 4, 4)

    epoch_train_loss, epoch_test_loss = train(model_2e, train_source, train_target, test_source, test_target, len(target_vocab))
    torch.save(model_2e.state_dict(), 'model_2e.pkl')
    plt.plot(epoch_train_loss, label = 'train_loss')
    plt.plot(epoch_test_loss, label = 'test_loss')
    plt.xlabel("epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Train and Test loss for (2,4,4) config")
    plt.legend()
    # plt.savefig('/Users/gauravmolugu/Downloads/10617/10617_F23_HW3_Part1/Figure_2e.png')
    plt.show()

#####################################

    # model = Transformer(len(source_vocab), len(target_vocab), 256, 2, 4, 4)
    # model.load_state_dict(torch.load("model_2e.pkl"))
    # model.eval()

    # #3
    # for i in range(8):
    #     source = test_source[i]
    #     target = test_target[i]
    #     target_pred, loglikelihood = model.predict(source, beam_size=3)
    #     source_sentence = decode_sentence(source, source_vocab)
    #     target_sentence = decode_sentence(target, target_vocab)
    #     pred_sentence = decode_sentence(target_pred, target_vocab)
    #     print(f"Source Sentence: {source_sentence}")
    #     print(f"Target Sentence: {target_sentence}")
    #     print(f"Predicted Sentence: {pred_sentence}")
    #     print(f"Average Loglikelihood: {loglikelihood}")

    # #4
    # source, target = train_source[:3], train_target[:3]
    # _, att_weights = model(source, target)
    # att_weights = att_weights.detach().numpy()
    # for i in range(3):
    #     output, _ = model.predict(source[i])
    #     for j in range(4):
    #         visualize_attention(source[i].detach().numpy(), output, source_vocab, target_vocab, att_weights[i, j])

    # #5
    # lls = []
    # for i in range(1, 9):
    #     total_ll = 0
    #     for j in tqdm(range(100), desc=f"Beam Size {i}"):
    #         _, ll = model.predict(test_target[j], beam_size=i)
    #         total_ll += ll
    #     total_ll /= 100
    #     lls.append(total_ll)
    # plt.plot(range(1,9),lls)
    # plt.xlabel("Beam Size")
    # plt.ylabel("Average Normalized Log-Likelihood")
    # plt.title("Average Normalized Log-Likelihood vs. Beam Size")
    # plt.show()

    #6
    # model_list = {'model_2a':Transformer(len(source_vocab), len(target_vocab), 256, 1, 1, 1),
    #               'model_2b':Transformer(len(source_vocab), len(target_vocab), 256, 1, 1, 4),
    #               'model_2c':Transformer(len(source_vocab), len(target_vocab), 256, 2, 2, 1),
    #               'model_2d':Transformer(len(source_vocab), len(target_vocab), 256, 2, 2, 4),
    #               'model_2e':Transformer(len(source_vocab), len(target_vocab), 256, 2, 4, 4)}
    
    # for mod, config in model_list.items():

    #     model = config
    #     model.load_state_dict(torch.load(mod+".pkl"))
    #     model.eval()

    #     bleu_1, bleu_2, bleu_3, bleu_4 = [],[],[],[]

    #     for source, target in zip(test_source, test_target):
    #         pred, loglikelihood = model.predict(source, beam_size=3)
    #         # print(pred)
    #         bleu_1.append(bleu_score(pred, target.tolist(), N=1))
    #         bleu_2.append(bleu_score(pred, target.tolist(), N=2))
    #         bleu_3.append(bleu_score(pred, target.tolist(), N=3))
    #         bleu_4.append(bleu_score(pred, target.tolist(), N=4))

    #     print(f'{mod} - Average BLEU-1 score : {np.mean(np.array(bleu_1))}')
    #     print(f'{mod} - Average BLEU-2 score : {np.mean(np.array(bleu_2))}')
    #     print(f'{mod} - Average BLEU-3 score : {np.mean(np.array(bleu_3))}')
    #     print(f'{mod} - Average BLEU-4 score : {np.mean(np.array(bleu_4))}')


