import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import CharacterTable, transform
from utils import restore_model, decode_sequences

error_rate = 0.6
reverse = True
model_path = './models/seq2seq.h5'
hidden_size = 512
sample_mode = 'argmax'
remove_chars = '[#$%"\+@<=>!&,-.?:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'

test_sentence = 'The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.'


if __name__ == '__main__':
    vocab = []
    with open('./data/words.txt', 'r') as f:
        for line in f:
            vocab.extend([re.sub(remove_chars, '', token)
                          for token in re.split("[-]", line)])
    vocab = list(filter(None, set(vocab)))
    # `maxlen` is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    maxlen = max([len(token) for token in vocab]) + 2
    train_encoder, train_decoder, train_target = transform(
        vocab, maxlen, error_rate=error_rate, shuffle=False)

    tokens = [re.sub(remove_chars, '', token)
              for token in re.split("[-\n ]", test_sentence)]
    tokens = list(filter(None, tokens))
    nb_tokens = len(tokens)
    misspelled_tokens, _, target_tokens = transform(
        tokens, maxlen, error_rate=error_rate, shuffle=False)

    input_chars = set(' '.join(train_encoder))
    target_chars = set(' '.join(train_decoder))
    input_ctable = CharacterTable(input_chars)
    target_ctable = CharacterTable(target_chars)
    
    encoder_model, decoder_model = restore_model(model_path, hidden_size)
    
    input_tokens, target_tokens, decoded_tokens = decode_sequences(
        misspelled_tokens, target_tokens, input_ctable, target_ctable,
        maxlen, reverse, encoder_model, decoder_model, nb_tokens,
        sample_mode=sample_mode, random=False)
    
    print('-')
    print('Input sentence:  ', ' '.join([token for token in input_tokens]))
    print('-')
    print('Decoded sentence:', ' '.join([token for token in decoded_tokens]))
    print('-')
    print('Target sentence: ', ' '.join([token for token in target_tokens]))
