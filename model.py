from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from keras import optimizers, metrics, backend as K

# For use with truncated metrics,
# take maxlen from the validation set.
# Hacky and hard-coded for now.
VAL_MAXLEN = 16


def truncated_acc(y_true, y_pred):
    y_true = y_true[:, :VAL_MAXLEN, :]
    y_pred = y_pred[:, :VAL_MAXLEN, :]
    
    acc = metrics.categorical_accuracy(y_true, y_pred)
    return K.mean(acc, axis=-1)


def truncated_loss(y_true, y_pred):
    y_true = y_true[:, :VAL_MAXLEN, :]
    y_pred = y_pred[:, :VAL_MAXLEN, :]
    
    loss = K.categorical_crossentropy(
        target=y_true, output=y_pred, from_logits=False)
    return K.mean(loss, axis=-1)


def seq2seq(hidden_size, nb_input_chars, nb_target_chars):
    """Adapted from:
    https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
    """
    
    # Define the main model consisting of encoder and decoder.
    encoder_inputs = Input(shape=(None, nb_input_chars),
                           name='encoder_data')
    encoder_lstm = LSTM(hidden_size, recurrent_dropout=0.2,
                        return_sequences=True, return_state=False,
                        name='encoder_lstm_1')
    encoder_outputs = encoder_lstm(encoder_inputs)
    
    encoder_lstm = LSTM(hidden_size, recurrent_dropout=0.2,
                        return_sequences=False, return_state=True,
                        name='encoder_lstm_2')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_outputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, nb_target_chars),
                           name='decoder_data')
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the return
    # states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(hidden_size, dropout=0.2, return_sequences=True,
                        return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_softmax = Dense(nb_target_chars, activation='softmax',
                            name='decoder_softmax')
    decoder_outputs = decoder_softmax(decoder_outputs)

    # The main model will turn `encoder_input_data` & `decoder_input_data`
    # into `decoder_target_data`
    model = Model(inputs=[encoder_inputs, decoder_inputs],
                  outputs=decoder_outputs)
    
    adam = optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy', truncated_acc, truncated_loss])
    
    # Define the encoder model separately.
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

    # Define the decoder model separately.
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_softmax(decoder_outputs)
    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
                          outputs=[decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
