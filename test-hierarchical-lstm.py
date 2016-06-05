import numpy as np

from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import SimpleRNN

from keras import backend as K


def main():
    input = np.random.rand(1, 20, 30, 2).astype('float32')

    tempt = input.reshape((1, 600, 2))
    target = np.sum(tempt, axis=1)

    model = Sequential()
    model.add(TimeDistributed(SimpleRNN(output_dim=2, input_shape=(30, 2), init='identity', inner_init='identity', activation='linear'), input_shape=(20, 30, 2)))
    model.add(SimpleRNN(output_dim=2, init='identity', inner_init='identity', activation='linear'))

    get_model_output = K.function([model.layers[0].input], [model.layers[-1].output])

    model_output = get_model_output([input])[0]

    print target.shape, ':', model_output.shape

    print np.array_equal(target, model_output)
    print 'Target ->', target
    print 'Model ->', model_output







if __name__ == '__main__':
    main()