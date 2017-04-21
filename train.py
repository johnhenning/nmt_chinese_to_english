import cPickle
import numpy as np
from model import LSTM, Weights
import sys
from IPython import embed
from numba import cuda


def train_encoder(lstm, x, y, epochs):
    outputs = []

    for i in range(epochs):
        for x_i in x:
            outputs.append(lstm.predict(x_i))
        BTTEncoder(outputs, y, lstm)


def train(lstm, x, y, epochs, network_type):
    outputs = list()
    for i in range(epochs):
        print '\nEpoch %d' % i

        if network_type == 'encoder':
            for x_i in x:
                outputs.append(lstm.predict(x_i))
            BTTEncoder(outputs, x, y, lstm)

        elif network_type == 'decoder':
            for y_i, x_i in zip(y, x):
                outputs.append(lstm.predict(y_i[:-1], x_i[-1].h))
            print 'Before BTT Decoder'
            y_input = [x[:-1] for x in y]
            y_output = [x[1:] for x in y]
            BTTDecoder(outputs, y_input, y_output, lstm)

        outputs = list()


def BTTEncoder(outputs, x, y, lstm):
    diff = Weights(lstm.input_nodes,lstm.hidden_nodes,lstm.output_nodes, 0)
    total_cost = 0
    for output, yi, xi in zip(outputs, y, x):
        total_cost += cost(yi, output[-1].v)

        de_dsoftmax = error_gradient(output[-1].v, yi)
        dsoftmax_dh = dsoftmax_dx(output[-1].h)

        de_dh = de_dsoftmax*dsoftmax_dh
        de_ds = np.zeros_like(de_dh)

        diff.whv += de_dh
        diff.bv += de_dsoftmax

        for i in range(len(xi) - 1, -1, -1):
            state = output[i+1]
            
            # lstm cell gradients
            dh_ds = state.o
            dh_do = state.s

            # s to gates gradients
            ds_dg = state.i
            ds_di = state.g
            ds_df = output[i].s

            # gate to activation gradients
            dg_dtanh = 1. - state.g**2
            di_dsigmoid = state.i*(1-state.i)
            df_dsigmoid = state.f*(1-state.f)
            do_dsigmoid = state.o*(1-state.o)

            dtanh_dwgx = xi[i]
            dtanh_dwgh = output[i].h
            dtanh_dbg = 1

            dsigmoid_dwix = xi[i]
            dsigmoid_dwih = output[i].h
            dsigmoid_dbi = 1

            dsigmoid_dwfx = xi[i]
            dsigmoid_dwfh = output[i].h
            dsigmoid_dbf = 1

            dsigmoid_dwox = xi[i]
            dsigmoid_dwoh = output[i].h
            dsigmoid_dbo = 1

            # g gradients
            diff.wgx += np.dot(dtanh_dwgx, ((de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh).T)
            diff.wgh += ((de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dwgh).T
            diff.bg += (de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dbg

            # i gradients
            diff.wix += np.dot(dsigmoid_dwix, ((de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid).T)
            diff.wih += ((de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dwih).T
            diff.bi += (de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dbi

            # f gradients
            diff.wfx += np.dot(dsigmoid_dwfx, (de_dh*dh_ds*ds_df*df_dsigmoid).T)
            diff.wfh += (de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dwfh).T
            diff.bf += de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dbf

            # o gradients
            diff.wox += np.dot(dsigmoid_dwox, (de_dh * dh_do * do_dsigmoid).T)
            diff.woh += (de_dh * dh_do * do_dsigmoid * dsigmoid_dwoh).T
            diff.bo += de_dh * dh_do * do_dsigmoid * dsigmoid_dbo

            de_ds = (de_dh * dh_ds + de_ds) * state.f

            de_dh = (diff.wgh + diff.wih + diff.wfh + diff.woh).T

        update_weights(lstm, diff, 0.8)

    print total_cost


def update_weights(lstm, diff, learning_rate):
    lstm.weights.wgx -= diff.wgx * learning_rate
    lstm.weights.wgh -= diff.wgh * learning_rate
    lstm.weights.bg -= diff.bg * learning_rate

    lstm.weights.wix -= diff.wix * learning_rate
    lstm.weights.wih -= diff.wih * learning_rate
    lstm.weights.bi -= diff.bi * learning_rate

    lstm.weights.wfx -= diff.wfx * learning_rate
    lstm.weights.wfh -= diff.wfh * learning_rate
    lstm.weights.bf -= diff.bf * learning_rate

    lstm.weights.wox -= diff.wox * learning_rate
    lstm.weights.woh -= diff.woh * learning_rate
    lstm.weights.bo -= diff.bo * learning_rate

    lstm.weights.whv -= diff.whv * learning_rate
    lstm.weights.bv -= diff.bv * learning_rate


def BTTDecoder(outputs, x, y, lstm):
    diff = Weights(lstm.input_nodes, lstm.hidden_nodes, lstm.output_nodes, 0)
    count = 0
    for output, yi, xi in zip(outputs, y, x):
        de_dh = np.zeros_like(output[0].h.shape)
        de_ds = np.zeros_like(de_dh)

        for i in range(len(xi) - 1, -1, -1):
            print 'output shape', len(output), len(yi)
            state = output[i + 1]
            de_dsoftmax = error_gradient(output[i+1].v, yi[i])
            dsoftmax_dh = dsoftmax_dx(output[i+1].h)

            print de_dsoftmax.shape, dsoftmax_dh.shape
            de_dh += np.dot(dsoftmax_dh, de_dsoftmax.T)

            diff.whv += de_dh
            diff.bv += de_dsoftmax

            # lstm cell gradients
            dh_ds = state.o
            dh_do = state.s

            # s to gates gradients
            ds_dg = state.i
            ds_di = state.g
            ds_df = output[i].s

            # gate to activation gradients
            dg_dtanh = 1. - state.g ** 2
            di_dsigmoid = state.i * (1 - state.i)
            df_dsigmoid = state.f * (1 - state.f)
            do_dsigmoid = state.o * (1 - state.o)

            dtanh_dwgx = xi[i]
            dtanh_dwgh = output[i].h
            dtanh_dbg = 1

            dsigmoid_dwix = xi[i]
            dsigmoid_dwih = output[i].h
            dsigmoid_dbi = 1

            dsigmoid_dwfx = xi[i]
            dsigmoid_dwfh = output[i].h
            dsigmoid_dbf = 1

            dsigmoid_dwox = xi[i]
            dsigmoid_dwoh = output[i].h
            dsigmoid_dbo = 1

            # g gradients
            diff.wgx += np.dot(dtanh_dwgx, ((de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh).T)
            diff.wgh += ((de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dwgh).T
            diff.bg += (de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dbg

            # i gradients
            diff.wix += np.dot(dsigmoid_dwix, ((de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid).T)
            diff.wih += ((de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dwih).T
            diff.bi += (de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dbi

            # f gradients
            diff.wfx += np.dot(dsigmoid_dwfx, (de_dh*dh_ds*ds_df*df_dsigmoid).T)
            diff.wfh += (de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dwfh).T
            diff.bf += de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dbf

            # o gradients
            diff.wox += np.dot(dsigmoid_dwox, (de_dh * dh_do * do_dsigmoid).T)
            diff.woh += (de_dh * dh_do * do_dsigmoid * dsigmoid_dwoh).T
            diff.bo += de_dh * dh_do * do_dsigmoid * dsigmoid_dbo

            de_ds = (de_dh * dh_ds + de_ds) * state.f

            de_dh = (diff.wgh + diff.wih + diff.wfh + diff.woh).T

        update_weights(lstm, diff, 0.8)


def error_gradient(output, y):
    return -(y - output)




def cost(y, output):
    c = 0
    for i in range(y.shape[0]):
        c += 0.5 * (y[i] - output[i]) ** 2 / len(y)
    return c


def dsoftmax_dx(x):
    ex = np.exp(x - np.max(x))
    dex = (ex*ex.sum() + ex*ex) / (ex.sum() ** 2)
    return dex


def main():
    print 'Loading Data'
    x = cPickle.load(open('english_matrices.pkl', 'rb'))
    y = cPickle.load(open('chinese_matrices.pkl', 'rb'))
    print 'Done'

    # x = np.random.random((10, 10, 50, 1))
    # y = np.random.random((10, 10, 50, 1))
    encoder_lstm = LSTM(50, 100, 50)
    encoder_lstm.load_weights('encoder.pkl')
    outputs = []
    for i in range(10000):
        outputs.append(encoder_lstm.predict(x[i]))
    # for _ in range(10):
    #     for i in range(20):
    #         idx_start = i*500
    #         idx_end = min((i+1)*500, len(x))
    #         sys.stdout.write('\n\nTraining Data %d - %d' % (idx_start, idx_end))
    #         train(encoder_lstm, x[idx_start:idx_end], y[idx_start:idx_end][0], 50, 'encoder')
    #         encoder_lstm.save_weights('encoder.pkl')
    # outputs = encoder_lstm.predict(x[:10000])
    # encoder_lstm.save_weights('encoder.pkl')
    embed()
    decoder_lstm = LSTM(50, 100, 50)
    for _ in range(4):
        for i in range(20):
            idx_start = i * 500
            idx_end = min((i + 1) * 500, len(x))
            sys.stdout.write('\n\nTraining Data %d - %d' % (idx_start, idx_end))
            train(decoder_lstm, outputs[idx_start:idx_end], y[idx_start:idx_end], 50, 'decoder')
            decoder_lstm.save_weights('decoder.pkl')
if __name__ == '__main__':
    main()