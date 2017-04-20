import numpy as np
from model import LSTM, Weights
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
    print 'x shape', x.shape, 'y shape', y.shape
    for i in range(epochs):
        for x_i in x:
            outputs.append(lstm.predict(x_i))
        if network_type == 'encoder':
            BTTEncoder(outputs, x, y, lstm)

        elif network_type == 'decoder':
            BTTDecoder(outputs, x, y, lstm)

        outputs = list()


def BTTEncoder(outputs, x, y, lstm):
    diff = Weights(lstm.input_nodes,lstm.hidden_nodes,lstm.output_nodes, 0)
    for output, yi, xi in zip(outputs, y, x):
        print 'Cost: ', cost(yi, output[-1].v)
        print 'Cost Gradient', error_gradient(output[-1].v, yi)

        de_dsoftmax = error_gradient(output[-1].v, yi)
        dsoftmax_dh = dsoftmax_dx(output[-1].h)

        de_dh = de_dsoftmax*dsoftmax_dh
        de_ds = np.zeros_like(de_dh)

        diff.whv += de_dh / len(xi)
        diff.bv += de_dsoftmax / len(xi)

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
            diff.wgx += np.dot(dtanh_dwgx, ((de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh).T) / len(xi)
            diff.wgh += ((de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dwgh).T / len(xi)
            diff.bg += (de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dbg / len(xi)

            # i gradients
            diff.wix += np.dot(dsigmoid_dwix, ((de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid).T) / len(xi)
            diff.wih += ((de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dwih).T / len(xi)
            diff.bi += (de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dbi / len(xi)

            # f gradients
            diff.wfx += np.dot(dsigmoid_dwfx, (de_dh*dh_ds*ds_df*df_dsigmoid).T) / len(xi)
            diff.wfh += (de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dwfh).T / len(xi)
            diff.bf += de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dbf / len(xi)

            # o gradients
            diff.wox += np.dot(dsigmoid_dwox, (de_dh * dh_do * do_dsigmoid).T) / len(xi)
            diff.woh += (de_dh * dh_do * do_dsigmoid * dsigmoid_dwoh).T / len(xi)
            diff.bo += de_dh * dh_do * do_dsigmoid * dsigmoid_dbo / len(xi)

            de_ds = (de_dh * dh_ds + de_ds) * state.f

            de_dh = (diff.wgh + diff.wih + diff.wfh + diff.woh).T


def BTTDecoder(outputs, x, y, lstm):
    diff = Weights(lstm.input_nodes, lstm.hidden_nodes, lstm.output_nodes, 0)
    for output, yi, xi in zip(outputs, y, x):

        de_dh = np.zeros_like(output[0].h.shape)
        de_ds = np.zeros_like(de_dh)

        for i in range(len(xi) - 1, -1, -1):
            state = output[i + 1]

            de_dsoftmax = error_gradient(output[i+1].v, yi[i])
            dsoftmax_dh = dsoftmax_dx(output[i+1].h)
            de_dh += de_dsoftmax*dsoftmax_dh

            diff.whv += de_dh / len(xi)
            diff.bv += de_dsoftmax / len(xi)

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
            diff.wgx += np.dot(dtanh_dwgx, ((de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh).T) / len(xi)
            diff.wgh += ((de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dwgh).T / len(xi)
            diff.bg += (de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dbg / len(xi)

            # i gradients
            diff.wix += np.dot(dsigmoid_dwix, ((de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid).T) / len(xi)
            diff.wih += ((de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dwih).T / len(xi)
            diff.bi += (de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dbi / len(xi)

            # f gradients
            diff.wfx += np.dot(dsigmoid_dwfx, (de_dh * dh_ds * ds_df * df_dsigmoid).T) / len(xi)
            diff.wfh += (de_dh * dh_ds * ds_df * df_dsigmoid * dsigmoid_dwfh).T / len(xi)
            diff.bf += de_dh * dh_ds * ds_df * df_dsigmoid * dsigmoid_dbf / len(xi)

            # o gradients
            diff.wox += np.dot(dsigmoid_dwox, (de_dh * dh_do * do_dsigmoid).T) / len(xi)
            diff.woh += (de_dh * dh_do * do_dsigmoid * dsigmoid_dwoh).T / len(xi)
            diff.bo += de_dh * dh_do * do_dsigmoid * dsigmoid_dbo / len(xi)

            de_ds = (de_dh * dh_ds + de_ds) * state.f

            de_dh = (diff.wgh + diff.wih + diff.wfh + diff.woh).T




def error_gradient(output, y):
    c = 0
    for i in range(len(y)):
        c += 2*(y[i] - output[i])

    return c / len(y)


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
    # x, y = load_data()
    x = np.random.random((10, 10, 50, 1))
    y = np.random.random((10, 10, 50, 1))
    encoder_lstm = LSTM(50, 100, 50)
    train(encoder_lstm, x, y[:][0], 20, 'encoder')

    encoder_lstm.save_weights('encoder.pkl')


if __name__ == '__main__':
    main()