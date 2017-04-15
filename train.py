import numpy as np
from model import LSTM, Weights
from IPython import embed


def train_encoder(lstm, x, y, epochs):
    outputs = []

    for i in range(epochs):
        for x_i in x:
            outputs.append(lstm.predict(x_i))
        BTTEncoder(outputs, y, lstm)


def train(lstm, x, y, epochs, network_type):
    outputs = list()

    for i in range(epochs):
        for x_i in x:
            outputs.append(lstm.predict(x_i))
        if network_type == 'encoder':
            BTTEncoder(outputs, y, x, lstm)

        elif network_type == 'decoder':
            BTTDecoder(outputs, y, lstm)

        outputs = list()


def BTTEncoder(outputs, y, x, lstm):
    diff = Weights(lstm.input_nodes,lstm.hidden_nodes,lstm.output_nodes, 0)
    for output, yi, xi in zip(outputs, y, x):
        de_dsoftmax = error_gradient(output[-1].v, yi)
        dsoftmax_dh = dsoftmax_dx(output[-1].h)

        de_dh = de_dsoftmax*dsoftmax_dh
        de_ds = np.zeros_like(de_dh)

        diff.whv += de_dh / len(outputs)
        diff.bv += de_dh / len(outputs)

        for i in range(len(output) - 1, 1, -1):
            state = output[i]
            
            # lstm cell gradients
            dh_ds = state.o
            dh_do = state.s

            # s to gates gradients
            ds_dg = state.i
            ds_di = state.g
            ds_df = output[i-1].s

            # gate to activation gradients
            dg_dtanh = 1. - state.g**2
            di_dsigmoid = state.i*(1-state.i)
            df_dsigmoid = state.f*(1-state.f)
            do_dsigmoid = state.o*(1-state.o)

            dtanh_dwgx = xi[i]
            dtanh_dwgh = output[i - 1].h
            dtanh_dbg = 1

            dsigmoid_dwix = xi[i]
            dsigmoid_dwih = output[i -1].h
            dsigmoid_dbi = 1

            dsigmoid_dwfx = xi[i]
            dsigmoid_dwfh = output[i - 1].h
            dsigmoid_dbf = 1

            dsigmoid_dwox = xi[i]
            dsigmoid_dwoh = output[i - 1].h
            dsigmoid_dbo = 1

            # g gradients
            diff.wgx += (de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dwgx / len(outputs)
            diff.wgh += (de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dwgh / len(outputs)
            diff.bg += (de_dh * dh_ds + de_ds) * ds_dg * dg_dtanh * dtanh_dbg / len(outputs)

            # i gradients
            diff.wix += (de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dwix / len(outputs)
            diff.wih += (de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dwih / len(outputs)
            diff.bi += (de_dh * dh_ds + de_ds) * ds_di * di_dsigmoid * dsigmoid_dbi / len(outputs)

            # f gradients
            diff.wfx += de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dwfx / len(outputs)
            diff.wfh += de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dwfh / len(outputs)
            diff.bf += de_dh*dh_ds*ds_df*df_dsigmoid*dsigmoid_dbf / len(outputs)

            # o gradients
            diff.wox += de_dh * dh_do * do_dsigmoid * dsigmoid_dwox / len(outputs)
            diff.woh += de_dh * dh_do * do_dsigmoid * dsigmoid_dwoh / len(outputs)
            diff.bo += de_dh * dh_do * do_dsigmoid * dsigmoid_dbo / len(outputs)

            de_ds = (de_dh * dh_ds + de_ds) * state.f

            de_dh = diff.wgh + diff.wih + diff.wfh + diff.woh







def BTTDecoder(outputs, y, lstm):
    return


def error_gradient(output, y):
    c = 0
    for i in range(len(y)):
        c += 2*(y[i] - output[i])

    return c / len(y)


def cost(y, output):
    c = 0
    for i in range(y.shape[0]):
        c += 0.5 * (y[i] - output[i][-1]) ** 2
    return c

def dsoftmax_dx(x):
    ex = np.exp(x - np.max(x))
    dex = (ex*ex.sum() + ex*ex) / (ex.sum() ** 2)
    return dex



def main():
    # x, y = load_data()

    encoder_lstm = LSTM(50, 100, 50)
    train(encoder_lstm, x, y[:][0], 20, 'encoder')

    encoder_lstm.save_weights()


embed()