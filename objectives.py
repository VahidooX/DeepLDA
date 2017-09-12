import theano.tensor as T
import theano
import numpy as np
from theano.compile.ops import as_op


@as_op(itypes=[theano.tensor.ivector],
       otypes=[theano.tensor.ivector])
def numpy_unique(a):
    return np.unique(a)


def lda_loss(n_components, margin):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """
    def inner_lda_objective(y_true, y_pred):
        """
        It is the loss function of LDA as introduced in the original paper. 
        It is adopted from the the original implementation in the following link:
        https://github.com/CPJKU/deep_lda
        Note: it is implemented by Theano tensor operations, and does not work on Tensorflow backend
        """
        r = 1e-4

        # init groups
        yt = T.cast(y_true.flatten(), "int32")
        groups = numpy_unique(yt)

        def compute_cov(group, Xt, yt):
            Xgt = Xt[T.eq(yt, group).nonzero()[0], :]
            Xgt_bar = Xgt - T.mean(Xgt, axis=0)
            m = T.cast(Xgt_bar.shape[0], 'float32')
            return (1.0 / (m - 1)) * T.dot(Xgt_bar.T, Xgt_bar)

        # scan over groups
        covs_t, updates = theano.scan(fn=compute_cov, outputs_info=None,
                                      sequences=[groups], non_sequences=[y_pred, yt])

        # compute average covariance matrix (within scatter)
        Sw_t = T.mean(covs_t, axis=0)

        # compute total scatter
        Xt_bar = y_pred - T.mean(y_pred, axis=0)
        m = T.cast(Xt_bar.shape[0], 'float32')
        St_t = (1.0 / (m - 1)) * T.dot(Xt_bar.T, Xt_bar)

        # compute between scatter
        Sb_t = St_t - Sw_t

        #costs = T.nlinalg.trace(T.dot(T.nlinalg.matrix_inverse(St_t), Sb_t))
        #return -costs

        # cope for numerical instability (regularize)
        Sw_t += T.identity_like(Sw_t) * r

        # return T.cast(T.neq(yt[0], -1), 'float32')*T.nlinalg.trace(T.dot(T.nlinalg.matrix_inverse(St_t), Sb_t))

        # compute eigenvalues
        evals_t = T.slinalg.eigvalsh(Sb_t, St_t)

        # get eigenvalues
        top_k_evals = evals_t[-n_components:]

        # maximize variance between classes
        # (k smallest eigenvalues below threshold)
        thresh = T.min(top_k_evals) + margin
        top_k_evals = top_k_evals[(top_k_evals <= thresh).nonzero()]
        costs = T.mean(top_k_evals)

        return -costs

    return inner_lda_objective