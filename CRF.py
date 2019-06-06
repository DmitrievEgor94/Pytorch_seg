import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary
import numpy as np

def get_prediction(pred_after_cnn, image):
    unary = pred_after_cnn.transpose((2, 0, 1))

    unary = -unary.reshape((unary.shape[0],-1))

    unary = np.ascontiguousarray(unary).astype(np.float32)

    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], pred_after_cnn.shape[2])

    d.setUnaryEnergy(unary)

    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    return res

# def get_prediction(pred_after_cnn, image):
#     d = dcrf.DenseCRF2D(pred_after_cnn.shape[1], pred_after_cnn.shape[0], pred_after_cnn.shape[2])
#
#     U = pred_after_cnn.transpose((2, 0, 1))
#     U = -U.reshape((U.shape[0],-1))
#
#     U = np.ascontiguousarray(U).astype(np.float32)
#
#     d.setUnaryEnergy(U)
#
#     d.addPairwiseGaussian(sxy=3, compat=3)
#     d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
#
#     Q = d.inference(5)
#
#     map = np.argmax(Q, axis=0).reshape(image.shape[0:2])
#
#     return map