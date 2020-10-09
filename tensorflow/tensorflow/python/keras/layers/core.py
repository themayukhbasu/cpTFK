"""
    Core Keras layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import operator
import sys
import textwrap
import types as python_types
import warnings

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context

from tensorflow.python.keras import backend as K


from tensorflow.python.keras.engine.base_layer import Layer


from tensorflow.python.ops import array_ops

from tensorflow.python.ops import math_ops


from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.Masking')
class Masking(Layer):
    """
        Masks a sequence by using a mask value to skip timesteps.

        For each timestep in the input tensor (dimension #1 in the tensor),
        if all values in the input tensor at that timestep
        are equal to `mask_value`, then the timestep will be masked (skipped)
        in all downstream layers (as long as they support masking).

        if any downstream layer does not support masking yet receives such
        an input mask, an exception will be raised.

        Example:
            Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
            to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
            lack data for these timesteps. You can:

            - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
            - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

            ```python
            samples, timesteps, features = 32, 10, 8
            inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
            inputs[:, 3, :] = 0.
            inputs[:, 5, :] = 0.

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Masking(mask_value=0.,
                                            input_shape=(timesteps, features)))
            model.add(tf.keras.layers.LSTM(32))

            output = model(inputs
            # The time step 3 and 5 will be skipped from LSTM calculation.
            ```
        See [the masking and padding guide](
            https://www.tensorflow.org/guide/keras/masking_and_padding)
        for more details.
    """
    """
        # -- mb !ND
        We are trying to mask a timestep (1st dimension of inputs[timestep x feature] )
        in case all the feature elements are equal to the `mask_value`
        
        What is under the hood?
        
        Lets look in the `call` method.
         1. It computes the `boolean_mask`
         2. It applies the mask to the inputs
         3. it computes `outputs._keras_mask`
         
         1.1:
            First of all, note that the `mask_value` arg 
            can be of any numerical value and dtype.
            
            What we are basically going to populate the `boolean_mask` array
            with False for the elements which are equal to the `mask_value`
            and True for the elements which are != `mask_value`
            
            So we would first need to do is to find out which elements in
            the `inputs` match_the `mask_value`:
                `K.any(
                    math_ops.not_equal(inputs, self.mask_value), 
                    axis=-1, 
                    keepdims=True
                )`
            We are looking at the last axis, because that's the feature dimension
            The above `maths_ops.not_equal` command returns 
                True when elements != `mask_value`
                False when elements == `mask_value`
            So for eg,
                if for a timestep, there are 4 feature elements 
                all match the `mask_vale`
                then the resulant timestep x feature will be
                    [False, False, False, False]
            The K.any() or the tf.keras.backend.any() 
                will compute bitwise reduction (logical OR) across dimensions of a tensor
                # K.any() does the following:
                #   Calls the tf.math.reduce_any()
                # This does NOT have the same function as python's any() method
                # Example:
                #   x = [
                #       [True, True],
                #       [False, False]
                #   ]
                #   reduce_any(x) # True
                #   reduce_any(x, axis=0) # [True, True]
                #   reduce_any(x, axis=1) # [True, False]
            So `boolean mask` will be of shape `(timestep, 1)`,
            if the feature elements == `mask value`, then `[timestep #, False]`    // `timestep #` = the timestep index
            if the feature elements != `mask value`, then `[timestep #, True]` 
                 
        2.1:
            Now its going to compute the `outputs` of the layer 
            by bitwise multiplication of the `inputs` with the `boolean_mask`
            
            But before we can multiple, there is one more step to be followed
            Since the dtype of `boolean_mask` is `bool` i.e. True or False,
            we have to typecast it to dtype of the inputs
        
        3.1:
            !!pending
    """

    def __init__(self, mask_value=0., **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value
        self._compute_output_and_mask_jointly = True

    def compute_mask(self, inputs, mask=None):
        # --mb !Done
        ## Don't think this method is being used!
        # the K.any() is tf.keras.backend.any
        # Computes Bitwise reduction (logical OR) of elements across dimensions of a tensor.
        # This does the following:
        #   1. Typecasts the x (tensor) to boolean
        #   2. calls the tf.math.reduce_any()
        # This does NOT have the same function as python's any() method
        # Example:
        #   x = [
        #       [True, True],
        #       [False, False]
        #   ]
        #   any(x) # True
        #   any(x, axis=0) # [True, True]
        #   any(x, axis=1) # [True, False]

        return K.any(math_ops.not_equal(inputs, self.mask_value), axis=1)

    def call(self, inputs):
        # --mb !ND
        boolean_mask = K.any(
            math_ops.not_equal(inputs, self.mask_value),
            axis=-1,    # reducing the last axis
            keepdims=True)
        outputs = inputs * math_ops.cast(boolean_mask, inputs.dtype)
        # Compute the mask and outputs simultaneously.
        outputs._keras_mask = array_ops.squeeze(boolean_mask, asix=-1)
        return outputs