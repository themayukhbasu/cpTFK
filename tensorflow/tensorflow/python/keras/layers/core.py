"""
    Core Keras layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context  # --mb for eager execution
from tensorflow.python.framework import constant_op

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.python.keras.utils import conv_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
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
        # --mb !Done
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
                # This has almost similar function as python's any() method, except that it works on arrays
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
            `outputs._keras_mask = array_ops.squeeze(boolean_mask, asix=-1)`
                the command attaches a mask tensor to the `output._keras_mask`
            The `_keras_mask` contains a mask tensor
                (2D tensor with shape (batch, sequence_length) or (batch, timestep) )
            The `array_ops.squeeze()` is tf.squeeze
                It removes dimensions of size 1 from shape of a tensor
                Example 1:
                    # `t` is a tensor of shape [1, 2, 1, 3, 1, 1]
                    tf.shape( tf.squeeze(t) )  # [2, 3]
                    tf.shape( tf.squeeze(t, [2, 4]) )   # [1, 2, 3, 1]
                Example 2:
                    > t = np.array([[[[1], [2], [3]], [[4], [5], [6]]]])
                    > t.shape
                    #  (1, 2, 3, 1)
                    > tf.squeeze(t)
                    #  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
                    #       array([[1, 2, 3],
                    #              [4, 5, 6]])>
            Example of _keras_mask:
                Padded Sequence:
                    [[ 711  632   71    0    0    0]
                     [  73    8 3215   55  927    0]
                     [  83   91    1  645 1253  927]]
                _keras_masked:
                   tf.Tensor(
                    [[ True  True  True False False False]
                     [ True  True  True  True  True False]
                     [ True  True  True  True  True  True]], shape=(3, 6), dtype=bool) 
            Why the `squeeze` needed?
                `boolean_mask` has a shape of (batch, timestep, 1)
                    tf.Tensor(
                    [[ [True]  [True]  [True] [False] [False]]
                     [ [True]  [True]  [True]  [True]  [True] [False]]], shape=(2, 5, 1), dtype=bool)
                squeeze remove the single dimensioned elements [[True] [False]] -> [True False]     
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
        # --mb !done
        boolean_mask = K.any(
            math_ops.not_equal(inputs, self.mask_value),
            axis=-1,  # reducing the last axis
            keepdims=True)
        outputs = inputs * math_ops.cast(boolean_mask, inputs.dtype)
        # Compute the mask and outputs simultaneously.
        outputs._keras_mask = array_ops.squeeze(boolean_mask, asix=-1)
        return outputs


@keras_export('keras.layers.Flatten')
class Flatten(Layer):
    """
        Flattens the input. Does not affect the batch size.

        Note: If inputs are shaped `(batch,)` without a feature axis,
        then flattening adds an extra channel dimension and output shape is
        `(batch, 1)`.

        Arguments:
            date_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, ..., channels)` while `channels_first` corresponds to
                inputs with shape `(batch, channels, ...)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".

        Example:
             model = tf.keras.Sequential()
             model.add(tf.keras.layers.Conv2D(64, 3, 3, input_shape=(3, 32, 32)))
             model.output_shape
             ## (None, 1, 10, 64)

             model.add(Flatten())
             model.output_shape
             ## (None, 640)

    """

    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(min_ndim=1)
        self._channels_first = self.data_format == 'channels_first'

    def call(self, inputs):
        if self._channels_first:
            rank = inputs.shape.rank
            if rank and rank > 1:
                # Switch to channels-last format.
                permutation = [0]
                permutation.extend(range(2, rank))
                permutation.append(1)
                inputs = array_ops.transpose(inputs, perm=permutation)

        if context.execution_eagerly():
            # Full static shape is guaranteed to be available.
            # Performance: Using `constant_op` is much faster than passing a list.
            flattened_shape = constant_op.constant([inputs.shape[0], -1])
            return gen_array_ops.reshape(inputs, flattened_shape)
