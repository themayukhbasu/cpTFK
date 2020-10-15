"""
    Core Keras layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import numpy as np

from tensorflow.python.eager import context  # --mb for eager execution
from tensorflow.python.framework import constant_op

from tensorflow.python.framework import tensor_shape

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
    """
        # --mb
        Util Function Explanation:
        - tf.rank()
            The rank of a tensor is the number of dimensions of the tensor
            Example:
                # shape of tensor 't' is [2, 2, 3]
                t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
                tf.rank(t)  # 3
  
        - tf.reshape()
            syntax:     `tf.reshape(tensor, shape, name=None)`
            Given a `tensor`, this operation returns a new `tf.Tensor` that has the same values
            as the `tensor` in the same order, except with a new shape given by `shape`
            **  If one component of the `shape` is the special value `-1`, the size 
                of that dimension is computed so that the total size remains constant.
                In particular, a `shape` of `[-1]` flattens into 1-D.
                At most one component of `shape` can be `-1`.
            Example:
                >>> t = [[[1, 1, 1],
                          [2, 2, 2]],
                         [[3, 3, 3],
                          [4, 4, 4]],
                         [[5, 5, 5],
                          [6, 6, 6]]]
                
                >>> print(tf.shape(t).numpy())    
                # [3 2 3]
                
                >>> tf.reshape(t, [2, -1]) 
                # <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
                # array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                #        [4, 4, 4, 5, 5, 5, 6, 6, 6]], dtype=int32)>
        
        - tf.constant()
            syntax:     `tf.constant(value, dtype=None, shape=None, name='Const')`
            Creates a constant tensor from a tensor-like object.       
                
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
        else:
            input_shape = inputs.shape
            rank = input_shape.rank
            if rank == 1:
                # --mb
                #   rank=1 means that the input shape is `(batch,)`
                #   so the dimension is expanded so that the shape becomes `(batch,1)`
                return array_ops.expand_dims_v2(inputs, axis=1)
            else:
                batch_dim = tensor_shape.dimension_value(input_shape[0])
                non_batch_dims = input_shape[1:]
                # reshape in a way that preserves as much shape info as possible.
                if non_batch_dims.is_fully_defined():
                    # --mb
                    # tf.TensorShape.is_fully_defined() returns true if the all the shape is fully defined
                    # for example a tensor of shape TensorShape([None, 256]) is partially defined
                    # TensorShape([16, 256]) is fully defined
                    # Example:
                    #   >>> s = tf.TensorShape([16,256])
                    #   >>> s
                    #    TensorShape([16, 256])
                    #   >>> s.is_fully_defined()
                    #   True
                    #   >>> tf.TensorShape([None, 256]).is_fully_defined()
                    #   False

                    # functools.reduce() explanation:
                    #   Apply function of two arguments cumulatively to the items of iterable,
                    #   from left to right, so as to reduce the iterable to a single value
                    #   Syntax: functools.reduce(function, iterable)
                    #   Example:
                    #       functools.reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])
                    #       calculates ((((1+2)+3)+4)+5).

                    # --mb
                    # returns the multiplied value of the non-batch dimensions
                    # For ex:  [2, 5, 2] => [20]
                    last_dim = int(functools.reduce(operator.mul, non_batch_dims))
                    # --mb
                    # >>> a.shape
                    # (2, 3)
                    # >>> last_dim = int(functools.reduce(operator.mul, a.shape))
                    # >>> last_dim
                    # 6
                    # >>> tf.constant([-1, last_dim])
                    # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([-1,  6], dtype=int32)>
                    flattened_shape = constant_op.constant([-1, last_dim])
                elif batch_dim is not None:
                    # --mb If batch_dim is defined
                    flattened_shape = constant_op.constant([int(batch_dim), -1])
                else:
                    flattened_shape = [array_ops.shape_v2(inputs)[0], -1]
                # --mb reshapes the inputs as per flattened_shape; -1 would be autocalculated
                return array_ops.reshape(inputs, flattened_shape)

    def compute_output_shape(self, input_shape):
        # !to-do
        input_shape = tensor_shape.as_shape(input_shape).as_list()
        if not input_shape:
            output_shape = tensor_shape.TensorShape([1])
        else:
            output_shape = [input_shape[0]]
        if np.all(input_shape[1:]):
            output_shape += [np.prod(input_shape[1:], dtype=int)]
        else:
            output_shape += [None]
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        # !to-do
        config = super(Flatten, self).get_config()
        config.update({'data_format': self.data_format})
        return config


@keras_export('keras.layers.Dense')
class Dense(Layer):
    # --mb !ND
    """
        Just your regular densely-connected NN layer.

        `Dense` implements the operation:
            `output = activation(dot(input, kernel) + bias)`
        where `activation` is the element-wise activation function
        passed as the `activation` argument, `kernel` is a weights matrix
        created by the layer, and `bias` is a bias vector created by the layer
        (only applicable if `use_bias` is `True`).

        Note: If the input to the layer has a rank greater than 2, then `Dense`
        computes the dot product between the `inputs` and the `kernel` along the
        last axis if the `inputs` and axis 1 of the `kernel` (using `tf.tensordot`).
            For example, if input has dimensions `(batch_size, d0, d1)`,
            then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates
            along axis 2 of the `input`, on every sub-tensor shape `(1, 1, d1)`
            (there are `batch_size * d0` such sub-tensors)
            The output in this case will have the shape `(batch_size, d0, units)`.


    """
    pass