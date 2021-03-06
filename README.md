# SeparableConvGRU2D
Simple implementation of SeparableConvGRU2D layer in tensorflow keras. 

ConvGRU is useful, but parameter size is not small. 
For more compact model implementation, SeparableConvGRU is suitable.

This module works correctly in my personal project (2D object detection).
But, this does not mean "perfect"...

## Base code

https://github.com/KoertS/ConvGRU2D

## My environment

- ubuntu 18.04
- python 3.8
- tensorflow 2.4.1

## Example

```python
import tensorflow as tf
from ConvGRU2D import ConvGRU2D
from SeparableConvGRU2D import SeparableConvGRU2D

steps = 10
height = 32
width = 32
input_channels = 3
output_channels = 6

inputs = tf.keras.Input(shape=(steps, height, width, input_channels))

# ConvGRU
outputs_convgru = ConvGRU2D(filters=output_channels, kernel_size=3)(inputs)
model_convgru = tf.keras.Model(inputs=inputs, outputs=outputs_convgru, name="convgru_model")
model_convgru.summary()

# SeparableConvGRU
outputs_sepconvgru = SeparableConvGRU2D(filters=output_channels, kernel_size=3)(inputs)
model_sepconvgru = tf.keras.Model(inputs=inputs, outputs=outputs_sepconvgru, name="sepconvgru_model")
model_sepconvgru.summary()
```

### parameter size

- ConvGRU : 1,476
- SeparableConvGRU : 423
