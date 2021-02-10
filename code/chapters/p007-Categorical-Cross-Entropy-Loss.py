"""
Calculating the loss with Categorical Cross Entropy
Associated with YT NNFS tutorial: https://www.youtube.com/watch?v=dEXPMQXoiLc
"""

import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

print(-math.log(0.7))
print(-math.log(0.5))

# BUT if we had this instead
softmax_output = [0, 0.3, 0.7]
target_output = [1, 0, 0]
# where the actual target was predicted all wrong (0%),
# run running the -math.log(0) would raise an error
# therefore we clip the values so they are always non-zero