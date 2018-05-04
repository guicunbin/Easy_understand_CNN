from test_data import *
from test_utils import *
import tensorflow as tf
import numpy as np
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess_config=tf.ConfigProto(gpu_options=gpu_options)
#top_tf = tf.nn.conv2d(input=tf.cast(bottom,tf.float32), filter=tf.cast(weight,tf.float32), strides=[1,1,5,5], padding="VALID", data_format="NCHW")

#bottom from [N, C_in, H_in, W_in] to [N, H_in, W_in, C_in]
bottom_np = np.array(bottom).reshape([1,3,16,16]).transpose([0,2,3,1]).astype(np.float32)
#weight from [C_out, C_in, K_h, k_w] to [K_h, K_w, C_in, C_out]
weight_np = np.array(weight).reshape([1,3,5, 5]).transpose([2,3,1,0]).astype(np.float32)
top_tf = tf.nn.conv2d(input=bottom_np, filter=weight_np, strides=[1,5,5,1], padding="VALID", data_format="NHWC")
sess = tf.Session(config = sess_config)
top_tf_np = sess.run(top_tf).transpose([0,3,1,2]).reshape([-1,])
print "tf ==>\n",top_tf_np
assert [1,2,3] == [1,2,3]
print "cpp =>\n",np.array(top);
assert np_array_equal(top_tf_np,np.array(top))
print "     \n test conv2d  success !!!!\n  "

