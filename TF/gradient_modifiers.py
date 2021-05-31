import tensorflow as tf


"""
## To be added to a model's `train_step` as follows:
def train_step(self, data): # data should be the dataloader?
    # Unpack data
    x, y = data # ys are one-hot encoded 
    
    with tf.GradientTape() as tape:
        # using sparse categorical cross-entropy
        #y_pred = self(x, training=True)
        y_pred = self.model(x, training=True)
        
        # Compute losses
        #loss = self.loss_fn(y, y_pred)
        loss = self.compiled_loss(y, y_pred)
        
    # Compute gradients
    trainable_params = self.trainable_variables
    gradients = tape.gradient(loss, trainable_params) # accessing the gradients
    
    # access gradients and apply adaptive gradient clipping if desired
    if self.use_agc:
        gradients = adaptive_clip_grad(trainable_params, gradients, 
                    clip_factor=self.clip_factor, eps=self.eps)
    
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_params))    
"""


def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5

def unitwise_norm(x):
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2,]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor=0.01,
                       eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads