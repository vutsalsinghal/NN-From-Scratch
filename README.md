# NN-From-Scratch


Forward propagation:

- hidden_layer_input      = dot_prod(x,w_in)+b_in
- hidden_layer_activation = sigmoid(hidden_layer_activation)
- output_layer_input      = dot_prod(hidden_layer_activation,w_o)+b_out
- output                  = sigmoid(output_layer_input)

Backward Propagation:

- Error (E)          = y - output
- slope_output_layer = sigmoid_derivative(output)
- slope_hidden_layer = sigmoid_derivative(hidden_layer_activation)
- d_output           = E*slope_output_layer
- error_hidden_layer = sigmoid_derivative(d_output,w_o.T)
- d_hidden_layer     = error_hidden_layer*slope_hidden_layer

Weights and bias update:
w = w + d_w*lr
- w_o = w_o + dot_prod(hidden_layer_activation.T,d_output)*lr
- w_in = w_in + dot_prod(X.T,d_hidden_layer)*lr

- b_in  = b_in + sum(d_hidden_layer, axis=0)*lr
- b_out = b_out + sum(d_output, axis=0)*lr
