## Easy Questions (1-35)

**1. What is deep learning?**
A subset of machine learning using neural networks with multiple layers to learn hierarchical representations of data.

**2. What is an artificial neural network?**
A computational model inspired by biological neurons, consisting of interconnected nodes organized in layers that process information.

**3. What are the three main types of layers in a neural network?**
Input layer, hidden layer(s), and output layer.

**4. What is a perceptron?**
The simplest neural network unit that takes weighted inputs, adds a bias, and applies an activation function to produce output.

**5. What is the purpose of an activation function?**
To introduce non-linearity into the network, enabling it to learn complex patterns.

**6. Name three common activation functions.**
ReLU (Rectified Linear Unit), Sigmoid, and Tanh (Hyperbolic Tangent).

**7. What is the ReLU activation function?**
f(x) = max(0, x) - outputs the input if positive, otherwise zero.

**8. What is the sigmoid activation function?**
f(x) = 1/(1 + e^(-x)) - outputs values between 0 and 1, often used for binary classification.

**9. What is forward propagation?**
The process of passing input data through the network layers to generate predictions.

**10. What is backpropagation?**
The algorithm for computing gradients of the loss function with respect to network weights by propagating errors backward.

**11. What is a weight in a neural network?**
A learnable parameter that determines the strength of connection between neurons.

**12. What is a bias in a neural network?**
A learnable parameter added to the weighted sum, allowing the activation function to shift left or right.

**13. What is an epoch in deep learning?**
One complete pass through the entire training dataset.

**14. What is a batch?**
A subset of training samples processed together in one forward/backward pass.

**15. What is the learning rate?**
A hyperparameter controlling the step size of parameter updates during optimization.

**16. What is a loss function?**
A function measuring the difference between predicted and actual values, guiding the learning process.

**17. What is cross-entropy loss?**
A loss function commonly used for classification tasks, measuring the difference between predicted and true probability distributions.

**18. What is mean squared error (MSE)?**
A loss function for regression tasks that averages the squared differences between predictions and targets.

**19. What is gradient descent?**
An optimization algorithm that iteratively updates parameters in the direction that minimizes the loss function.

**20. What is overfitting in deep learning?**
When a model learns training data too well, including noise, and performs poorly on new data.

**21. What is dropout?**
A regularization technique that randomly deactivates a fraction of neurons during training.

**22. What is the purpose of validation data?**
To evaluate model performance during training and tune hyperparameters without using test data.

**23. What is a convolutional neural network (CNN)?**
A neural network architecture specialized for processing grid-structured data like images.

**24. What is a recurrent neural network (RNN)?**
A neural network with loops that processes sequential data by maintaining memory of previous inputs.

**25. What is a fully connected layer?**
A layer where every neuron is connected to every neuron in the previous layer.

**26. What is the vanishing gradient problem?**
When gradients become extremely small during backpropagation, preventing early layers from learning effectively.

**27. What is the exploding gradient problem?**
When gradients become extremely large during backpropagation, causing unstable training.

**28. What is batch normalization?**
A technique that normalizes layer inputs to stabilize and accelerate training.

**29. What is transfer learning?**
Using a pre-trained model on a new task, leveraging learned features to reduce training time and data requirements.

**30. What is fine-tuning?**
Adjusting the weights of a pre-trained model on a new dataset for a specific task.

**31. What is data augmentation?**
Artificially increasing dataset size by applying transformations to existing data (rotation, flipping, scaling).

**32. What is early stopping?**
Halting training when validation performance stops improving to prevent overfitting.

**33. What is a hyperparameter?**
A configuration setting that is not learned from data (e.g., learning rate, number of layers).

**34. What is the difference between parameters and hyperparameters?**
Parameters are learned during training (weights, biases), while hyperparameters are set before training (learning rate, architecture).

**35. What is GPU acceleration in deep learning?**
Using Graphics Processing Units to parallelize matrix operations, dramatically speeding up training and inference.

## Medium Questions (36-65)

**36. Explain the difference between batch, mini-batch, and stochastic gradient descent.**
Batch GD uses all samples per update (slow, stable), SGD uses one sample (fast, noisy), mini-batch uses a subset (balanced trade-off).

**37. What is momentum in gradient descent?**
An optimization technique that accumulates a moving average of past gradients to accelerate convergence and reduce oscillations.

**38. What is the Adam optimizer?**
Adaptive Moment Estimation - combines momentum and adaptive learning rates, maintaining per-parameter learning rates based on first and second moment estimates.

**39. What is weight initialization and why is it important?**
Setting initial weight values before training. Poor initialization can cause vanishing/exploding gradients or slow convergence. Methods include Xavier/Glorot and He initialization.

**40. What is Xavier/Glorot initialization?**
A weight initialization method that draws weights from a distribution with variance proportional to 1/(n_in + n_out), helping maintain gradient flow.

**41. What is He initialization?**
Weight initialization for ReLU networks, drawing from distribution with variance proportional to 2/n_in, accounting for ReLU's zeroing of negative values.

**42. What is a convolutional layer?**
A layer that applies learnable filters across input using convolution operations, detecting local patterns like edges or textures.

**43. What are filters/kernels in CNNs?**
Small matrices that slide over input data, performing element-wise multiplication and summation to extract features.

**44. What is stride in convolution?**
The number of pixels by which the filter moves across the input. Larger strides reduce output dimensions.

**45. What is padding in CNNs?**
Adding extra pixels (usually zeros) around input borders to control output dimensions and preserve edge information.

**46. What is pooling in CNNs?**
A downsampling operation that reduces spatial dimensions while retaining important features.

**47. What is max pooling?**
A pooling operation that selects the maximum value within each pooling window.

**48. What is average pooling?**
A pooling operation that computes the average value within each pooling window.

**49. What is a receptive field?**
The region of input that influences a particular neuron's activation, growing larger in deeper layers.

**50. What is the purpose of the flatten layer?**
To convert multi-dimensional feature maps into a 1D vector for input to fully connected layers.

**51. What is ResNet and what problem does it solve?**
Residual Network introduces skip connections that allow gradients to flow directly through layers, enabling training of very deep networks (100+ layers).

**52. What are skip connections/residual connections?**
Direct pathways that add layer inputs to outputs (identity mapping), helping gradients flow and enabling deeper networks.

**53. What is VGGNet?**
A CNN architecture using small (3×3) convolutional filters stacked deeply, demonstrating that depth improves performance.

**54. What is Inception/GoogLeNet?**
A CNN architecture using inception modules that apply multiple filter sizes in parallel, capturing features at different scales efficiently.

**55. What is depthwise separable convolution?**
A efficient convolution factorized into depthwise (spatial) and pointwise (channel) convolutions, reducing parameters and computation.

**56. What is 1×1 convolution?**
A convolution with 1×1 kernel used for dimensionality reduction, increasing non-linearity, or mixing channel information.

**57. What is global average pooling?**
Averaging each feature map into a single value, reducing parameters and serving as alternative to fully connected layers.

**58. What is an LSTM network?**
Long Short-Term Memory - an RNN variant with gates (forget, input, output) that control information flow, addressing vanishing gradients in long sequences.

**59. What is the forget gate in LSTM?**
A gate that decides what information to discard from the cell state, using sigmoid activation to output values between 0 and 1.

**60. What is the input gate in LSTM?**
A gate that decides what new information to store in the cell state, combining sigmoid and tanh activations.

**61. What is the output gate in LSTM?**
A gate that decides what information from the cell state to output, filtering the cell state through tanh and sigmoid.

**62. What is GRU (Gated Recurrent Unit)?**
A simpler RNN variant than LSTM with only reset and update gates, offering similar performance with fewer parameters.

**63. What is sequence-to-sequence (Seq2Seq) model?**
An architecture with encoder-decoder structure that maps input sequences to output sequences, used in translation and generation.

**64. What is an autoencoder?**
A neural network trained to compress input into a latent representation (encode) and reconstruct it (decode), learning efficient encodings.

**65. What is the bottleneck in an autoencoder?**
The middle layer with fewer neurons that creates a compressed representation, forcing the network to learn essential features.

## Hard Questions (66-100)

**66. Explain the mathematical formulation of backpropagation through time (BPTT).**
BPTT unfolds RNNs across time steps and applies backpropagation. For hidden state h_t = f(W_h·h_{t-1} + W_x·x_t), gradients flow: ∂L/∂h_t = ∂L/∂h_{t+1} × ∂h_{t+1}/∂h_t + ∂L/∂y_t, involving repeated matrix multiplications that cause vanishing/exploding gradients.

**67. Derive the gradient update rule for LSTM forget gate.**
For forget gate f_t = σ(W_f·[h_{t-1}, x_t] + b_f), gradient is: ∂L/∂W_f = Σ_t (∂L/∂f_t × ∂f_t/∂W_f) = Σ_t (∂L/∂c_t × c_{t-1} × f_t(1-f_t) × [h_{t-1}, x_t]^T), where c_t is cell state.

**68. What is the attention mechanism and its mathematical formulation?**
Attention computes weighted sum of values based on query-key similarity: Attention(Q,K,V) = softmax(QK^T/√d_k)V, where Q=queries, K=keys, V=values, d_k is key dimension. This allows focusing on relevant parts of input.

**69. Explain multi-head attention in transformers.**
Multi-head attention runs multiple attention mechanisms in parallel with different learned projections: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O, where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V). This captures different representation subspaces.

**70. What is the transformer architecture?**
An architecture using only attention mechanisms (no recurrence/convolution). It consists of encoder-decoder stacks with multi-head self-attention, position-wise feedforward networks, and positional encodings.

**71. Why do transformers need positional encodings?**
Since transformers have no recurrence or convolution, they can't inherently capture sequence order. Positional encodings add position information using sine/cosine functions: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d)).

**72. What is self-attention?**
Attention where queries, keys, and values all come from the same sequence, allowing each position to attend to all positions and capture dependencies.

**73. Explain the concept of teacher forcing.**
A training technique for sequence models where true previous outputs (not predictions) are fed as inputs at each time step, accelerating training but potentially causing train-test mismatch.

**74. What is scheduled sampling?**
A curriculum learning approach that gradually transitions from teacher forcing to using model predictions during training, reducing exposure bias.

**75. What is the exposure bias problem?**
The mismatch between training (using ground truth) and inference (using predictions) in sequence models, where errors compound during generation.

**76. What is beam search?**
A heuristic search algorithm that explores multiple hypotheses simultaneously, keeping the top-k candidates at each step, used for sequence generation.

**77. What is the difference between greedy decoding and beam search?**
Greedy decoding selects the highest probability token at each step (fast, suboptimal). Beam search maintains multiple hypotheses (slower, better quality).

**78. What is a variational autoencoder (VAE)?**
A generative model that encodes inputs into a probability distribution (typically Gaussian) in latent space, enabling sampling and generation. Loss = reconstruction + KL divergence.

**79. Explain the reparameterization trick in VAEs.**
To enable gradient flow through stochastic sampling, instead of z ~ N(μ,σ²), we compute z = μ + σ⊙ε where ε ~ N(0,I), making randomness external to the computation graph.

**80. What is a Generative Adversarial Network (GAN)?**
A framework with generator G creating fake samples and discriminator D distinguishing real from fake, trained adversarially: min_G max_D E[log D(x)] + E[log(1-D(G(z)))].

**81. Explain mode collapse in GANs.**
When the generator learns to produce limited varieties of samples that fool the discriminator, failing to capture full data distribution diversity.

**82. What is Wasserstein GAN (WGAN)?**
A GAN variant using Wasserstein distance instead of JS divergence, providing more stable training and meaningful loss curves. Requires weight clipping or gradient penalty.

**83. What is the gradient penalty in WGAN-GP?**
A soft constraint enforcing the Lipschitz condition: λE[(||∇_x̂ D(x̂)||_2 - 1)²], where x̂ = εx + (1-ε)G(z), replacing hard weight clipping for more stable training.

**84. What is StyleGAN?**
A GAN architecture that controls image generation at different scales using adaptive instance normalization (AdaIN), enabling fine-grained style control and high-quality image synthesis.

**85. What is neural style transfer?**
A technique that combines content from one image with style from another by optimizing to match content features (from deep layers) and style features (Gram matrices from multiple layers).

**86. Explain the Gram matrix in style transfer.**
The Gram matrix G^l_{ij} = Σ_k F^l_{ik}F^l_{jk} captures feature correlations at layer l, representing texture/style information independent of spatial structure.

**87. What is distillation in deep learning?**
Training a smaller "student" model to mimic a larger "teacher" model by matching soft probability distributions (with temperature), transferring knowledge efficiently.

**88. What is knowledge distillation temperature?**
A softening parameter T in softmax: p_i = exp(z_i/T)/Σ_j exp(z_j/T). Higher T produces softer distributions, revealing teacher's uncertainty for better student learning.

**89. What is pruning in neural networks?**
Removing unnecessary weights or neurons based on magnitude, gradients, or importance scores to reduce model size while maintaining performance.

**90. What is quantization in deep learning?**
Reducing numerical precision of weights/activations (e.g., FP32 to INT8), decreasing memory and computation while accepting minor accuracy loss.

**91. What is Neural Architecture Search (NAS)?**
Automatically discovering optimal network architectures using search algorithms (RL, evolutionary, gradient-based) over a defined search space.

**92. Explain differentiable NAS (DARTS).**
A continuous relaxation of architecture search where operations are weighted combinations, enabling gradient-based optimization: α* = argmin L_val(w*(α), α), where α are architecture parameters.

**93. What is the lottery ticket hypothesis?**
Dense networks contain sparse subnetworks ("winning tickets") that, when trained from their original initialization, can match full network performance, suggesting initialization matters more than overparameterization.

**94. What is network morphism?**
Techniques for modifying network architecture while preserving functionality, enabling efficient architecture search by starting from working networks.

**95. Explain capsule networks.**
Networks using capsules (groups of neurons representing entity properties like pose, texture) with dynamic routing, better preserving spatial hierarchies than CNNs' pooling.

**96. What is dynamic routing by agreement in capsule networks?**
An iterative process where lower-level capsules route output to higher-level capsules based on agreement, measured by scalar product: c_ij = softmax(b_ij), where b_ij accumulates agreement scores.

**97. What is neural ordinary differential equations (Neural ODEs)?**
Parameterizing continuous transformations as ODEs: dh(t)/dt = f(h(t), t, θ), enabling continuous-depth models, constant memory backpropagation (adjoint method), and irregular time series modeling.

**98. What is the adjoint sensitivity method in Neural ODEs?**
Computing gradients without storing intermediate states by solving a second ODE backward in time: da/dt = -a^T ∂f/∂h, where a is adjoint state, reducing memory from O(L) to O(1).

**99. What is meta-learning in deep learning?**
Learning to learn - training models that quickly adapt to new tasks with few examples by learning good initializations (MAML), optimizers, or embeddings across task distributions.

**100. Explain Model-Agnostic Meta-Learning (MAML).**
An algorithm finding parameter initialization θ that enables fast adaptation to new tasks: θ* = argmin_θ Σ_T L_T(θ - α∇L_T(θ)), optimizing for post-adaptation performance across tasks through second-order gradients.

---

These questions span fundamental concepts to cutting-edge deep learning research, covering architectures, optimization, generative models, and advanced techniques!