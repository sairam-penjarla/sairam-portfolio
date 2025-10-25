## Easy Questions (1-40)

**1. What is Machine Learning?**
Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

**2. What are the three main types of machine learning?**
Supervised learning, unsupervised learning, and reinforcement learning.

**3. What is supervised learning?**
Learning where the model is trained on labeled data (input-output pairs) to predict outputs for new inputs.

**4. What is unsupervised learning?**
Learning where the model finds patterns in unlabeled data without predefined outputs.

**5. What is a training set?**
The portion of data used to train a machine learning model.

**6. What is a test set?**
The portion of data used to evaluate a trained model's performance on unseen data.

**7. What is overfitting?**
When a model learns the training data too well, including noise, and performs poorly on new data.

**8. What is underfitting?**
When a model is too simple to capture the underlying patterns in the data.

**9. What is a feature in machine learning?**
An individual measurable property or characteristic of the data used as input to a model.

**10. What is a label?**
The output or target variable that the model tries to predict in supervised learning.

**11. What is regression?**
A supervised learning task that predicts continuous numerical values.

**12. What is classification?**
A supervised learning task that predicts discrete categories or classes.

**13. What is clustering?**
An unsupervised learning technique that groups similar data points together.

**14. Name a common regression algorithm.**
Linear Regression.

**15. Name a common classification algorithm.**
Logistic Regression, Decision Trees, or Naive Bayes.

**16. What is the purpose of a validation set?**
To tune hyperparameters and evaluate model performance during training without using the test set.

**17. What is accuracy in classification?**
The percentage of correct predictions out of all predictions made.

**18. What is Mean Squared Error (MSE)?**
The average of squared differences between predicted and actual values, commonly used in regression.

**19. What is a confusion matrix?**
A table showing true positives, true negatives, false positives, and false negatives for classification.

**20. What is precision?**
The ratio of true positives to all predicted positives (TP / (TP + FP)).

**21. What is recall (sensitivity)?**
The ratio of true positives to all actual positives (TP / (TP + FN)).

**22. What is the F1 score?**
The harmonic mean of precision and recall, providing a balanced metric.

**23. What is a decision tree?**
A tree-like model that makes decisions by splitting data based on feature values.

**24. What is K-Nearest Neighbors (KNN)?**
An algorithm that classifies data points based on the majority class of their k nearest neighbors.

**25. What is a neural network?**
A model inspired by biological neurons, consisting of interconnected layers of nodes.

**26. What is an activation function?**
A function that introduces non-linearity into neural networks (e.g., ReLU, sigmoid, tanh).

**27. What is gradient descent?**
An optimization algorithm that iteratively adjusts parameters to minimize the loss function.

**28. What is a loss function?**
A function that measures how well a model's predictions match the actual values.

**29. What is batch size in training?**
The number of training examples used in one iteration of gradient descent.

**30. What is an epoch?**
One complete pass through the entire training dataset.

**31. What is the learning rate?**
A hyperparameter that controls how much to adjust model parameters during optimization.

**32. What is normalization?**
Scaling features to a similar range, typically [0, 1] or with mean 0 and standard deviation 1.

**33. What is the bias-variance tradeoff?**
The balance between a model's ability to minimize bias (error from incorrect assumptions) and variance (error from sensitivity to training data).

**34. What is cross-validation?**
A technique that divides data into multiple folds to evaluate model performance more reliably.

**35. What is k-fold cross-validation?**
Splitting data into k subsets, training on k-1 folds and validating on the remaining fold, repeated k times.

**36. What is feature engineering?**
The process of creating, selecting, or transforming features to improve model performance.

**37. What is dimensionality reduction?**
Reducing the number of features while preserving important information.

**38. What is PCA (Principal Component Analysis)?**
A dimensionality reduction technique that transforms data into uncorrelated principal components.

**39. What is ensemble learning?**
Combining multiple models to improve overall performance.

**40. What is bagging?**
An ensemble method that trains multiple models on different subsets of data and averages predictions.

## Medium Questions (41-70)

**41. Explain the difference between L1 and L2 regularization.**
L1 (Lasso) adds absolute values of coefficients to the loss function, promoting sparsity. L2 (Ridge) adds squared coefficients, shrinking all weights but not to zero.

**42. What is the vanishing gradient problem?**
When gradients become extremely small during backpropagation in deep networks, preventing early layers from learning effectively.

**43. What is dropout in neural networks?**
A regularization technique that randomly deactivates neurons during training to prevent overfitting.

**44. What is batch normalization?**
A technique that normalizes layer inputs to stabilize and accelerate training in neural networks.

**45. What is the ROC curve?**
A graph showing the tradeoff between true positive rate and false positive rate at various classification thresholds.

**46. What is AUC (Area Under Curve)?**
The area under the ROC curve, measuring a classifier's ability to distinguish between classes (0.5 = random, 1.0 = perfect).

**47. What is the difference between parametric and non-parametric models?**
Parametric models have a fixed number of parameters (e.g., linear regression), while non-parametric models' complexity grows with data (e.g., KNN).

**48. What is the kernel trick in SVM?**
A method to implicitly map data to higher dimensions without explicitly computing the transformation, enabling non-linear classification.

**49. What are support vectors in SVM?**
Data points closest to the decision boundary that define the maximum margin hyperplane.

**50. What is random forest?**
An ensemble of decision trees trained on random subsets of data and features, with predictions averaged or voted on.

**51. What is boosting?**
An ensemble method that sequentially trains models, each focusing on correcting errors of previous models.

**52. What is the difference between bagging and boosting?**
Bagging trains models in parallel independently, while boosting trains models sequentially with each correcting previous errors.

**53. What is XGBoost?**
An optimized gradient boosting algorithm known for speed, performance, and handling missing values.

**54. What is feature selection?**
The process of choosing the most relevant features to reduce dimensionality and improve model performance.

**55. What is the curse of dimensionality?**
As dimensionality increases, data becomes sparse and distance metrics become less meaningful, making learning difficult.

**56. What is transfer learning?**
Using a pre-trained model on a new but related task, leveraging learned features to improve performance with less data.

**57. What is data augmentation?**
Artificially expanding the training dataset by applying transformations (rotation, flipping, etc.) to existing data.

**58. What is the softmax function?**
An activation function that converts raw scores into probabilities that sum to 1, used in multi-class classification.

**59. What is backpropagation?**
The algorithm for computing gradients of the loss function with respect to network weights using the chain rule.

**60. What is the difference between stochastic, batch, and mini-batch gradient descent?**
Stochastic uses one sample per update, batch uses all samples, mini-batch uses a subset, balancing speed and stability.

**61. What is early stopping?**
A regularization technique that stops training when validation performance stops improving to prevent overfitting.

**62. What is a convolutional neural network (CNN)?**
A neural network with convolutional layers designed for processing grid-like data such as images.

**63. What is pooling in CNNs?**
A downsampling operation that reduces spatial dimensions while retaining important features (e.g., max pooling, average pooling).

**64. What is a recurrent neural network (RNN)?**
A neural network with loops that process sequential data by maintaining hidden states across time steps.

**65. What is the exploding gradient problem?**
When gradients become extremely large during training, causing unstable updates and divergence.

**66. What is LSTM (Long Short-Term Memory)?**
An RNN architecture with gates that control information flow, addressing vanishing gradient problems in long sequences.

**67. What is GRU (Gated Recurrent Unit)?**
A simplified version of LSTM with fewer gates, offering similar performance with less complexity.

**68. What is attention mechanism?**
A technique that allows models to focus on relevant parts of input when making predictions, weighing their importance.

**69. What is the transformer architecture?**
A neural network architecture based entirely on attention mechanisms, used in models like BERT and GPT.

**70. What is one-hot encoding?**
Representing categorical variables as binary vectors where only one element is 1 and others are 0.

## Hard Questions (71-100)

**71. Explain the mathematics behind backpropagation in detail.**
Backpropagation uses the chain rule to compute partial derivatives of the loss with respect to each weight. Starting from the output layer, gradients flow backward: ∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w, where L is loss, a is activation, z is weighted input, and w is weight.

**72. What is the bias-variance decomposition formula?**
Expected prediction error = Bias² + Variance + Irreducible Error. Bias is error from wrong assumptions, variance is error from sensitivity to training data, and irreducible error is inherent noise.

**73. Derive the normal equation for linear regression.**
Minimizing (y - Xβ)ᵀ(y - Xβ), taking derivative and setting to zero: -2Xᵀ(y - Xβ) = 0, solving gives β = (XᵀX)⁻¹Xᵀy.

**74. What is the EM (Expectation-Maximization) algorithm?**
An iterative algorithm for finding maximum likelihood estimates in models with latent variables. E-step computes expected values of latent variables, M-step maximizes likelihood given these expectations.

**75. Explain how t-SNE works.**
t-SNE (t-Distributed Stochastic Neighbor Embedding) maps high-dimensional data to low dimensions by preserving pairwise similarities. It uses Gaussian distributions in high-dimensional space and Student's t-distribution in low-dimensional space to avoid crowding.

**76. What is variational autoencoder (VAE)?**
A generative model that learns a probabilistic latent representation. It encodes data into a distribution (typically Gaussian), samples from it, and decodes back to reconstruct the input, trained with reconstruction loss plus KL divergence regularization.

**77. What is Generative Adversarial Network (GAN)?**
A framework with two networks: a generator creates fake samples and a discriminator distinguishes real from fake. They're trained adversarially, with the generator trying to fool the discriminator.

**78. Explain the concept of information gain in decision trees.**
Information gain measures the reduction in entropy after splitting on a feature: IG = H(parent) - Σ(p_i × H(child_i)), where H is entropy and p_i is the proportion of samples in each child node.

**79. What is the difference between discriminative and generative models?**
Discriminative models learn P(Y|X), the decision boundary directly (e.g., logistic regression). Generative models learn P(X|Y) and P(Y), modeling data distribution (e.g., Naive Bayes).

**80. Explain the mathematical foundation of SVM optimization.**
SVM finds the maximum margin hyperplane by solving: minimize ½||w||² subject to y_i(w·x_i + b) ≥ 1. This quadratic programming problem can be solved using Lagrange multipliers and the dual formulation.

**81. What is contrastive learning?**
A self-supervised learning approach that learns representations by pulling similar samples together and pushing dissimilar samples apart in embedding space.

**82. Explain batch normalization's effect on the optimization landscape.**
Batch normalization smooths the loss surface, reducing sensitivity to hyperparameters and allowing higher learning rates. It also reduces internal covariate shift, making optimization more stable.

**83. What is the difference between model-based and model-free reinforcement learning?**
Model-based RL learns a model of environment dynamics and plans using it. Model-free RL learns value functions or policies directly from experience without modeling transitions.

**84. Explain the bias in different sampling techniques.**
Convenience sampling has selection bias. Stratified sampling reduces variance but requires knowing population strata. Importance sampling can correct for different distributions but has high variance if weights are poorly chosen.

**85. What is catastrophic forgetting in neural networks?**
When a neural network trained sequentially on multiple tasks forgets previously learned tasks upon learning new ones, due to weight adjustments that overwrite previous knowledge.

**86. Explain the reparameterization trick in VAEs.**
To backpropagate through stochastic nodes, instead of sampling z ~ N(μ, σ²), we sample ε ~ N(0,1) and compute z = μ + σε. This makes the randomness external to the function, enabling gradient flow.

**87. What is meta-learning?**
Learning to learn - training models that can quickly adapt to new tasks with limited data by learning good initialization, optimization strategies, or model architectures from multiple related tasks.

**88. Explain the cold start problem in recommender systems.**
The difficulty in making recommendations for new users (no history) or new items (no ratings). Solutions include content-based filtering, demographic information, or transfer learning from similar users/items.

**89. What is the difference between implicit and explicit regularization?**
Explicit regularization directly adds penalty terms to loss (L1, L2). Implicit regularization emerges from training procedures themselves (early stopping, dropout, SGD noise, architecture constraints).

**90. Explain how BERT's masked language modeling works.**
BERT randomly masks tokens in input sequences and trains to predict them using bidirectional context. Unlike traditional language models that predict next tokens unidirectionally, BERT uses full context from both directions.

**91. What is curriculum learning?**
Training strategy where models learn from easier examples first, progressively increasing difficulty. This mimics human learning and can improve convergence and final performance on complex tasks.

**92. Explain the concept of neural architecture search (NAS).**
Automatically discovering optimal neural network architectures using search algorithms (evolutionary, reinforcement learning, gradient-based) rather than manual design, optimizing for performance and efficiency.

**93. What is causal inference and how does it differ from correlation?**
Causal inference establishes cause-effect relationships, not just associations. It requires interventions, counterfactual reasoning, or careful experimental design (RCTs, instrumental variables, propensity scoring) to control confounding.

**94. Explain how gradient clipping prevents exploding gradients.**
Gradient clipping rescales gradients if their norm exceeds a threshold: g = g × threshold/||g|| when ||g|| > threshold. This prevents parameter updates from being too large while maintaining direction.

**95. What is adversarial training?**
Training models on adversarially perturbed examples (small input changes that fool the model) to improve robustness. The loss includes both clean and adversarial examples: L = L_clean + λL_adversarial.

**96. Explain the lottery ticket hypothesis.**
Dense neural networks contain sparse subnetworks ("winning tickets") that, when trained in isolation from proper initialization, can match the full network's performance, suggesting network overparameterization and initialization importance.

**97. What is self-supervised learning and give examples?**
Learning representations from unlabeled data by creating pretext tasks from the data itself. Examples: predicting image rotations, masked token prediction (BERT), contrastive learning (SimCLR), jigsaw puzzles.

**98. Explain the difference between online and offline reinforcement learning.**
Online RL learns by interacting with the environment directly, collecting new data during training. Offline RL learns from a fixed dataset without environment interaction, addressing safety and cost concerns.

**99. What is neural ordinary differential equations (Neural ODEs)?**
A framework treating neural networks as continuous transformations defined by ODEs: dh/dt = f(h(t), t, θ). This enables continuous-depth networks, memory-efficient training, and modeling irregular time series.

**100. Explain multi-task learning and its theoretical benefits.**
Training a single model on multiple related tasks simultaneously. Benefits include: shared representations learning common features, implicit data augmentation, regularization through task diversity (preventing overfitting to any single task), and improved sample efficiency.