## Easy Questions (1-35)

**1. What is computer vision?**
A field of AI enabling computers to interpret and understand visual information from images and videos.

**2. What is an image in computer vision?**
A 2D or 3D array of pixels, where each pixel contains intensity values (grayscale) or color channel values (RGB).

**3. What is a pixel?**
The smallest unit of an image, representing a single point with intensity or color values.

**4. What are the three color channels in RGB images?**
Red, Green, and Blue channels, each typically with values from 0-255.

**5. What is grayscale image?**
An image with single channel representing intensity, with values typically from 0 (black) to 255 (white).

**6. What is image resolution?**
The dimensions of an image measured in pixels (width × height), determining image detail level.

**7. What is image classification?**
Assigning a single label to an entire image from predefined categories.

**8. What is object detection?**
Identifying and localizing multiple objects in an image with bounding boxes and class labels.

**9. What is semantic segmentation?**
Classifying each pixel in an image into predefined categories, without distinguishing individual object instances.

**10. What is instance segmentation?**
Identifying and segmenting each individual object instance in an image separately.

**11. What is a convolutional layer?**
A layer that applies learnable filters across an image using convolution operations to extract features.

**12. What is a filter/kernel in CNN?**
A small matrix (e.g., 3×3, 5×5) that slides over the image, performing element-wise multiplication and summation to detect features.

**13. What is stride in convolution?**
The number of pixels by which the filter moves. Stride=1 moves one pixel at a time; stride=2 skips every other pixel.

**14. What is padding?**
Adding extra pixels (usually zeros) around image borders to control output dimensions and preserve edge information.

**15. What is "valid" padding?**
No padding applied; output size is smaller than input: output_size = (input_size - kernel_size) / stride + 1.

**16. What is "same" padding?**
Padding added so output size equals input size (when stride=1).

**17. What is pooling?**
A downsampling operation that reduces spatial dimensions while retaining important features.

**18. What is max pooling?**
Selecting the maximum value within each pooling window, commonly used for spatial downsampling.

**19. What is average pooling?**
Computing the average value within each pooling window.

**20. What is a feature map?**
The output of applying a convolutional filter to an image or previous layer, representing detected features.

**21. What is a receptive field?**
The region of the input image that influences a particular neuron's activation.

**22. What is data augmentation in computer vision?**
Artificially expanding the training dataset by applying transformations (rotation, flipping, cropping, color jittering) to images.

**23. What is image normalization?**
Scaling pixel values to a standard range (e.g., [0,1] or [-1,1]) or standardizing to zero mean and unit variance.

**24. What is transfer learning in computer vision?**
Using a pre-trained model (trained on large datasets like ImageNet) as a starting point for new tasks.

**25. What is ImageNet?**
A large-scale dataset with ~14 million labeled images across 20,000+ categories, used for pre-training vision models.

**26. What is the ILSVRC competition?**
ImageNet Large Scale Visual Recognition Challenge - an annual competition that drove major CV breakthroughs (AlexNet, VGGNet, ResNet).

**27. What is a bounding box?**
A rectangle defined by coordinates (x, y, width, height) that encloses an object in an image.

**28. What is IoU (Intersection over Union)?**
A metric measuring bounding box overlap: IoU = Area of Overlap / Area of Union. Used to evaluate object detection.

**29. What is non-maximum suppression (NMS)?**
A technique to remove duplicate detections by keeping only the highest-confidence bounding box among overlapping boxes.

**30. What is image preprocessing?**
Operations applied to images before feeding to models (resizing, normalization, augmentation).

**31. What is edge detection?**
Identifying boundaries where pixel intensity changes sharply, revealing object contours and structures.

**32. What is the Sobel operator?**
A filter that computes image gradients to detect edges in horizontal and vertical directions.

**33. What is image thresholding?**
Converting grayscale images to binary by setting pixels above a threshold to white and below to black.

**34. What is histogram equalization?**
Adjusting image contrast by redistributing pixel intensity values to span the full range.

**35. What is color space?**
A mathematical model for representing colors (RGB, HSV, LAB, etc.), each with different properties.

## Medium Questions (36-70)

**36. What is AlexNet and its significance?**
A CNN that won ILSVRC 2012, featuring 8 layers, ReLU activation, dropout, and data augmentation. It demonstrated deep learning's superiority for image classification.

**37. What is VGGNet's architecture principle?**
Using small (3×3) convolutional filters stacked deeply (16-19 layers), showing that depth improves performance while keeping filters small.

**38. What is the 3×3 convolution advantage in VGGNet?**
Two 3×3 convolutions have the same receptive field as one 5×5 but with fewer parameters and more non-linearity (two ReLUs vs. one).

**39. What is GoogLeNet/Inception?**
A CNN using inception modules that apply multiple filter sizes (1×1, 3×3, 5×5) in parallel, capturing features at different scales efficiently.

**40. What is the purpose of 1×1 convolution?**
Dimensionality reduction, increasing non-linearity without spatial filtering, and cross-channel information mixing. Also called "network in network."

**41. What is ResNet (Residual Network)?**
A very deep CNN (50-152 layers) using skip connections (residual connections) that add layer inputs to outputs, enabling gradient flow and preventing degradation.

**42. Explain the residual block formula.**
H(x) = F(x) + x, where F(x) is the learned residual mapping (conv layers) and x is the identity. The network learns residuals rather than direct mappings.

**43. Why do skip connections help training deep networks?**
They create direct gradient paths, preventing vanishing gradients; allow identity mappings (easier to optimize); and enable feature reuse across layers.

**44. What is DenseNet?**
A CNN where each layer receives inputs from all previous layers, creating dense connectivity. This encourages feature reuse and improves gradient flow.

**45. What is the difference between ResNet and DenseNet connections?**
ResNet adds (summation) features from previous layers; DenseNet concatenates features, preserving information from all layers but increasing feature dimensions.

**46. What is MobileNet?**
A lightweight CNN using depthwise separable convolutions for efficient mobile and embedded deployment.

**47. What is depthwise separable convolution?**
Factorizing standard convolution into depthwise (each channel filtered separately) and pointwise (1×1 conv mixing channels), reducing parameters by ~8-9×.

**48. What is EfficientNet?**
A family of CNNs that uniformly scales depth, width, and resolution using compound scaling, achieving better accuracy-efficiency trade-offs.

**49. What is global average pooling?**
Averaging each feature map into a single value, reducing parameters compared to fully connected layers while maintaining spatial invariance.

**50. What is batch normalization in CNNs?**
Normalizing layer activations across the batch: BN(x) = γ((x-μ)/σ) + β, where μ and σ are batch statistics, γ and β are learnable parameters.

**51. What is the Region Proposal Network (RPN) in Faster R-CNN?**
A fully convolutional network that generates object proposals by predicting objectness scores and bounding box refinements at each spatial location.

**52. What is anchor box in object detection?**
Predefined boxes of various scales and aspect ratios used as references for predicting object locations.

**53. What is R-CNN (Region-based CNN)?**
An object detection method that generates region proposals (selective search), extracts features with CNN, and classifies each region.

**54. What is Fast R-CNN's improvement over R-CNN?**
Instead of running CNN on each proposal separately, it processes the entire image once and uses RoI pooling to extract features for each proposal.

**55. What is Faster R-CNN's improvement?**
Replaces selective search with learnable Region Proposal Network (RPN), making the entire pipeline end-to-end trainable and much faster.

**56. What is YOLO (You Only Look Once)?**
A single-stage object detection method that frames detection as regression, dividing images into grids and predicting bounding boxes and classes directly.

**57. What is the difference between one-stage and two-stage detectors?**
Two-stage (Faster R-CNN) generates proposals then classifies (slower, more accurate). One-stage (YOLO, SSD) predicts directly (faster, less accurate).

**58. What is SSD (Single Shot Detector)?**
A one-stage detector using multi-scale feature maps to detect objects at different scales, predicting classes and boxes at each location.

**59. What is Feature Pyramid Network (FPN)?**
An architecture creating multi-scale feature representations by combining low-resolution, semantically strong features with high-resolution, semantically weak features.

**60. What is focal loss?**
A loss function for addressing class imbalance: FL(p) = -(1-p)^γ log(p), down-weighting easy examples to focus on hard negatives. Used in RetinaNet.

**61. What is Mask R-CNN?**
An extension of Faster R-CNN adding a mask prediction branch for instance segmentation, predicting pixel-level masks for each detected object.

**62. What is RoI Align?**
An improvement over RoI pooling that uses bilinear interpolation to avoid quantization, preserving spatial accuracy for precise localization.

**63. What is U-Net?**
A CNN architecture for semantic segmentation with encoder-decoder structure and skip connections, widely used in medical image segmentation.

**64. What is the encoder-decoder architecture in segmentation?**
Encoder progressively downsamples to capture context; decoder progressively upsamples to generate pixel-wise predictions.

**65. What is FCN (Fully Convolutional Network)?**
A network using only convolutional layers (no fully connected), enabling dense predictions for segmentation with arbitrary input sizes.

**66. What is transposed convolution (deconvolution)?**
An upsampling operation that learns to increase spatial dimensions, used in decoder networks for segmentation and generation.

**67. What is atrous/dilated convolution?**
Convolution with gaps (dilation rate) between kernel elements, increasing receptive field without increasing parameters or reducing resolution.

**68. What is DeepLab?**
A semantic segmentation architecture using atrous convolution, Atrous Spatial Pyramid Pooling (ASPP), and conditional random fields for refined boundaries.

**69. What is ASPP (Atrous Spatial Pyramid Pooling)?**
A module applying parallel atrous convolutions with different rates to capture multi-scale context.

**70. What is mean Average Precision (mAP)?**
The primary metric for object detection, averaging precision across different IoU thresholds and classes.

## Hard Questions (71-100)

**71. Explain the mathematical formulation of standard convolution vs. depthwise separable convolution.**
Standard: Output(i,j,k) = Σ_c Σ_m Σ_n Input(i+m, j+n, c) × Kernel(m,n,c,k). Cost: H×W×C_in×C_out×K². Depthwise separable: First depthwise (each channel separately): Cost H×W×C×K², then pointwise (1×1): Cost H×W×C_in×C_out. Total: H×W×K²×C + H×W×C_in×C_out ≈ 1/9 for K=3.

**72. Derive the receptive field formula for stacked convolutions.**
For a single layer with kernel k and stride s: RF = k. For L layers: RF_L = RF_{L-1} + (k-1)×∏_{i=1}^{L-1}s_i. With stride 1: RF = 1 + Σ(k_i-1). Example: Three 3×3 layers give RF = 1+2+2+2 = 7×7, same as one 7×7 layer.

**73. What is Vision Transformer (ViT) and how does it work?**
ViT splits images into patches (16×16), linearly embeds them, adds positional encodings, and processes with standard transformer encoder. It treats image patches as tokens, applying self-attention to capture global dependencies.

**74. Explain the patch embedding process in ViT.**
Image (H×W×C) is divided into N patches of size P×P. Each patch is flattened to vector of dimension P²×C, then linearly projected to embedding dimension D. Result: (N, D) sequence of patch embeddings.

**75. Why does ViT require large-scale pre-training?**
Transformers lack inductive biases (locality, translation equivariance) that CNNs have. Without sufficient data, ViTs underperform. With large datasets (JFT-300M), they learn these properties and surpass CNNs.

**76. What is the difference between global and local attention in vision?**
Global attention (ViT) computes O(N²) complexity for N patches, capturing long-range dependencies but expensive. Local attention (Swin) restricts attention to windows, reducing to O(N), trading global receptive field for efficiency.

**77. Explain Swin Transformer's shifted window mechanism.**
Swin computes self-attention within non-overlapping windows (local). To enable cross-window connections, it shifts windows between layers, creating connections across spatial locations while maintaining efficiency.

**78. What is DETR (Detection Transformer)?**
An end-to-end object detection framework using transformers. CNN extracts features, transformer encoder-decoder processes them, and learnable object queries predict boxes and classes via bipartite matching loss.

**79. Explain the bipartite matching in DETR.**
DETR predicts a fixed set of N detections and uses Hungarian algorithm to find optimal one-to-one matching between predictions and ground truth, eliminating need for NMS and anchor boxes.

**80. What is contrastive learning in computer vision?**
Learning representations by pulling similar images (positive pairs) together and pushing dissimilar images (negative pairs) apart in embedding space. Examples: SimCLR, MoCo.

**81. Explain SimCLR's framework.**
Create two augmented views of each image, encode with shared CNN, project to latent space, and maximize agreement using NT-Xent loss: contrastive loss encourages embeddings of positive pairs to be similar while being dissimilar to negatives.

**82. What is MoCo (Momentum Contrast)?**
A contrastive learning framework using a momentum-updated encoder to maintain a large, consistent queue of negative samples, enabling large batch sizes without computational overhead.

**83. What is knowledge distillation in computer vision?**
Training a smaller student model to mimic a larger teacher model by matching soft probability distributions (with temperature): L = αL_CE + (1-α)L_KD, where L_KD = KL(Teacher||Student).

**84. Explain neural architecture search (NAS) in computer vision.**
Automatically discovering optimal CNN architectures by searching over operations (conv types, connections) using methods like reinforcement learning (NASNet), evolutionary algorithms, or differentiable search (DARTS).

**85. What is EfficientNet's compound scaling method?**
Uniformly scaling depth (d), width (w), and resolution (r) with fixed ratios: depth = α^φ, width = β^φ, resolution = γ^φ, where α·β²·γ² ≈ 2, balancing all dimensions for optimal accuracy-efficiency.

**86. Explain the receptive field vs. effective receptive field.**
Theoretical receptive field is the input region that can affect a neuron. Effective receptive field is smaller, with Gaussian distribution of actual influence, as central pixels contribute more than peripheral ones due to network structure.

**87. What is test-time augmentation (TTA)?**
Applying multiple augmentations to test images, getting predictions for each, and averaging results. Improves accuracy but increases inference time proportionally.

**88. What is domain adaptation in computer vision?**
Adapting models trained on source domain to target domain with different distributions. Methods include adversarial training (aligning feature distributions) or self-training on target data.

**89. Explain CycleGAN's cycle consistency loss.**
For unpaired image-to-image translation: L_cyc = ||F(G(x)) - x|| + ||G(F(y)) - y||, ensuring translating from X→Y→X reconstructs original, without requiring paired training data.

**90. What is StyleGAN's style-based generator?**
A GAN architecture where generator takes constant input and modulates features at each layer using learned style codes via AdaIN (Adaptive Instance Normalization), enabling fine-grained control over generated images.

**91. Explain AdaIN (Adaptive Instance Normalization) in StyleGAN.**
AdaIN(x, y) = σ(y) × ((x - μ(x))/σ(x)) + μ(y), where x is content features, y is style code. It normalizes content then scales/shifts using style, transferring style while preserving content structure.

**92. What is the perceptual loss in style transfer?**
Using pre-trained CNN (VGG) feature differences instead of pixel differences: L_perceptual = ||φ(generated) - φ(target)||², where φ extracts features. Captures semantic similarity better than pixel-wise loss.

**93. Explain multi-scale training in object detection.**
Training with images at different resolutions (e.g., 320-608 pixels), improving robustness to scale variation. The model learns to detect objects across scales, improving mAP significantly.

**94. What is Feature Pyramid Network's mathematical formulation?**
Bottom-up pathway creates semantically strong features C_i. Top-down pathway: M_i = Upsample(M_{i+1}) + Lateral(C_i), merging high-level semantics with high-resolution details through lateral connections.

**95. Explain Group Normalization and when it's better than Batch Normalization.**
GN divides channels into groups and normalizes within groups: GN(x) = γ((x-μ_g)/σ_g) + β. Unlike BN, it's independent of batch size, working better with small batches or in detection/segmentation tasks.

**96. What is Neural Radiance Fields (NeRF)?**
A method representing 3D scenes as continuous functions mapping 5D coordinates (x,y,z,θ,φ) to volume density and color. Trained on 2D images, it enables novel view synthesis through volumetric rendering.

**97. Explain the volume rendering equation in NeRF.**
Color C(r) = ∫T(t)σ(r(t))c(r(t),d)dt, where T(t) = exp(-∫σ(r(s))ds) is transmittance, σ is density, c is color. Approximated with stratified sampling along rays and integrated using quadrature.

**98. What is the lottery ticket hypothesis in computer vision?**
Pruned CNNs contain sparse "winning ticket" subnetworks that, when trained from their original initialization, match full network performance, suggesting initialization matters more than overparameterization during training.

**99. Explain soft attention vs. hard attention in vision.**
Soft attention computes weighted averages over all locations (differentiable, trainable with backprop). Hard attention samples discrete locations (non-differentiable, requires REINFORCE). Soft is more common due to easier optimization.

**100. What is neural architecture search efficiency and once-for-all networks?**
Once-for-all networks train a single supernet containing all possible sub-networks once, then extract optimal sub-networks for different hardware constraints without retraining, dramatically reducing NAS computational cost from thousands to single GPU-days.