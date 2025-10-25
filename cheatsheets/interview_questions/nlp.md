## Easy Questions (1-35)

**1. What is Natural Language Processing (NLP)?**
A field of AI focused on enabling computers to understand, interpret, and generate human language.

**2. What is tokenization?**
The process of breaking text into smaller units called tokens (words, subwords, or characters).

**3. What is a corpus?**
A large collection of text documents used for training and evaluating NLP models.

**4. What is stemming?**
Reducing words to their root form by removing suffixes (e.g., "running" → "run"), often using rule-based approaches.

**5. What is lemmatization?**
Reducing words to their base dictionary form (lemma) using vocabulary and morphological analysis (e.g., "better" → "good").

**6. What is the difference between stemming and lemmatization?**
Stemming uses rules and may produce non-words; lemmatization uses dictionaries and produces valid words, being more accurate but slower.

**7. What are stop words?**
Common words (e.g., "the", "is", "and") often removed during preprocessing as they carry little semantic meaning.

**8. What is a vocabulary in NLP?**
The set of unique tokens that a model recognizes and can process.

**9. What is a bag-of-words (BoW) model?**
A text representation that counts word occurrences, ignoring grammar and word order.

**10. What is TF-IDF?**
Term Frequency-Inverse Document Frequency - a statistic measuring word importance by balancing frequency in a document against rarity across documents.

**11. What is the TF-IDF formula?**
TF-IDF(t,d) = TF(t,d) × IDF(t), where TF is term frequency and IDF(t) = log(N/df(t)), N is total documents, df(t) is documents containing term t.

**12. What is word embedding?**
Dense vector representations of words that capture semantic relationships in continuous space.

**13. What is Word2Vec?**
A technique for learning word embeddings using shallow neural networks with two architectures: CBOW and Skip-gram.

**14. What is the difference between CBOW and Skip-gram?**
CBOW predicts a target word from context words; Skip-gram predicts context words from a target word.

**15. What is GloVe?**
Global Vectors for Word Representation - learns embeddings by factorizing word co-occurrence matrices.

**16. What is cosine similarity?**
A metric measuring similarity between two vectors as the cosine of the angle between them: cos(θ) = (A·B)/(||A|| ||B||).

**17. What is named entity recognition (NER)?**
Identifying and classifying named entities (people, organizations, locations, dates) in text.

**18. What is part-of-speech (POS) tagging?**
Assigning grammatical categories (noun, verb, adjective) to each word in text.

**19. What is sentiment analysis?**
Determining the emotional tone or opinion expressed in text (positive, negative, neutral).

**20. What is text classification?**
Categorizing text into predefined classes or categories.

**21. What is a language model?**
A model that learns the probability distribution of word sequences, predicting the next word given previous words.

**22. What is perplexity?**
A metric measuring how well a language model predicts text, calculated as the exponential of average negative log-likelihood. Lower is better.

**23. What is n-gram?**
A contiguous sequence of n items (words/characters) from text. E.g., bigram (n=2), trigram (n=3).

**24. What is an n-gram language model?**
A model predicting the next word based on the previous n-1 words using probability: P(w_i|w_{i-n+1}...w_{i-1}).

**25. What is sequence-to-sequence (Seq2Seq) modeling?**
An architecture mapping input sequences to output sequences, used in translation, summarization, and dialogue.

**26. What is machine translation?**
Automatically translating text from one language to another using computational methods.

**27. What is text summarization?**
Condensing long documents into shorter versions while preserving key information.

**28. What is the difference between extractive and abstractive summarization?**
Extractive selects and combines existing sentences; abstractive generates new sentences paraphrasing the content.

**29. What is question answering (QA)?**
A task where systems provide answers to questions posed in natural language.

**30. What is coreference resolution?**
Identifying when different expressions refer to the same entity (e.g., "John" and "he" referring to the same person).

**31. What is dependency parsing?**
Analyzing grammatical structure by identifying relationships between words (subject, object, modifier).

**32. What is constituency parsing?**
Breaking sentences into sub-phrases or constituents based on phrase structure grammar.

**33. What is one-hot encoding in NLP?**
Representing words as binary vectors with one 1 and rest 0s, with dimension equal to vocabulary size.

**34. What is the limitation of one-hot encoding?**
It creates sparse, high-dimensional vectors that don't capture semantic similarity between words.

**35. What is padding in NLP?**
Adding special tokens to make all sequences in a batch the same length for efficient processing.

## Medium Questions (36-70)

**36. What is the attention mechanism in NLP?**
A technique allowing models to focus on relevant parts of input when generating output, computing weighted sums based on learned importance scores.

**37. What is the mathematical formulation of attention?**
Attention(Q,K,V) = softmax(QK^T/√d_k)V, where Q=queries, K=keys, V=values, and d_k is the key dimension for scaling.

**38. What is self-attention?**
Attention where queries, keys, and values all come from the same sequence, allowing each position to attend to all positions.

**39. What is the encoder-decoder architecture?**
A framework where an encoder processes input into a context representation, and a decoder generates output from that representation.

**40. What is the bottleneck problem in Seq2Seq models?**
The encoder must compress all input information into a fixed-size vector, losing information in long sequences. Attention mechanisms address this.

**41. What is BLEU score?**
Bilingual Evaluation Understudy - a metric measuring translation quality by comparing n-gram overlap between generated and reference translations.

**42. What is ROUGE score?**
Recall-Oriented Understudy for Gisting Evaluation - metrics measuring summarization quality through n-gram and longest common subsequence overlap.

**43. What is subword tokenization?**
Breaking words into smaller meaningful units (subwords), balancing vocabulary size and ability to handle rare/unseen words.

**44. What is Byte-Pair Encoding (BPE)?**
A subword tokenization algorithm that iteratively merges the most frequent character pairs, building a vocabulary of common subwords.

**45. What is WordPiece tokenization?**
Similar to BPE but chooses merges that maximize likelihood on training data, used in BERT.

**46. What is SentencePiece?**
A language-independent tokenizer treating text as raw character sequences, learning subword units directly from data.

**47. What is the [CLS] token?**
A special token added at the start of BERT inputs, whose final representation is used for classification tasks.

**48. What is the [SEP] token?**
A separator token used to distinguish between different segments (sentences) in BERT inputs.

**49. What is the [MASK] token?**
A special token used in BERT's masked language modeling to replace words that the model must predict.

**50. What is the [PAD] token?**
A padding token used to make sequences the same length in a batch.

**51. What is teacher forcing?**
A training technique using ground truth previous outputs (not predictions) as inputs at each time step in sequence generation.

**52. What is beam search decoding?**
A heuristic search keeping top-k candidate sequences at each step, exploring multiple hypotheses for better generation quality.

**53. What is greedy decoding?**
Selecting the highest probability token at each generation step, fast but potentially suboptimal.

**54. What is top-k sampling?**
Sampling the next token from the k most probable tokens, adding diversity while maintaining quality.

**55. What is nucleus (top-p) sampling?**
Sampling from the smallest set of tokens whose cumulative probability exceeds p, dynamically adjusting the number of candidates.

**56. What is temperature in text generation?**
A parameter T controlling randomness: p_i = exp(z_i/T)/Σexp(z_j/T). Higher T increases diversity, lower T makes output more deterministic.

**57. What is contextualized word embedding?**
Embeddings that vary based on context, unlike static embeddings (Word2Vec, GloVe) that have fixed representations.

**58. What is ELMo?**
Embeddings from Language Models - contextualized embeddings from bidirectional LSTM language models, combining forward and backward representations.

**59. What is the difference between static and contextualized embeddings?**
Static embeddings (Word2Vec) are fixed per word; contextualized embeddings (BERT, ELMo) change based on surrounding context, handling polysemy better.

**60. What is transfer learning in NLP?**
Pre-training models on large corpora then fine-tuning on specific tasks, leveraging learned linguistic knowledge.

**61. What is domain adaptation in NLP?**
Adapting models trained on one domain (e.g., news) to perform well on another domain (e.g., medical text).

**62. What is zero-shot learning in NLP?**
Models performing tasks without task-specific training examples, relying on pre-trained knowledge and instructions.

**63. What is few-shot learning in NLP?**
Learning from very few examples (typically 1-10) per class, using pre-trained models and prompting.

**64. What is prompt engineering?**
Designing input text templates that guide language models to produce desired outputs without fine-tuning.

**65. What is in-context learning?**
The ability of language models to learn from examples provided in the prompt without parameter updates.

**66. What is cross-lingual transfer?**
Applying models trained in one language to tasks in other languages, leveraging multilingual representations.

**67. What is back-translation?**
A data augmentation technique translating text to another language and back, creating paraphrased versions.

**68. What is active learning in NLP?**
Iteratively selecting the most informative examples for annotation, efficiently improving models with minimal labeled data.

**69. What is distant supervision?**
Automatically generating training labels using heuristics or external knowledge bases, trading label noise for scale.

**70. What is multi-task learning in NLP?**
Training a single model on multiple related tasks simultaneously, learning shared representations and improving generalization.

## Hard Questions (71-100)

**71. What is BERT and explain its pre-training objectives?**
Bidirectional Encoder Representations from Transformers uses two objectives: Masked Language Modeling (predicting randomly masked tokens using bidirectional context) and Next Sentence Prediction (classifying if sentence B follows sentence A).

**72. Explain masked language modeling (MLM) in detail.**
MLM randomly masks 15% of tokens (80% replaced with [MASK], 10% random word, 10% unchanged) and trains to predict them using bidirectional context. This enables learning deep bidirectional representations unlike unidirectional language models.

**73. What is the transformer architecture and why is it revolutionary?**
An architecture using only attention mechanisms (no recurrence), with multi-head self-attention and position-wise feedforward networks. Revolutionary because it enables parallelization (no sequential dependency), captures long-range dependencies, and scales efficiently.

**74. Explain multi-head attention mathematically.**
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O, where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V). Each head learns different aspects of relationships, and h different representations are learned in parallel.

**75. Why is scaled dot-product attention scaled by √d_k?**
Without scaling, dot products grow large in magnitude for large d_k, pushing softmax into regions with extremely small gradients. Scaling by √d_k keeps variance constant, maintaining gradient flow.

**76. What are positional encodings and why are they necessary?**
Transformers lack inherent position awareness. Positional encodings add position information: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d)), using different frequencies to distinguish positions.

**77. What is GPT and how does it differ from BERT?**
Generative Pre-trained Transformer uses unidirectional (left-to-right) causal language modeling, predicting next tokens. Unlike BERT's bidirectional encoding, GPT is designed for generation tasks.

**78. Explain the difference between autoencoding and autoregressive language models.**
Autoencoding (BERT) corrupts input and reconstructs it using bidirectional context (better for understanding). Autoregressive (GPT) predicts next tokens left-to-right (better for generation). Both pre-train unsupervised then fine-tune.

**79. What is T5 and its unified framework?**
Text-to-Text Transfer Transformer frames all NLP tasks as text-to-text, using the same model, loss, and training procedure. Input is task prefix + text, output is target text, enabling unified pre-training and transfer.

**80. What is BART and how does it combine BERT and GPT?**
Bidirectional and Auto-Regressive Transformer corrupts documents with various noise functions and trains a Seq2Seq model to reconstruct them. Encoder is bidirectional (BERT-like), decoder is autoregressive (GPT-like).

**81. Explain the denoising autoencoder objective in BART.**
BART applies diverse corruptions (token masking, deletion, permutation, rotation, text infilling) to input and trains to reconstruct the original. This combines benefits of both understanding (encoder) and generation (decoder).

**82. What is XLNet and how does it improve on BERT?**
XLNet uses Permutation Language Modeling, predicting tokens in random orders (not just masked positions), capturing bidirectional context without [MASK] tokens' discrepancy between pre-training and fine-tuning.

**83. Explain permutation language modeling.**
XLNet samples a factorization order of the sequence and predicts tokens according to that permutation while using bidirectional context from tokens appearing earlier in the permutation. Expected over all permutations enables full bidirectional learning.

**84. What is RoBERTa and its improvements over BERT?**
Robustly optimized BERT approach removes Next Sentence Prediction, trains with larger batches and more data, uses dynamic masking (different masks each epoch), and trains longer, significantly improving performance.

**85. What is ELECTRA and its replaced token detection?**
Efficiently Learning an Encoder that Classifies Token Replacements. Instead of masking, a generator produces replacements and discriminator detects which tokens are replaced. More sample-efficient as learning signal comes from all tokens, not just masked ones.

**86. What is the curse of multilingual models?**
Adding more languages to a fixed-capacity model can hurt per-language performance due to capacity dilution. Solutions include larger models, language-specific adapters, or selective language sampling.

**87. What is knowledge distillation in NLP?**
Training smaller "student" models to mimic larger "teacher" models by matching output distributions (with temperature softening). Examples include DistilBERT (6 layers vs BERT's 12) retaining 97% performance at 60% size.

**88. What is the lottery ticket hypothesis in NLP?**
Large pre-trained models contain smaller subnetworks that, when identified and trained from initialization, can match full model performance, suggesting over-parameterization during training but efficient inference.

**89. Explain sparse attention mechanisms.**
Reducing O(n²) attention complexity by attending to subset of positions using patterns (local windows, strided patterns, random). Examples: Longformer, BigBird, Sparse Transformer enable processing longer sequences efficiently.

**90. What is retrieval-augmented generation (RAG)?**
Combining parametric knowledge (in model weights) with non-parametric knowledge (retrieved documents). Models retrieve relevant documents then generate responses conditioned on both query and retrieved context.

**91. What is the difference between extractive and generative QA?**
Extractive QA selects answer spans from given context (SQuAD-style). Generative QA generates free-form answers, potentially synthesizing information or answering without context (open-domain QA).

**92. What is entity linking/disambiguation?**
Connecting entity mentions in text to canonical entries in knowledge bases (e.g., linking "Apple" to Apple Inc. vs. the fruit based on context).

**93. Explain the exposure bias problem in neural text generation.**
Training uses ground truth previous tokens (teacher forcing) but inference uses model predictions. Errors compound during generation as the model never trained on its own mistakes, causing distribution mismatch.

**94. What is minimum risk training/reinforcement learning for NLP?**
Directly optimizing evaluation metrics (BLEU, ROUGE) instead of cross-entropy by treating generation as RL problem. Policy gradient methods like REINFORCE update based on metric rewards.

**95. What is contrastive learning in NLP?**
Learning representations by pulling similar examples together and pushing dissimilar apart in embedding space. SimCSE uses dropout as augmentation, contrasting different dropout masks of the same sentence.

**96. Explain the theoretical relationship between attention and kernel methods.**
Attention can be viewed as kernel smoothing: output is weighted average of values where weights are kernel evaluations between query and keys. Linearized attention approximates this with explicit feature maps, reducing complexity.

**97. What is prompt-based learning and P-tuning?**
Instead of fine-tuning all parameters, prompt-based methods keep models frozen and learn continuous prompts (soft prompts) prepended to input. P-tuning uses prompt encoder networks to optimize these continuous prompts.

**98. What is adapter-based fine-tuning?**
Inserting small trainable adapter layers between frozen pre-trained layers. Only adapters are trained for new tasks, enabling parameter-efficient multi-task learning while maintaining pre-trained knowledge.

**99. Explain prefix-tuning and LoRA.**
Prefix-tuning prepends learnable vectors to each layer's key-value projections. LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to weight updates: W' = W + BA where B and A are low-rank, dramatically reducing trainable parameters.

**100. What is chain-of-thought prompting?**
Encouraging language models to generate intermediate reasoning steps before final answers, significantly improving performance on complex reasoning tasks. Few-shot examples demonstrate the reasoning process: "Let's think step by step..."