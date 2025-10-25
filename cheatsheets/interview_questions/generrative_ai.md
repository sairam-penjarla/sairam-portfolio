## Easy Questions (1-35)

**1. What is LLM chat completion?**
A task where a language model generates text responses based on input prompts or conversation history, creating human-like dialogue.

**2. What is a prompt in LLM context?**
The input text or instruction given to a language model to guide its response generation.

**3. What is temperature in text generation?**
A parameter (0.0-2.0) controlling randomness: lower values make output more deterministic, higher values increase creativity and diversity.

**4. What is top-p (nucleus) sampling?**
A sampling method that selects from the smallest set of tokens whose cumulative probability exceeds p, dynamically adjusting the candidate pool.

**5. What is top-k sampling?**
Sampling the next token from only the k most probable tokens, adding controlled randomness to generation.

**6. What is a system prompt?**
An initial instruction that sets the AI's behavior, role, and constraints for the entire conversation.

**7. What is RAG (Retrieval-Augmented Generation)?**
A technique that enhances LLM responses by retrieving relevant information from external knowledge sources before generating answers, combining retrieval with generation.

**8. What problem does RAG solve?**
It addresses LLM limitations like outdated knowledge, hallucinations, and lack of domain-specific information by grounding responses in retrieved data.

**9. What are the main components of a RAG system?**
A retriever that extracts relevant information from external sources, a knowledge base/vector store, and a generator (LLM) that produces responses.

**10. What is a vector database in RAG?**
A specialized database storing embeddings (numerical representations) of documents, enabling efficient semantic similarity search.

**11. What is an embedding?**
A dense vector representation of text that captures semantic meaning, allowing similar concepts to have similar vectors.

**12. What is semantic search?**
Searching based on meaning rather than keyword matching, using embeddings to find conceptually similar content.

**13. What is chunking in RAG?**
Breaking documents into smaller segments (chunks) for more efficient retrieval and processing within context limits.

**14. What is an AI agent?**
An autonomous or semi-autonomous software system that uses LLMs to understand, plan, and execute tasks by interacting with its environment and using tools.

**15. What distinguishes agents from simple chatbots?**
Agents can perform complex tasks, act independently, make decisions, use external tools, and carry out multi-step workflows that evolve over time.

**16. What is tool use in agents?**
The ability of agents to call external functions, APIs, or services to perform actions like web searches, calculations, or database queries.

**17. What is agent memory?**
The capability to store and recall information from past interactions, enabling personalized and context-aware responses.

**18. What is a token in LLM context?**
A unit of text (word, subword, or character) that the model processes, typically 3-4 characters per token in English.

**19. What is context window?**
The maximum amount of text (in tokens) that an LLM can process at once, including prompt and response.

**20. What is streaming in chat completion?**
Delivering model output incrementally as it's generated, rather than waiting for the complete response.

**21. What is few-shot prompting?**
Providing a few examples within the prompt to guide the model's response format and style.

**22. What is zero-shot prompting?**
Asking the model to perform a task without providing examples, relying only on instructions.

**23. What is chain-of-thought prompting?**
Encouraging language models to generate intermediate reasoning steps before final answers, improving performance on complex reasoning tasks.

**24. What is a completion API?**
An interface that accepts text input and returns generated text continuation from an LLM.

**25. What is chat history?**
The accumulated conversation between user and assistant, maintained to provide context for ongoing dialogue.

**26. What is a function call in LLM APIs?**
A structured way for models to indicate they want to use external tools, returning parameters for function execution.

**27. What is hallucination in LLMs?**
When models generate plausible-sounding but factually incorrect or fabricated information.

**28. What is grounding in LLM context?**
Connecting model outputs to verifiable sources or retrieved information to reduce hallucinations.

**29. What is fine-tuning an LLM?**
Further training a pre-trained model on specific data to adapt it to particular tasks or domains.

**30. What is prompt engineering?**
Designing input text templates that guide language models to produce desired outputs without modifying model weights.

**31. What is a message role in chat APIs?**
Labels (system, user, assistant) that identify who sent each message in a conversation.

**32. What is max tokens?**
The maximum number of tokens the model can generate in a single response.

**33. What is a stop sequence?**
A text pattern that tells the model to stop generating when encountered.

**34. What is latency in LLM inference?**
The time delay between sending a prompt and receiving the complete response.

**35. What is batch processing in LLM APIs?**
Processing multiple requests together for improved throughput and cost efficiency.

## Medium Questions (36-70)

**36. What is the difference between standard RAG and agentic RAG?**
Standard RAG relies on one-shot retrieval while agentic RAG uses autonomous agents that plan, remember, and use tools dynamically, retrieving information at multiple points during execution rather than just at the beginning.

**37. How does vector similarity search work in RAG?**
Query and document embeddings are compared using metrics like cosine similarity or dot product, ranking documents by semantic relevance to retrieve the most similar ones.

**38. What is the retrieval-generation pipeline?**
The workflow where: 1) user query is embedded, 2) similar documents are retrieved, 3) retrieved context is combined with query, 4) LLM generates response using this context.

**39. What is context relevance in RAG?**
A measure of how well retrieved documents relate to the user's query, critical for response quality.

**40. What is answer faithfulness in RAG?**
The degree to which the generated answer is supported by the retrieved context, indicating grounding accuracy.

**41. What are the limitations of large context windows vs. RAG?**
Large context windows struggle to distinguish valuable information when flooded with unfiltered data, especially information buried in the middle. Costs also increase linearly with larger contexts.

**42. What is hybrid search in RAG?**
Combining semantic (vector) search with keyword-based search to improve retrieval accuracy.

**43. What is re-ranking in RAG?**
A second-stage process that reorders retrieved documents using more sophisticated models to improve relevance of top results.

**44. What is query decomposition in RAG?**
Breaking complex queries into simpler sub-queries that can be processed separately and combined for better results.

**45. What is document augmentation in RAG?**
Enhancing stored documents with metadata, summaries, or generated questions to improve retrieval quality.

**46. What is agent planning?**
The ability of agents to break down complex tasks into manageable sub-tasks and determine the sequence of actions needed.

**47. What is ReAct (Reasoning + Acting)?**
An agent framework that interleaves reasoning traces with actions, allowing agents to think through problems and execute tool calls systematically.

**48. What is multi-agent collaboration?**
Systems where multiple specialized agents work together, each handling different aspects of a complex task.

**49. What is LangChain?**
A framework for building LLM applications with components for prompt templates, chains, agents, memory, and tool integration.

**50. What is LangGraph?**
A framework compatible with LangChain agents that allows developers to modify agent internals more easily, with state including input, chat history, intermediate steps, and agent outcomes.

**51. What is AutoGen?**
A multi-agent framework where agents can be assigned different roles and autonomously communicate to solve tasks.

**52. What is CrewAI?**
A framework for orchestrating role-based AI agents that collaborate like a crew, with defined roles, goals, and backstories.

**53. What is tool selection in agents?**
The agent's ability to choose the appropriate tool from available options based on the current task requirements.

**54. What is observation-action loop?**
The cycle where agents observe their environment, decide on actions, execute them, and observe the results to inform next steps.

**55. What is agent memory types?**
Short-term (conversation context), long-term (persistent storage), and semantic memory (knowledge embeddings) that agents can access.

**56. What is prompt chaining?**
Connecting multiple prompts sequentially where the output of one becomes input to the next, building complex workflows.

**57. What is constitutional AI?**
Training models with explicit principles and rules to ensure safe, helpful, and honest behavior through reinforcement learning.

**58. What is RLHF (Reinforcement Learning from Human Feedback)?**
Training LLMs using human preferences to align outputs with desired qualities like helpfulness and harmlessness.

**59. What is instruction tuning?**
Fine-tuning LLMs on datasets of instructions and desired responses to improve their ability to follow diverse commands.

**60. What is context caching?**
Storing processed context (like system prompts or long documents) to reuse across multiple requests, reducing latency and cost.

**61. What is response validation in agents?**
Checking agent outputs against criteria or constraints before executing actions or returning to users.

**62. What is human-in-the-loop (HITL)?**
Systems that require human approval or input at critical decision points before agents proceed.

**63. What is error recovery in agents?**
The ability to detect failures, retry with different approaches, or escalate to humans when stuck.

**64. What is task decomposition?**
Breaking complex objectives into smaller, manageable sub-tasks that can be solved independently.

**65. What is smart routing in RAG systems?**
Routing tasks to the right language models based on query type, such as sending programming queries to code-optimized models or healthcare queries to specialized medical LLMs.

**66. What is semantic chunking?**
Dividing documents based on semantic boundaries (topics, sections) rather than fixed sizes for more coherent retrieval units.

**67. What is query expansion?**
Generating multiple variations or related queries to increase recall in retrieval systems.

**68. What is negative sampling in RAG?**
Including examples of irrelevant documents during training to help models better distinguish relevant from irrelevant context.

**69. What is dynamic retrieval in agents?**
Retrieval that doesn't happen only at the beginning but can occur at any point during execution, with the agent deciding when it needs more information.

**70. What is mixture-of-agents?**
Systems that leverage multiple agents to help optimize LLM performance and cost by routing to specialized models.

## Hard Questions (71-100)

**71. Explain the mathematical formulation of semantic similarity in vector search.**
Cosine similarity: sim(A,B) = (A·B)/(||A|| ||B||) = Σ(A_i × B_i)/√(ΣA_i² × ΣB_i²), measuring angle between vectors. Values range from -1 to 1, where 1 indicates identical direction. Dot product: A·B = Σ(A_i × B_i) is faster but sensitive to magnitude.

**72. What is the lost-in-the-middle problem and how does RAG address it?**
LLMs struggle when valuable information is buried in the middle portion of large contexts. RAG addresses this by retrieving only the most relevant chunks, keeping context focused and placing important information prominently.

**73. Explain advanced RAG architectures and their evolution.**
Advanced RAG pipelines can dynamically fetch additional chunks during retrieval to ensure accurate, context-rich responses. They include multi-hop retrieval (iterative fetching), self-reflection (validating and refining), and agentic approaches with planning.

**74. What is self-RAG and how does it improve reliability?**
Self-RAG adds reflection tokens where the model evaluates retrieval necessity, relevance of retrieved docs, and response quality. It can trigger re-retrieval or generation refinement, creating a self-correcting loop.

**75. What is FLARE (Forward-Looking Active Retrieval)?**
A method where the model generates responses incrementally and retrieves additional information when uncertain (low probability tokens), proactively preventing hallucinations.

**76. Explain the bipartite matching loss in function calling.**
Modern function calling uses similarity matching between model-generated function representations and available tool embeddings, computing optimal assignment. Loss = -log P(correct_tool|query, context) encourages learning to select appropriate tools.

**77. What is the agent-environment interaction formalism?**
Modeled as a Markov Decision Process: agent observes state s_t, takes action a_t, receives observation o_{t+1} and optional reward r_t, transitions to s_{t+1}. Policy π(a|s) maps states to actions, optimized to maximize cumulative reward.

**78. Explain planning algorithms used in LLM agents.**
Tree-of-Thoughts explores multiple reasoning paths as a tree. Plan-and-Solve generates plans before execution. ReWOO (Reasoning WithOut Observation) plans all steps upfront, then executes. Each trades off between planning overhead and execution efficiency.

**79. What is the exposure bias problem in agent training?**
During training, agents see expert demonstrations or ground truth states. At deployment, they rely on their own (potentially erroneous) previous actions, creating distribution shift. Solutions include scheduled sampling, DAgger (Dataset Aggregation), or RL fine-tuning.

**80. Explain embedding models for RAG and their evaluation.**
Models like text-embedding-ada-002, Cohere embeddings, or Sentence-BERT create dense vectors. Evaluated on: retrieval accuracy (recall@k, MRR), semantic similarity benchmarks (MTEB), and downstream RAG task performance.

**81. What is ColBERT and late interaction?**
ColBERT creates token-level embeddings for queries and documents, performing late interaction (MaxSim) at retrieval time: Score(q,d) = Σ_{q_i} max_{d_j} (q_i · d_j^T). This balances expressiveness with efficiency.

**82. Explain knowledge distillation for agent policies.**
Training smaller student agents to mimic larger teacher agents by minimizing KL divergence: L = KL(π_teacher || π_student) + αL_task. Enables deploying efficient agents that retain most of the teacher's capabilities.

**83. What is the agent architecture proposed in ReAct?**
Thought-Action-Observation loop: 1) Thought: reasoning about current situation, 2) Action: tool call or final answer, 3) Observation: tool output or environment feedback. This explicit reasoning improves interpretability and error recovery.

**84. Explain the trade-offs in RAG chunk size optimization.**
Smaller chunks (128-256 tokens): precise retrieval but may lack context. Larger chunks (512-1024): more context but diluted relevance. Optimal size depends on: document structure, query complexity, and model context window. Hierarchical chunking can combine benefits.

**85. What is the evaluation framework for agentic systems?**
Agentic RAG supports ongoing validation and refinement, reducing errors in high-stakes tasks. Metrics include: task success rate, number of steps, tool usage efficiency, hallucination rate, human intervention frequency, and cost per task.

**86. Explain prompt optimization techniques for agents.**
DSPy (Declarative Self-improving Python) treats prompts as learnable parameters, optimizing through: metric-driven compilation, bootstrapping examples from model traces, and automatic refinement. Other methods include APE (Automatic Prompt Engineer) and gradient-based soft prompt tuning.

**87. What is the computational complexity analysis of RAG vs. fine-tuning?**
RAG: O(k × d) retrieval + O(n) generation per query, where k=retrieved docs, d=embedding dimension, n=output length. No training cost. Fine-tuning: O(B × L × P) training cost for batch B, sequence L, parameters P. One-time cost but model-specific. RAG scales better for frequent updates.

**88. Explain agent architectures: reflexive vs. deliberative.**
Reflexive agents (ReAct-style) interleave thinking and acting, adapting in real-time. Deliberative agents (Plan-and-Execute) create complete plans before execution. Hybrid approaches (BabyAGI) plan at high level, then execute reflexively. Trade-off: planning overhead vs. adaptability.

**89. What is constitutional AI for agents and how is it implemented?**
Agents are given constitutions (principles) like "be helpful and harmless." Implementation: 1) Supervised learning on principle-following demonstrations, 2) RL with reward modeling based on constitutional criteria, 3) Self-critique where agents evaluate their own actions against principles.

**90. Explain memory architectures in long-running agents.**
Episodic memory: stores event sequences (what happened when). Semantic memory: structured knowledge (facts, skills). Working memory: current context. Implementations use vector databases for semantic search, graph databases for relationships, and summarization for compression.

**91. What is the curse of tool accumulation in agents?**
As available tools increase, selection becomes harder: more false positives, higher latency, increased prompt size. Solutions: hierarchical tool organization, learned tool retrievers, adaptive tool subsetting based on task context.

**92. Explain evaluation metrics for RAG systems: faithfulness vs. relevance.**
Faithfulness: does answer follow from context? Measured by NLI models checking entailment. Relevance: does context relate to query? Measured by embedding similarity or reranker scores. Both needed: high relevance with low faithfulness indicates hallucination.

**93. What is the agent-computer interface problem?**
Agents need reliable ways to interact with software. Approaches: 1) APIs (structured, reliable but limited), 2) Computer-use APIs (screenshot + mouse/keyboard, flexible but brittle), 3) Hybrid with action grounding. Challenge: balancing capability with reliability.

**94. Explain the economics of RAG vs. long context vs. fine-tuning.**
RAG: retrieval cost + smaller context × usage frequency. Long context: higher per-token cost × large windows × usage. Fine-tuning: training cost (one-time) + inference (regular). RAG wins for: frequently updating knowledge, domain switching, or cost-sensitive high-volume applications.

**95. What is meta-prompting for agent orchestration?**
A meta-agent receives user requests, decomposes into subtasks, dynamically generates specialized prompts for sub-agents, and synthesizes results. Enables scaling to complex problems without hardcoded workflows.

**96. Explain the verification-refinement loop in agents.**
Agentic systems track what they've done and use that history to guide their next steps. Pattern: 1) Generate candidate action/response, 2) Verify against constraints/goals, 3) If invalid, refine and retry, 4) If valid, execute. Reduces errors at cost of latency.

**97. What are the security implications of agentic systems?**
Risks: prompt injection via retrieved docs, unauthorized tool access, data exfiltration, unintended actions. Mitigations: sandboxing, permission systems, action logging, human approval gates, adversarial testing, and constitutional constraints.

**98. Explain the role of structured outputs in agent reliability.**
Function calling and JSON mode force agents to produce valid tool calls or structured data. Reduces parsing errors, enables validation, and improves integration. Implemented via: constrained decoding, grammar-based sampling, or finetuning on structured data.

**99. What is the future of agentic AI according to 2025 trends?**
2025 is the year of the agent, with agentic AI focusing on autonomy, planning, and orchestration, executing multi-step workflows, collaborating with other agents, and aligning outputs with strategic objectives. 33% of enterprise software is expected to include agentic AI by 2028, up from under 1% in 2024.

**100. Explain the convergence of RAG and agents in modern systems.**
In the age of reasoning LLMs, retrieval-augmented generation can be seamlessly incorporated with agents requiring more flexibility and control over their entire workflow. RAG offers access to external data to ground the decisions an agent makes, creating systems that combine knowledge retrieval with autonomous task execution.