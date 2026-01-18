"""
LLM Workflow Demonstration
===========================
This module demonstrates how a Large Language Model (LLM) works by implementing
a simplified version of the core components: tokenization, embedding, 
transformer/self-attention, prediction, and response generation.

Author: AI Assistant
Date: January 3, 2026
"""

import numpy as np
from typing import List, Dict, Tuple
import re


class SimpleTokenizer:
    """
    Tokenizer that breaks text into tokens for the model to process.
    
    This is a simplified version of tokenization used in real LLMs.
    Real tokenizers (like BPE or WordPiece) are more sophisticated.
    """
    
    def __init__(self):
        """Initialize the tokenizer with a vocabulary."""
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        # Add special tokens
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.vocab_size = 4
        
        # Tokenize and add words to vocabulary
        for text in texts:
            words = self._split_text(text)
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1
        
        # Create reverse vocabulary for decoding
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into words and punctuation."""
        # Simple word splitting with punctuation handling
        text = text.lower()
        words = re.findall(r'\b\w+\b|[.,!?;]', text)
        return words
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text into token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        words = self._split_text(text)
        token_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        words = [self.reverse_vocab.get(idx, "<UNK>") for idx in token_ids]
        # Simple reconstruction (not perfect but demonstrates the concept)
        text = " ".join(words)
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.,!?;])', r'\1', text)
        return text


class EmbeddingLayer:
    """
    Embedding layer that converts tokens into numerical vectors that represent meaning.
    
    Each token is mapped to a high-dimensional vector (embedding) where similar
    words have similar vectors.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize the embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize random embeddings (in real LLMs, these are learned)
        # Each token gets a random vector representation
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
    def embed(self, token_ids: List[int]) -> np.ndarray:
        """
        Convert token IDs to embedding vectors.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Array of shape (sequence_length, embedding_dim)
        """
        # Look up embedding vector for each token
        embedded = np.array([self.embeddings[idx] for idx in token_ids])
        return embedded


class SelfAttentionLayer:
    """
    Self-Attention mechanism that allows the model to focus on the most relevant
    parts of the input text.
    
    Attention helps the model understand relationships between different words
    in the sequence, regardless of their distance from each other.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize the self-attention layer.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        
        # Query, Key, Value matrices (simplified - in real transformers, these are learned)
        self.W_query = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_key = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_value = np.random.randn(embedding_dim, embedding_dim) * 0.1
        
    def attention(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply self-attention mechanism to the embeddings.
        
        The attention mechanism:
        1. Computes Query, Key, Value from input embeddings
        2. Calculates attention scores (how much each token should attend to others)
        3. Applies attention weights to values to get context-aware representations
        
        Args:
            embeddings: Input embeddings of shape (sequence_length, embedding_dim)
            
        Returns:
            Context-aware embeddings of shape (sequence_length, embedding_dim)
        """
        # Compute Query, Key, Value
        Q = np.dot(embeddings, self.W_query)  # Queries: what am I looking for?
        K = np.dot(embeddings, self.W_key)     # Keys: what do I contain?
        V = np.dot(embeddings, self.W_value)   # Values: what do I actually represent?
        
        # Compute attention scores: how much should each token attend to others?
        # Using scaled dot-product attention
        scores = np.dot(Q, K.T) / np.sqrt(self.embedding_dim)
        
        # Apply softmax to get attention weights (probabilities)
        attention_weights = self._softmax(scores, axis=1)
        
        # Apply attention weights to values
        context_vectors = np.dot(attention_weights, V)
        
        return context_vectors
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax values for each set of scores."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PredictionLayer:
    """
    Prediction layer that converts context-aware vectors into probability
    distributions over the vocabulary for next token prediction.
    """
    
    def __init__(self, embedding_dim: int, vocab_size: int):
        """
        Initialize the prediction layer.
        
        Args:
            embedding_dim: Dimension of the input embeddings
            vocab_size: Size of the vocabulary
        """
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Output projection matrix (in real LLMs, this is learned)
        self.W_output = np.random.randn(embedding_dim, vocab_size) * 0.1
        
    def predict(self, context_vector: np.ndarray) -> np.ndarray:
        """
        Predict probability distribution over vocabulary.
        
        Args:
            context_vector: Context-aware embedding vector
            
        Returns:
            Probability distribution over vocabulary
        """
        # Project to vocabulary size
        logits = np.dot(context_vector, self.W_output)
        
        # Apply softmax to get probabilities
        probabilities = self._softmax(logits)
        
        return probabilities
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class SimpleLLM:
    """
    A simplified Large Language Model that demonstrates the complete workflow:
    Tokenization → Embedding → Transformer/Self-Attention → Prediction → Response Generation
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize the LLM.
        
        Args:
            embedding_dim: Dimension of embedding vectors (default: 64)
        """
        self.embedding_dim = embedding_dim
        self.tokenizer = SimpleTokenizer()
        self.embedding_layer = None
        self.attention_layer = None
        self.prediction_layer = None
        
    def train(self, training_texts: List[str]) -> None:
        """
        Train the model on a corpus of text (simplified training).
        
        In reality, LLM training involves:
        - Massive datasets (billions of tokens)
        - Gradient descent optimization
        - Multiple transformer layers
        - Extensive compute resources
        
        Args:
            training_texts: List of training text samples
        """
        print("=" * 70)
        print("TRAINING THE MODEL")
        print("=" * 70)
        
        # Step 1: Build vocabulary from training texts
        print("\n1. Building vocabulary from training texts...")
        self.tokenizer.build_vocab(training_texts)
        print(f"   Vocabulary size: {self.tokenizer.vocab_size} tokens")
        
        # Step 2: Initialize model components
        print("\n2. Initializing model components...")
        self.embedding_layer = EmbeddingLayer(
            self.tokenizer.vocab_size, 
            self.embedding_dim
        )
        self.attention_layer = SelfAttentionLayer(self.embedding_dim)
        self.prediction_layer = PredictionLayer(
            self.embedding_dim, 
            self.tokenizer.vocab_size
        )
        print("   ✓ Embedding layer initialized")
        print("   ✓ Self-attention layer initialized")
        print("   ✓ Prediction layer initialized")
        
        print("\n   Note: In real LLMs, extensive training would occur here")
        print("   using backpropagation and gradient descent on massive datasets.")
        
    def generate_response(self, prompt: str, max_length: int = 20) -> str:
        """
        Generate a response iteratively by predicting one word at a time.
        
        This demonstrates the complete LLM workflow:
        1. Tokenization: Break input into tokens
        2. Embedding: Convert tokens to vectors
        3. Self-Attention: Focus on relevant parts
        4. Prediction: Generate next token
        5. Response Generation: Repeat until complete
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated sequence
            
        Returns:
            Generated text response
        """
        print("\n" + "=" * 70)
        print("GENERATING RESPONSE")
        print("=" * 70)
        
        # Step 1: TOKENIZATION
        print(f"\n1. TOKENIZATION: Breaking text into tokens")
        print(f"   Input: '{prompt}'")
        token_ids = self.tokenizer.encode(prompt)
        print(f"   Token IDs: {token_ids}")
        tokens = [self.tokenizer.reverse_vocab[idx] for idx in token_ids]
        print(f"   Tokens: {tokens}")
        
        # Step 2: EMBEDDING
        print(f"\n2. EMBEDDING: Converting tokens to numerical vectors")
        embeddings = self.embedding_layer.embed(token_ids)
        print(f"   Shape: {embeddings.shape}")
        print(f"   Each token is now a {self.embedding_dim}-dimensional vector")
        print(f"   Example (first token vector): [{embeddings[0][:5]}...]")
        
        # Step 3: TRANSFORMER / SELF-ATTENTION
        print(f"\n3. SELF-ATTENTION: Focusing on relevant parts of input")
        context_vectors = self.attention_layer.attention(embeddings)
        print(f"   Shape: {context_vectors.shape}")
        print(f"   Attention mechanism has computed context-aware representations")
        print(f"   Each token now 'knows' about other tokens in the sequence")
        
        # Steps 4 & 5: PREDICTION & RESPONSE GENERATION
        print(f"\n4. PREDICTION & RESPONSE GENERATION:")
        print(f"   Iteratively predicting next tokens...")
        
        generated_ids = token_ids.copy()
        
        for i in range(max_length):
            # Use the last context vector to predict next token
            last_context = context_vectors[-1]
            
            # Step 4: PREDICTION - Get probability distribution
            probabilities = self.prediction_layer.predict(last_context)
            
            # Sample next token (using argmax for simplicity)
            # Real LLMs use temperature-based sampling for variety
            next_token_id = np.argmax(probabilities)
            
            # Check for end token or repetition
            if next_token_id == self.tokenizer.vocab.get("<END>", -1):
                break
            if next_token_id in [0, 1]:  # PAD or UNK
                break
                
            generated_ids.append(next_token_id)
            
            # Update embeddings and attention for next iteration
            new_embedding = self.embedding_layer.embed([next_token_id])
            embeddings = np.vstack([embeddings, new_embedding])
            context_vectors = self.attention_layer.attention(embeddings)
            
            # Show progress
            if (i + 1) % 5 == 0:
                print(f"   Generated {i + 1} tokens...")
        
        # Step 5: RESPONSE GENERATION - Decode back to text
        print(f"\n5. DECODING: Converting token IDs back to text")
        generated_text = self.tokenizer.decode(generated_ids)
        print(f"   Final token count: {len(generated_ids)}")
        
        return generated_text


def main():
    """
    Main demonstration of the LLM workflow.
    """
    print("\n" + "=" * 70)
    print("LARGE LANGUAGE MODEL (LLM) WORKFLOW DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how an LLM works through 5 key stages:")
    print("1. Tokenization - Breaking text into tokens")
    print("2. Embedding - Converting tokens to numerical vectors")
    print("3. Transformer/Self-Attention - Focusing on relevant information")
    print("4. Prediction - Computing probability distributions for next tokens")
    print("5. Response Generation - Iteratively generating coherent text")
    
    # Create training corpus
    training_corpus = [
        "The cat sat on the mat.",
        "A dog ran in the park.",
        "The bird flew over the tree.",
        "I love programming in Python.",
        "Machine learning is fascinating.",
        "Natural language processing is powerful.",
        "The sun rises in the east.",
        "Water flows in the river.",
        "Books contain knowledge and wisdom.",
        "Technology advances every day."
    ]
    
    # Initialize and train the model
    llm = SimpleLLM(embedding_dim=64)
    llm.train(training_corpus)
    
    # Generate responses for sample prompts
    prompts = [
        "the cat",
        "machine learning",
        "the sun"
    ]
    
    print("\n" + "=" * 70)
    print("TESTING THE MODEL")
    print("=" * 70)
    
    for prompt in prompts:
        response = llm.generate_response(prompt, max_length=10)
        print("\n" + "-" * 70)
        print(f"FINAL OUTPUT:")
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{response}'")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("• LLMs process text through multiple stages of transformation")
    print("• Tokenization converts text to numbers the model can process")
    print("• Embeddings capture semantic meaning in vector space")
    print("• Self-attention allows modeling complex relationships between words")
    print("• Prediction and generation happen iteratively, one token at a time")
    print("• Real LLMs are much more complex with billions of parameters!")
    print()


def huggingface_llama_demo():
    """
    Demonstration of using real LLMs with Hugging Face and LLaMA models.
    
    This section shows how to use production-ready LLMs like LLaMA, GPT-2,
    and other models from the Hugging Face model hub.
    
    Note: Requires additional packages:
    - transformers
    - torch
    - accelerate (for larger models)
    """
    print("\n" + "=" * 70)
    print("HUGGING FACE & LLAMA MODEL DEMONSTRATION")
    print("=" * 70)
    
    try:
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM, 
            pipeline,
            LlamaTokenizer,
            LlamaForCausalLM
        )
        import torch
        
        print("\n✓ Hugging Face transformers library loaded successfully!")
        
    except ImportError:
        print("\n⚠ Hugging Face transformers not installed.")
        print("\nTo run this demo, install required packages:")
        print("  pip install transformers torch accelerate")
        print("\nFor LLaMA models, you may also need:")
        print("  pip install sentencepiece protobuf")
        print("\nNote: LLaMA models require authentication and acceptance of license.")
        return
    
    # Example 1: Using GPT-2 (smaller, no authentication needed)
    print("\n" + "-" * 70)
    print("EXAMPLE 1: GPT-2 Model (OpenAI)")
    print("-" * 70)
    
    try:
        print("\nLoading GPT-2 model and tokenizer...")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        print("✓ Model loaded successfully!")
        
        # Set pad token (GPT-2 doesn't have one by default)
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        
        # Generate text
        prompt = "The future of artificial intelligence is"
        print(f"\nPrompt: '{prompt}'")
        
        print("\n--- Tokenization Stage ---")
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")
        print(f"Token IDs: {inputs['input_ids'].tolist()[0]}")
        print(f"Tokens: {gpt2_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        
        print("\n--- Generation Stage ---")
        print("Generating response (this may take a moment)...")
        
        with torch.no_grad():
            outputs = gpt2_model.generate(
                inputs['input_ids'],
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=gpt2_tokenizer.eos_token_id
            )
        
        generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated Text:\n{generated_text}")
        
    except Exception as e:
        print(f"\n⚠ Error with GPT-2: {e}")
    
    # Example 2: Using Hugging Face Pipeline (Simplified API)
    print("\n" + "-" * 70)
    print("EXAMPLE 2: Using Hugging Face Pipeline API")
    print("-" * 70)
    
    try:
        print("\nCreating text generation pipeline with DistilGPT-2...")
        generator = pipeline(
            'text-generation', 
            model='distilgpt2',
            device=-1  # Use CPU (-1) or GPU (0)
        )
        
        print("✓ Pipeline created successfully!")
        
        prompts = [
            "Machine learning is",
            "The best programming language for AI is"
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            result = generator(
                prompt, 
                max_length=40,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9
            )
            print(f"Generated: {result[0]['generated_text']}")
            
    except Exception as e:
        print(f"\n⚠ Error with pipeline: {e}")
    
    # Example 3: LLaMA Model (requires authentication)
    print("\n" + "-" * 70)
    print("EXAMPLE 3: LLaMA Models (Meta)")
    print("-" * 70)
    
    print("\nLLaMA Model Information:")
    print("• LLaMA (Large Language Model Meta AI) is Meta's open-source LLM")
    print("• Available in various sizes: 7B, 13B, 30B, 65B parameters")
    print("• LLaMA-2 and LLaMA-3 are newer versions with improvements")
    
    print("\nHow to use LLaMA models:")
    print("\n1. Request access from Meta/Hugging Face:")
    print("   https://huggingface.co/meta-llama")
    
    print("\n2. Login to Hugging Face:")
    print("   huggingface-cli login")
    
    print("\n3. Example code to load LLaMA:")
    print("""
    # For LLaMA-2 (7B model)
    from transformers import LlamaTokenizer, LlamaForCausalLM
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision for memory
        device_map="auto"            # Automatically distribute across GPUs
    )
    
    # Generate text
    prompt = "What is the meaning of life?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    """)
    
    print("\n4. Smaller alternative - TinyLlama (1.1B parameters):")
    print("   More accessible, no authentication needed:")
    
    try:
        print("\n   Loading TinyLlama (this is a real, working example)...")
        tiny_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tiny_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        print("   ✓ TinyLlama loaded successfully!")
        
        # Format prompt in chat format
        prompt = "What are the key components of a neural network?"
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Some models expect specific chat formatting
        formatted_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        print(f"\n   Prompt: '{prompt}'")
        print("   Generating response...")
        
        inputs = tiny_tokenizer(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = tiny_model.generate(
                **inputs,
                max_length=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tiny_tokenizer.eos_token_id
            )
        
        response = tiny_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n   Generated Response:\n   {response}")
        
    except Exception as e:
        print(f"\n   ⚠ Could not load TinyLlama: {e}")
        print("   (This is normal if you have limited RAM/GPU)")


def huggingface_advanced_features():
    """
    Advanced Hugging Face features for working with LLMs.
    
    Demonstrates:
    - Different decoding strategies
    - Streaming generation
    - Batch processing
    - Model quantization for efficiency
    """
    print("\n" + "=" * 70)
    print("ADVANCED HUGGING FACE FEATURES")
    print("=" * 70)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
        import torch
        from threading import Thread
        
        print("\n--- Feature 1: Different Decoding Strategies ---")
        print("""
Decoding strategies affect how the model generates text:

1. Greedy Decoding (do_sample=False):
   - Always picks the most likely next token
   - Deterministic but can be repetitive
   
2. Sampling (do_sample=True):
   - Randomly samples from probability distribution
   - More creative and diverse
   
3. Top-k Sampling (top_k=50):
   - Samples from top k most likely tokens
   - Balances quality and diversity
   
4. Top-p (Nucleus) Sampling (top_p=0.9):
   - Samples from smallest set of tokens with cumulative probability >= p
   - Adaptive, high quality
   
5. Temperature (temperature=0.7):
   - Controls randomness: lower = more deterministic, higher = more random
   - Typical range: 0.1 to 1.5
   
6. Beam Search (num_beams=5):
   - Maintains multiple hypotheses
   - Better quality but slower
        """)
        
        print("\nExample code for different strategies:")
        print("""
# Greedy (deterministic)
output = model.generate(input_ids, max_length=50)

# Sampling with temperature
output = model.generate(
    input_ids, 
    do_sample=True, 
    temperature=0.8,
    max_length=50
)

# Top-k sampling
output = model.generate(
    input_ids,
    do_sample=True,
    top_k=50,
    max_length=50
)

# Top-p (nucleus) sampling  
output = model.generate(
    input_ids,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    max_length=50
)

# Beam search
output = model.generate(
    input_ids,
    num_beams=5,
    max_length=50,
    early_stopping=True
)
        """)
        
        print("\n--- Feature 2: Streaming Generation ---")
        print("For real-time output (like ChatGPT's typing effect):")
        print("""
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

generation_kwargs = {
    "input_ids": inputs,
    "streamer": streamer,
    "max_length": 100,
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)
        """)
        
        print("\n--- Feature 3: Model Quantization ---")
        print("Reduce memory usage with quantization:")
        print("""
# 8-bit quantization (requires bitsandbytes)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# 4-bit quantization (even more memory efficient)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
        """)
        
        print("\n--- Feature 4: Batch Processing ---")
        print("Process multiple prompts efficiently:")
        print("""
prompts = [
    "The capital of France is",
    "Machine learning is used for",
    "The solar system has"
]

# Tokenize all prompts
inputs = tokenizer(
    prompts, 
    return_tensors="pt", 
    padding=True,
    truncation=True
)

# Generate for all at once
outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id
)

# Decode all results
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        """)
        
        print("\n--- Feature 5: Fine-tuning LLMs ---")
        print("Adapt pre-trained models to your specific task:")
        print("""
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your_dataset")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=500,
    logging_steps=100,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train
trainer.train()
        """)
        
    except ImportError:
        print("\n⚠ Transformers library not available")
        print("Install with: pip install transformers torch")


def practical_llm_applications():
    """
    Practical applications and use cases for LLMs.
    """
    print("\n" + "=" * 70)
    print("PRACTICAL LLM APPLICATIONS")
    print("=" * 70)
    
    applications = {
        "1. Text Generation": {
            "description": "Generate creative content, stories, articles",
            "models": ["GPT-2", "GPT-3", "LLaMA", "Falcon"],
            "code": """
generator = pipeline('text-generation', model='gpt2')
result = generator("Once upon a time", max_length=100)
            """
        },
        
        "2. Question Answering": {
            "description": "Answer questions based on context or knowledge",
            "models": ["BERT", "RoBERTa", "T5", "LLaMA-2-Chat"],
            "code": """
qa_pipeline = pipeline('question-answering')
result = qa_pipeline(question="What is AI?", context="Artificial Intelligence...")
            """
        },
        
        "3. Text Summarization": {
            "description": "Condense long documents into summaries",
            "models": ["BART", "T5", "Pegasus", "LLaMA"],
            "code": """
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
summary = summarizer("Long text...", max_length=130)
            """
        },
        
        "4. Translation": {
            "description": "Translate between languages",
            "models": ["MarianMT", "T5", "mBART", "NLLB"],
            "code": """
translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')
result = translator("Hello, how are you?")
            """
        },
        
        "5. Chatbots": {
            "description": "Conversational AI assistants",
            "models": ["LLaMA-2-Chat", "Vicuna", "Mistral-Instruct"],
            "code": """
# Using LLaMA-2-Chat format
prompt = "<s>[INST] <<SYS>>You are a helpful assistant<</SYS>> Hi there! [/INST]"
response = model.generate(tokenizer(prompt, return_tensors="pt").input_ids)
            """
        },
        
        "6. Code Generation": {
            "description": "Generate programming code from descriptions",
            "models": ["CodeLlama", "StarCoder", "Codex"],
            "code": """
# Using CodeLlama
prompt = "# Python function to calculate fibonacci numbers\\ndef fibonacci("
result = model.generate(tokenizer(prompt, return_tensors="pt").input_ids)
            """
        },
        
        "7. Sentiment Analysis": {
            "description": "Analyze emotional tone of text",
            "models": ["BERT", "RoBERTa", "DistilBERT"],
            "code": """
sentiment = pipeline('sentiment-analysis')
result = sentiment("I love this product!")
            """
        },
        
        "8. Named Entity Recognition": {
            "description": "Extract entities (names, places, etc.) from text",
            "models": ["BERT", "RoBERTa", "SpaCy-transformers"],
            "code": """
ner = pipeline('ner', grouped_entities=True)
entities = ner("Apple Inc. is located in Cupertino, California")
            """
        }
    }
    
    for app_name, details in applications.items():
        print(f"\n{app_name}: {details['description']}")
        print(f"   Popular Models: {', '.join(details['models'])}")
        print(f"   Example Code:\n{details['code']}")


def llm_best_practices():
    """
    Best practices for working with LLMs.
    """
    print("\n" + "=" * 70)
    print("LLM BEST PRACTICES & TIPS")
    print("=" * 70)
    
    practices = {
        "1. Prompt Engineering": [
            "Be specific and clear in your prompts",
            "Provide examples (few-shot learning)",
            "Use system messages to set context",
            "Break complex tasks into smaller steps",
            "Experiment with different phrasings"
        ],
        
        "2. Memory Management": [
            "Use quantization (4-bit/8-bit) for large models",
            "Clear cache: torch.cuda.empty_cache()",
            "Use gradient checkpointing for training",
            "Consider model sharding for multi-GPU",
            "Start with smaller models and scale up"
        ],
        
        "3. Performance Optimization": [
            "Use mixed precision (float16) when possible",
            "Batch multiple requests together",
            "Cache tokenized inputs for repeated prompts",
            "Use attention mask for variable-length sequences",
            "Consider model distillation for deployment"
        ],
        
        "4. Quality Control": [
            "Set max_length to prevent runaway generation",
            "Use temperature to control randomness",
            "Implement content filtering",
            "Validate outputs programmatically",
            "Use multiple samples and select best"
        ],
        
        "5. Deployment": [
            "Use model serving frameworks (TorchServe, TensorRT)",
            "Implement caching for common queries",
            "Monitor latency and throughput",
            "Consider API services (OpenAI, Anthropic) for production",
            "Version control your models and prompts"
        ],
        
        "6. Cost Management": [
            "Start with free/open-source models",
            "Use smaller models for simpler tasks",
            "Implement request batching",
            "Cache frequent responses",
            "Monitor API usage and costs"
        ]
    }
    
    for category, tips in practices.items():
        print(f"\n{category}")
        for tip in tips:
            print(f"   • {tip}")


if __name__ == "__main__":
    # Run the simplified LLM demonstration
    main()
    
    # Run Hugging Face and LLaMA demonstrations
    print("\n\n")
    huggingface_llama_demo()
    
    # Show advanced features
    huggingface_advanced_features()
    
    # Show practical applications
    practical_llm_applications()
    
    # Show best practices
    llm_best_practices()
    
    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\n📚 Additional Resources:")
    print("   • Hugging Face Hub: https://huggingface.co/models")
    print("   • LLaMA Models: https://huggingface.co/meta-llama")
    print("   • Transformers Docs: https://huggingface.co/docs/transformers")
    print("   • Model Cards: Check each model's page for usage examples")
    print("\n💡 Next Steps:")
    print("   1. Install transformers: pip install transformers torch")
    print("   2. Try GPT-2 or DistilGPT-2 (no authentication needed)")
    print("   3. Experiment with different parameters")
    print("   4. Request access to LLaMA models if needed")
    print("   5. Fine-tune models on your specific use case")
    print()
