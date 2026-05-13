# Quick Start Guide: E-Commerce RAG System

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Install core dependencies
pip install torch numpy scikit-learn

# Install recommended packages for full functionality
pip install sentence-transformers faiss-cpu transformers

# Optional: For fine-tuning capabilities
pip install peft bitsandbytes
```

### Step 2: Run the Demo

```bash
# Run the complete demonstration
python rag_demo.py

# With all features
python rag_demo.py --interactive --evaluate --export-results
```

### Step 3: Test Your Own Queries

```bash
# Interactive mode for testing queries
python rag_demo.py --interactive
```

## 📋 Basic Usage Example

```python
from phase2_rag_integration import ECommerceRAGSystem

# Create RAG system
rag = ECommerceRAGSystem()

# Ingest documents (one-time setup)
rag.ingest_documents()

# Ask questions
response = rag.query("How long do I have to return a product?")

# Access structured response
print(f"Answer: {response.answer}")
print(f"Eligibility: {response.eligibility}")
print(f"Time Window: {response.time_window}")
print(f"Citations: {len(response.citations)}")
```

## 🎯 What You Get

### 1. Complete RAG Pipeline
- Document ingestion and chunking
- Semantic search with embeddings
- Intent-based query routing
- Answer generation with citations
- Structured output format

### 2. Sample Data Included
- **Policy Documents**: Returns, warranty, shipping policies
- **Product Manuals**: Smartphone and laptop manuals
- **Evaluation Dataset**: Test queries with expected answers

### 3. Evaluation Framework
- Answer accuracy measurement
- Citation precision tracking
- Eligibility classification accuracy
- Time window extraction accuracy
- Overall system scoring

## 🔧 Configuration Options

### Basic Configuration

```python
# Fast and lightweight
rag = ECommerceRAGSystem(
    embedding_model_name="all-MiniLM-L6-v2",
    llm_model_name="t5-small",
    chunk_size=256,
    top_k_retrieval=3
)

# Better accuracy (requires more resources)
rag = ECommerceRAGSystem(
    embedding_model_name="all-mpnet-base-v2",
    llm_model_name="flan-t5-base",
    chunk_size=512,
    top_k_retrieval=5
)
```

### Using Configuration File

```json
// config.json
{
  "embedding_model_name": "all-MiniLM-L6-v2",
  "llm_model_name": "flan-t5-base",
  "chunk_size": 256,
  "top_k_retrieval": 3
}
```

```python
import json
with open('config.json', 'r') as f:
    config = json.load(f)

rag = ECommerceRAGSystem(**config)
```

## 📊 Understanding the Output

### Structured Response Format

```python
{
    "answer": "You can return most items within 30 days...",
    "eligibility": "yes",
    "required_steps": ["Log into account", "Select return option"],
    "time_window": "30 days",
    "citations": [
        "Returns and Refunds Policy - Return Window (Score: 0.89)"
    ],
    "confidence": 0.85,
    "query_type": "policy"
}
```

### Field Descriptions

| Field | Description | Example |
|-------|-------------|---------|
| `answer` | Generated response | "You can return within 30 days..." |
| `eligibility` | Yes/No/Conditional | "yes", "no", "conditional" |
| `required_steps` | Action items | ["Log in", "Select option"] |
| `time_window` | Time-related info | "30 days", "1 year" |
| `citations` | Source documents | ["Policy - Section"] |
| `confidence` | System confidence | 0.0 to 1.0 |
| `query_type` | Document type | "policy", "manual" |

## 🎮 Interactive Commands

When running in interactive mode (`--interactive`):

```
Available Commands:
  help     - Show all available commands
  quit/exit/q - Stop the interactive session

Example Queries:
  - "How do I return an item?"
  - "What's the warranty on electronics?"
  - "How do I connect to WiFi?"
  - "My device is overheating, what should I do?"
```

## 📈 Evaluation Metrics

The system automatically evaluates performance on:

1. **Answer Accuracy** (40%): How well answers match expected responses
2. **Citation Precision** (30%): Correctness of source attribution
3. **Eligibility Accuracy** (20%): Correct yes/no/conditional classification
4. **Time Window Accuracy** (10%): Correct extraction of time information

```bash
# Run evaluation
python rag_demo.py --evaluate

# Example output
Answer Accuracy: 0.850
Citation Precision: 0.920
Eligibility Accuracy: 0.900
Time Window Accuracy: 0.875
Overall Score: 0.881
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt
```

#### 2. Memory Issues
```python
# Use smaller models
rag = ECommerceRAGSystem(
    embedding_model_name="all-MiniLM-L6-v2",  # Smaller model
    chunk_size=128,                           # Smaller chunks
    top_k_retrieval=2                         # Fewer retrieved chunks
)
```

#### 3. Slow Performance
```python
# Reduce retrieval count
rag = ECommerceRAGSystem(top_k_retrieval=2)

# Or use faster models
rag = ECommerceRAGSystem(
    embedding_model_name="all-MiniLM-L6-v2",
    llm_model_name="t5-small"  # Faster than flan-t5-base
)
```

#### 4. GPU Not Available
The system automatically falls back to CPU if GPU is not available. No action needed.

## 🔍 System Architecture

### Core Components

1. **Document Ingestion** (`DocumentIngestion`)
   - Loads policy and manual documents
   - Chunks documents with metadata
   - Handles overlapping chunks

2. **Embedding Models** (`EmbeddingModels`)
   - Encodes text to vectors
   - Supports multiple models
   - Has fallback to TF-IDF

3. **Vector Store** (`VectorStore`)
   - Stores document embeddings
   - Performs similarity search
   - Uses FAISS or sklearn

4. **Query Router** (`QueryRouter`)
   - Classifies query intent
   - Routes to appropriate corpus
   - Uses encoder-only transformer

5. **Answer Generator** (`AnswerGenerator`)
   - Generates answers from context
   - Extracts structured information
   - Supports multiple LLMs

6. **RAG System** (`ECommerceRAGSystem`)
   - Orchestrates all components
   - Provides unified interface
   - Handles evaluation

## 📝 Code Structure

```
phase2_rag_integration.py    # Main RAG implementation
rag_demo.py                  # Interactive demonstration
config.json                  # Configuration template
requirements.txt             # Python dependencies
README_RAG_Integration.md    # Detailed documentation
QUICK_START.md              # This file
```

## 🎓 Next Steps

### 1. Customize for Your Data

```python
# Add your own documents
class MyDocumentIngestion(DocumentIngestion):
    def load_custom_documents(self):
        # Your document loading logic
        return self._process_documents(my_docs, doc_type="custom")
```

### 2. Fine-tune Models

```python
# Use LoRA for efficient fine-tuning
if HAS_PEFT and HAS_BNB:
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1
    )
```

### 3. Deploy to Production

```python
# Add monitoring and logging
import logging
logging.basicConfig(level=logging.INFO)

# Add caching for repeated queries
from functools import lru_cache

# Add rate limiting
import time
```

## 🤝 Contributing

To extend the system:

1. **Add New Document Types**: Extend `DocumentIngestion`
2. **Improve Routing**: Extend `QueryRouter`
3. **Enhance Generation**: Extend `AnswerGenerator`
4. **Add Metrics**: Extend `RAGEvaluator`

## 📚 Additional Resources

- [README_RAG_Integration.md](README_RAG_Integration.md) - Complete documentation
- [Phase 1 Code](phase1_transformers.py) - Original transformer implementations
- [Config Template](config.json) - Configuration options
- [Demo Script](rag_demo.py) - Interactive examples

## ✅ Checklist for Your Project

- [ ] Install dependencies
- [ ] Run basic demo
- [ ] Test with your data
- [ ] Evaluate performance
- [ ] Customize as needed
- [ ] Deploy to production

## 🆘 Support

### Common Questions

**Q: Can I use this with my own documents?**
A: Yes! Extend the `DocumentIngestion` class to load your documents.

**Q: Do I need a GPU?**
A: No, the system works on CPU, but GPU will be faster for large models.

**Q: Can I use different LLMs?**
A: Yes, the system supports any HuggingFace transformer model.

**Q: How do I improve accuracy?**
A: Try larger models, increase chunk size, or fine-tune on your data.

**Q: Is this production-ready?**
A: The architecture is production-ready, but you'll need to add monitoring, logging, and error handling for your specific use case.

---

**Ready to go? Run `python rag_demo.py` to see the system in action! 🎉**
