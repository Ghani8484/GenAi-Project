# Phase 2: E-Commerce Support Assistant with RAG Integration

## Overview

This implementation integrates a complete **Retrieval-Augmented Generation (RAG)** system into your E-Commerce Support Assistant project. The system builds upon the transformer implementations from Phase 1 and adds sophisticated retrieval, generation, and evaluation capabilities.

## Features Implemented

### ✅ Core RAG Components

1. **Document Ingestion & Chunking**
   - Automatic chunking with metadata preservation
   - Support for policy documents and product manuals
   - Overlapping chunks to maintain context
   - Metadata tracking (doc_type, source, section)

2. **Embeddings & Vector Store**
   - Multiple embedding model support (Sentence Transformers, TF-IDF fallback)
   - FAISS integration for efficient similarity search
   - Sklearn cosine similarity fallback
   - Configurable embedding dimensions

3. **Query Router**
   - Intent classification using encoder-only transformer from Phase 1
   - Automatic routing to policy or manual document corpora
   - Trained on comprehensive e-commerce query dataset

4. **Answer Generation with Citations**
   - Support for multiple LLMs (T5, GPT-2, custom models)
   - Citation extraction and formatting
   - Rule-based fallback for environments without transformers
   - Context-aware answer generation

5. **Structured Output Format**
   ```python
   {
       "answer": "Generated response",
       "eligibility": "yes/no/conditional",
       "required_steps": ["Step 1", "Step 2"],
       "time_window": "30 days",
       "citations": ["Source 1", "Source 2"],
       "confidence": 0.85,
       "query_type": "policy/manual"
   }
   ```

6. **Evaluation Framework**
   - Gold dataset with expected answers and citations
   - Multiple evaluation metrics:
     - Answer accuracy (keyword matching)
     - Citation precision
     - Eligibility accuracy
     - Time window accuracy
     - Overall weighted score

### ✅ Advanced Features

7. **Multiple LLM Support**
   - T5-small/flan-t5-base
   - DistilGPT2/GPT-2
   - Phi-2
   - Custom encoder-decoder from Phase 1
   - API-based LLM support

8. **Optional Fine-tuning Support**
   - LoRA (Parameter-Efficient Fine-Tuning)
   - QLoRA with 4-bit quantization
   - BitsAndBytes integration
   - PEFT compatibility

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Router    │───▶│  Intent         │
│                 │    │  (Classification)│    │  Classification │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Retrieved      │◀───│  Vector Store    │◀───│  Query          │
│  Chunks         │    │  (FAISS/         │    │  Embedding      │
│                 │    │   Similarity)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                              │
        │                                              │
        ▼                                              │
┌─────────────────┐    ┌──────────────────┐           │
│  Context        │───▶│  Answer          │           │
│  Assembly       │    │  Generator       │           │
│                 │    │  (LLM + Rules)   │           │
└─────────────────┘    └──────────────────┘           │
                               │                      │
                               ▼                      │
┌─────────────────┐    ┌──────────────────┐           │
│  Structured     │◀───│  Response        │◀──────────┘
│  Output         │    │  Formatter       │
│                 │    │                  │
└─────────────────┘    └──────────────────┘
```

## Installation

### Required Dependencies

```bash
# Core dependencies (required)
pip install torch numpy scikit-learn

# For advanced features (optional but recommended)
pip install sentence-transformers faiss-cpu transformers

# For fine-tuning (optional)
pip install peft bitsandbytes

# For PDF processing (if extending to PDF manuals)
pip install PyPDF2 pdfplumber
```

### Fallback System

The implementation includes intelligent fallbacks:
- **No Sentence Transformers** → Uses TF-IDF embeddings
- **No FAISS** → Uses sklearn cosine similarity
- **No transformers** → Uses rule-based generation
- **No GPU** → Automatically uses CPU

## Usage

### Basic Usage

```python
from phase2_rag_integration import ECommerceRAGSystem

# Initialize the RAG system
rag_system = ECommerceRAGSystem(
    embedding_model_name="all-MiniLM-L6-v2",
    llm_model_name="t5-small",
    chunk_size=256,
    top_k_retrieval=3
)

# Ingest documents (required before querying)
rag_system.ingest_documents()

# Query the system
response = rag_system.query("How long do I have to return a product?")

# Access structured response
print(f"Answer: {response.answer}")
print(f"Eligibility: {response.eligibility}")
print(f"Time Window: {response.time_window}")
print(f"Citations: {response.citations}")
```

### Complete Example

```python
# Initialize system with custom configuration
rag = ECommerceRAGSystem(
    embedding_model_name="all-MiniLM-L6-v2",  # Small, efficient model
    llm_model_name="flan-t5-base",           # Better instruction following
    chunk_size=512,                           # Larger chunks for more context
    top_k_retrieval=5                        # Retrieve more chunks
)

# Ingest all documents (policies + manuals)
rag.ingest_documents()

# Test different types of queries
queries = [
    "How long do I have to return a product?",
    "What is the warranty period for electronics?",
    "How do I connect my phone to WiFi?",
    "Is water damage covered by warranty?",
    "How long does shipping take?"
]

for query in queries:
    response = rag.query(query)
    print(f"\nQuery: {query}")
    print(f"Answer: {response.answer}")
    print(f"Type: {response.query_type}")
    print(f"Eligibility: {response.eligibility}")
    print(f"Time: {response.time_window}")
    print(f"Citations: {len(response.citations)}")

# Evaluate system performance
results = rag.evaluate_system()
print(f"Overall Score: {results['overall_score']:.3f}")
```

## Document Types

### Policy Documents

The system includes sample policy documents for:
- **Returns and Refunds Policy**
  - Return window (30 days standard, 14 days for electronics)
  - Return process and requirements
  - Refund timeline
  - Exceptions and restrictions

- **Product Warranty Terms**
  - Standard warranty (1 year)
  - Coverage details
  - Exclusions
  - Extended warranty options

- **Shipping and Delivery Policy**
  - Shipping options and costs
  - International shipping
  - Processing times
  - Delivery issues

### Product Manuals

Sample manuals included:
- **Smartphone User Manual**
  - Initial setup
  - WiFi connection
  - Factory reset
  - Battery optimization
  - Troubleshooting

- **Laptop User Manual**
  - First time setup
  - System recovery
  - Performance issues
  - Display problems

## Evaluation Metrics

The system evaluates performance on multiple dimensions:

### 1. Answer Accuracy (40% weight)
Measures how well the generated answer matches the expected response using keyword overlap.

### 2. Citation Precision (30% weight)
Measures whether the correct sources are cited in the response.

### 3. Eligibility Accuracy (20% weight)
Checks if the system correctly identifies eligibility (yes/no/conditional).

### 4. Time Window Accuracy (10% weight)
Verifies correct extraction of time-related information.

### Overall Score
Weighted average of all metrics, providing a comprehensive evaluation.

## Model Options

### Embedding Models

| Model | Size | Performance | Use Case |
|-------|------|-------------|----------|
| `all-MiniLM-L6-v2` | 22MB | Fast | General purpose |
| `all-MiniLM-L12-v2` | 44MB | Better | Higher accuracy |
| `all-mpnet-base-v2` | 438MB | Best | Production use |

### LLM Models

| Model | Size | Performance | Use Case |
|-------|------|-------------|----------|
| `t5-small` | 60MB | Fast | Quick responses |
| `flan-t5-base` | 250MB | Better | Instruction following |
| `phi-2` | 2.7GB | Best | Complex reasoning |
| API-based | - | Variable | No local compute |

## Fine-tuning with LoRA/QLoRA

### Setup

```python
# Check if PEFT is available
from phase2_rag_integration import HAS_PEFT, HAS_BNB

if HAS_PEFT and HAS_BNB:
    print("LoRA/QLoRA fine-tuning is available")
else:
    print("Install peft and bitsandbytes for fine-tuning")
```

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=16,                # Rank
    lora_alpha=32,       # Alpha parameter
    target_modules=["q", "v"],  # Target attention layers
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Apply to model
model = get_peft_model(model, lora_config)
```

### QLoRA Configuration

```python
# 4-bit quantization for memory efficiency
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

## Extending the System

### Adding Custom Documents

```python
class CustomDocumentIngestion(DocumentIngestion):
    def load_custom_documents(self):
        # Your custom document loading logic
        custom_docs = [
            {
                "id": "custom_policy",
                "title": "Custom Policy",
                "sections": [
                    {"section": "Section 1", "text": "Your content here"}
                ]
            }
        ]
        return self._process_documents(custom_docs, doc_type="policy")
```

### Custom Query Routing

```python
class CustomQueryRouter(QueryRouter):
    def route_query(self, query: str) -> str:
        # Your custom routing logic
        if "specific_keyword" in query.lower():
            return "custom_corpus"
        return super().route_query(query)
```

### Custom Answer Generation

```python
class CustomAnswerGenerator(AnswerGenerator):
    def _extract_structured_info(self, answer: str, query: str) -> Dict[str, Any]:
        # Your custom extraction logic
        structured_info = super()._extract_structured_info(answer, query)
        # Add custom extractions
        return structured_info
```

## Performance Optimization

### 1. Embedding Model Selection
- Use smaller models (`all-MiniLM-L6-v2`) for faster inference
- Use larger models (`all-mpnet-base-v2`) for better accuracy
- Cache embeddings for repeated queries

### 2. Vector Store Optimization
- Use FAISS for large-scale deployment
- Implement approximate nearest neighbors for speed
- Consider sharding for very large datasets

### 3. LLM Optimization
- Use smaller models for quick responses
- Implement response caching
- Use quantization (4-bit/8-bit) for memory efficiency

### 4. Chunking Strategy
- Adjust chunk size based on document structure
- Use overlap to maintain context
- Implement semantic chunking for better boundaries

## Error Handling

The system includes comprehensive error handling:

```python
try:
    response = rag_system.query(query)
except ValueError as e:
    print(f"Input error: {e}")
except RuntimeError as e:
    print(f"System error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Monitoring and Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Monitor performance
import time

start_time = time.time()
response = rag_system.query(query)
latency = time.time() - start_time
print(f"Query latency: {latency:.2f}s")
```

## Testing

### Unit Tests

```python
def test_document_ingestion():
    ingestion = DocumentIngestion(chunk_size=256)
    chunks = ingestion.get_all_chunks()
    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

def test_query_routing():
    router = QueryRouter()
    doc_type = router.route_query("How do I return an item?")
    assert doc_type in ["policy", "manual"]

def test_answer_generation():
    generator = AnswerGenerator()
    chunks = [RetrievedChunk(chunk=DocumentChunk(...), score=0.9)]
    response = generator.generate_answer("Test query", chunks)
    assert response.answer is not None
```

### Integration Tests

```python
def test_full_pipeline():
    rag = ECommerceRAGSystem()
    rag.ingest_documents()
    response = rag.query("How long is the warranty?")
    
    assert response.answer is not None
    assert response.eligibility is not None
    assert len(response.citations) > 0
```

## Deployment Considerations

### 1. Scalability
- Use distributed vector stores for large datasets
- Implement load balancing for high traffic
- Consider microservices architecture

### 2. Security
- Sanitize user inputs
- Implement rate limiting
- Secure API endpoints

### 3. Monitoring
- Track query latency
- Monitor accuracy metrics
- Log user feedback

### 4. Maintenance
- Regular model updates
- Document corpus updates
- Performance monitoring

## Future Enhancements

1. **Multi-modal Support**
   - Image-based queries
   - Voice input processing
   - Video manual integration

2. **Advanced Retrieval**
   - Hybrid search (keyword + semantic)
   - Query expansion
   - Re-ranking with cross-encoders

3. **Personalization**
   - User context awareness
   - Query history
   - Preference learning

4. **Real-time Updates**
   - Live document updates
   - Dynamic policy changes
   - Version control

## Conclusion

This RAG implementation provides a complete, production-ready system for e-commerce customer support. It combines the transformer knowledge from Phase 1 with modern retrieval and generation techniques to create a sophisticated support assistant.

The system is designed to be:
- **Modular**: Easy to extend and customize
- **Robust**: Multiple fallback mechanisms
- **Scalable**: Supports various model sizes and architectures
- **Evaluable**: Comprehensive evaluation framework
- **Practical**: Real-world e-commerce use cases

## Support

For questions or issues, please refer to:
- Code documentation and comments
- Example usage in `main()` function
- Evaluation results for performance insights
- Architecture diagrams for system understanding
