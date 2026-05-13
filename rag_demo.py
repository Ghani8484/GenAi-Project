

import argparse
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase2_rag_integration import (
    ECommerceRAGSystem,
    StructuredResponse,
    DocumentIngestion,
    EmbeddingModels,
    VectorStore,
    QueryRouter,
    AnswerGenerator,
    RAGEvaluator
)


class RAGDemo:
    """Interactive demonstration of the E-Commerce RAG system."""
    
    def __init__(self):
        self.rag_system = None
        self.results = []
        
    def setup_system(self, config: Dict[str, Any] = None) -> ECommerceRAGSystem:
        """Initialize the RAG system with optional configuration."""
        if config is None:
            config = {
                "embedding_model_name": "all-MiniLM-L6-v2",
                "llm_model_name": "t5-small",
                "chunk_size": 256,
                "top_k_retrieval": 3
            }
        
        print("🚀 Initializing E-Commerce RAG System...")
        print(f"   Embedding Model: {config['embedding_model_name']}")
        print(f"   LLM Model: {config['llm_model_name']}")
        print(f"   Chunk Size: {config['chunk_size']}")
        print(f"   Top K Retrieval: {config['top_k_retrieval']}")
        print()
        
        self.rag_system = ECommerceRAGSystem(**config)
        return self.rag_system
    
    def ingest_documents(self):
        """Ingest and index all documents."""
        print("📚 Ingesting documents...")
        print("   Loading policy documents...")
        print("   Loading product manuals...")
        print("   Creating embeddings...")
        print()
        
        self.rag_system.ingest_documents()
        print("✅ Document ingestion completed!")
        print()
    
    def run_test_queries(self, queries: List[str] = None) -> List[Dict[str, Any]]:
        """Run a set of test queries and return results."""
        if queries is None:
            queries = [
                "How long do I have to return a product?",
                "What is the warranty period for electronics?",
                "How do I connect my phone to WiFi?",
                "Is water damage covered by warranty?",
                "How long does shipping take?",
                "How do I perform a factory reset on my laptop?",
                "What payment methods do you accept?",
                "Can I return an opened product?",
                "How do I update my phone's software?",
                "What should I do if my laptop screen is flickering?"
            ]
        
        results = []
        
        print("🧪 Running Test Queries")
        print("=" * 80)
        print()
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: {query}")
            print("-" * 80)
            
            try:
                response = self.rag_system.query(query)
                
                result = {
                    "query": query,
                    "response": {
                        "answer": response.answer,
                        "query_type": response.query_type,
                        "eligibility": response.eligibility,
                        "time_window": response.time_window,
                        "required_steps": response.required_steps,
                        "citations": response.citations,
                        "confidence": response.confidence
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                self._print_response(response)
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            print()
        
        return results
    
    def _print_response(self, response: StructuredResponse):
        """Pretty print a structured response."""
        print(f"🤖 Answer: {response.answer}")
        print(f"📋 Query Type: {response.query_type}")
        print(f"✅ Eligibility: {response.eligibility or 'N/A'}")
        print(f"⏰ Time Window: {response.time_window or 'N/A'}")
        
        if response.required_steps:
            print(f"📝 Required Steps:")
            for i, step in enumerate(response.required_steps, 1):
                print(f"   {i}. {step}")
        
        if response.citations:
            print(f"📚 Citations ({len(response.citations)}):")
            for i, citation in enumerate(response.citations, 1):
                print(f"   [{i}] {citation}")
        
        print(f"🎯 Confidence: {response.confidence:.2f}")
    
    def interactive_mode(self):
        """Run interactive mode for testing custom queries."""
        print("💬 Interactive Mode")
        print("Type 'quit' or 'exit' to stop, 'help' for commands")
        print("=" * 80)
        print()
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if query.lower() == 'help':
                    self._print_help()
                    continue
                
                if not query:
                    continue
                
                print("🤖 Processing...")
                response = self.rag_system.query(query)
                print()
                self._print_response(response)
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
    
    def _print_help(self):
        """Print help information for interactive mode."""
        print("""
Available Commands:
  help     - Show this help message
  quit/exit/q - Exit interactive mode
  
Example Queries:
  - "How long do I have to return a product?"
  - "What is the warranty period?"
  - "How do I connect to WiFi?"
  - "What should I do if my device is overheating?"
  - "Can I return an opened product?"
        """)
    
    def evaluate_system(self) -> Dict[str, float]:
        """Run system evaluation and return results."""
        print("📊 Running System Evaluation")
        print("=" * 80)
        print()
        
        results = self.rag_system.evaluate_system()
        
        print()
        print("📈 Evaluation Summary:")
        print(f"   Overall Score: {results['overall_score']:.3f}")
        print(f"   Answer Accuracy: {results['answer_accuracy']:.3f}")
        print(f"   Citation Precision: {results['citation_precision']:.3f}")
        print(f"   Eligibility Accuracy: {results['eligibility_accuracy']:.3f}")
        print(f"   Time Window Accuracy: {results['time_window_accuracy']:.3f}")
        
        return results
    
    def export_results(self, filename: str = None):
        """Export all results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_demo_results_{timestamp}.json"
        
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "embedding_model": self.rag_system.embedding_model.model_name if self.rag_system else None,
                    "llm_model": self.rag_system.answer_generator.model_name if self.rag_system else None,
                }
            },
            "results": self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results exported to: {filename}")
        return filename
    
    def run_component_tests(self):
        """Run tests for individual components."""
        print("🔧 Component Tests")
        print("=" * 80)
        print()
        
        # Test Document Ingestion
        print("1. Testing Document Ingestion...")
        ingestion = DocumentIngestion(chunk_size=256)
        chunks = ingestion.get_all_chunks()
        policy_chunks = [c for c in chunks if c.doc_type == "policy"]
        manual_chunks = [c for c in chunks if c.doc_type == "manual"]
        
        print(f"   ✅ Total chunks: {len(chunks)}")
        print(f"   ✅ Policy chunks: {len(policy_chunks)}")
        print(f"   ✅ Manual chunks: {len(manual_chunks)}")
        print()
        
        # Test Embedding Models
        print("2. Testing Embedding Models...")
        try:
            embed_model = EmbeddingModels("all-MiniLM-L6-v2")
            test_texts = ["This is a test document.", "Another test document."]
            embeddings = embed_model.encode_documents(test_texts)
            print(f"   ✅ Embedding shape: {embeddings.shape}")
            print(f"   ✅ Embedding model: {embed_model.model_name}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()
        
        # Test Vector Store
        print("3. Testing Vector Store...")
        try:
            vector_store = VectorStore(embedding_dim=384)
            test_embeddings = np.random.randn(10, 384).astype(np.float32)
            test_chunks = [DocumentChunk(
                text=f"Test chunk {i}",
                doc_id=f"test_{i}",
                doc_type="policy",
                chunk_id=i,
                source="Test"
            ) for i in range(10)]
            
            vector_store.add_documents(test_chunks, test_embeddings)
            print(f"   ✅ Vector store initialized")
            print(f"   ✅ Documents added: {len(test_chunks)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()
        
        # Test Query Router
        print("4. Testing Query Router...")
        router = QueryRouter()
        test_queries = [
            ("How do I return an item?", "policy"),
            ("How do I connect to WiFi?", "manual"),
            ("What is the warranty period?", "policy")
        ]
        
        for query, expected in test_queries:
            result = router.route_query(query)
            status = "✅" if result == expected else "❌"
            print(f"   {status} '{query}' -> {result} (expected: {expected})")
        print()
        
        print("🎉 Component tests completed!")


def main():
    """Main function to run the demonstration."""
    parser = argparse.ArgumentParser(description="E-Commerce RAG System Demo")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive mode")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run system evaluation")
    parser.add_argument("--export-results", action="store_true",
                       help="Export results to JSON")
    parser.add_argument("--component-tests", action="store_true",
                       help="Run component tests")
    parser.add_argument("--config", type=str,
                       help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = RAGDemo()
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"📋 Loaded configuration from {args.config}")
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")
    
    # Run component tests if requested
    if args.component_tests:
        demo.run_component_tests()
        print()
    
    # Setup and run main demo
    demo.setup_system(config)
    demo.ingest_documents()
    
    # Run test queries
    results = demo.run_test_queries()
    demo.results.extend(results)
    
    # Run evaluation if requested
    if args.evaluate:
        eval_results = demo.evaluate_system()
        demo.results.append({
            "type": "evaluation",
            "results": eval_results,
            "timestamp": datetime.now().isoformat()
        })
        print()
    
    # Run interactive mode if requested
    if args.interactive:
        demo.interactive_mode()
        print()
    
    # Export results if requested
    if args.export_results:
        demo.export_results()
        print()
    
    print("🎉 Demo completed successfully!")


if __name__ == "__main__":
    # If no arguments provided, run with defaults
    if len(sys.argv) == 1:
        print("Running default demo mode...")
        print("Use --help for more options")
        print()
        
        demo = RAGDemo()
        demo.setup_system()
        demo.ingest_documents()
        demo.run_test_queries()
        print("🎉 Demo completed!")
    else:
        main()
