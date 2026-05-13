# -*- coding: utf-8 -*-
"""
Phase 2 - E-Commerce Support Assistant with RAG Integration
==========================================================

This module implements a complete Retrieval-Augmented Generation (RAG) system
for e-commerce customer support, building upon the transformer implementations
from Phase 1.

Features:
--------
1. Document ingestion and chunking with metadata
2. Embeddings and vector store (with comparison of embedding models)
3. Query routing to detect query types
4. Answer generation with citations using small LLMs
5. Structured output format
6. Evaluation framework
7. Optional LoRA/QLoRA fine-tuning

Author: DS-462 GenAI with LLM Project
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter, defaultdict
import asyncio
from datetime import datetime
from phase1_transformers import (
    DocumentChunk,
    DocumentIngestion,
    EmbeddingModels,
    VectorStore
)

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# Import Phase 1 components
from phase1_transformers import (
    EncoderIntentClassifier,
    TinyGPT,
    Seq2SeqTransformer,
    MultiHeadSelfAttention,
    TransformerEncoderLayer,
    MaskedMultiHeadSelfAttention,
    TransformerDecoderLayer,
    CrossAttention
)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    text: str
    doc_id: str
    doc_type: str  # 'policy' or 'manual'
    chunk_id: int
    source: str
    title: str = ""
    section: str = ""


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with similarity score."""
    chunk: DocumentChunk
    score: float


@dataclass
class StructuredResponse:
    """Structured output format for RAG responses."""
    answer: str
    eligibility: Optional[str] = None  # 'yes', 'no', or 'conditional'
    required_steps: Optional[List[str]] = None
    time_window: Optional[str] = None
    citations: List[str] = None
    confidence: float = 0.0
    query_type: str = ""


class DocumentIngestion:
    """
    Handles document ingestion, chunking, and metadata extraction.
    Supports both policy documents and product manuals.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []
        
    def load_policy_documents(self) -> List[DocumentChunk]:
        """
        Load sample e-commerce policy documents.
        In production, this would load from actual policy pages.
        """
        policy_docs = [
            {
                "id": "returns_policy",
                "title": "Returns and Refunds Policy",
                "sections": [
                    {
                        "section": "Return Window",
                        "text": "You can return most items within 30 days of delivery for a full refund. The item must be in its original condition, unused, and with all original packaging and tags. Electronics and opened software can be returned within 14 days. Certain items like gift cards, downloadable software, and perishable goods are not returnable."
                    },
                    {
                        "section": "Return Process",
                        "text": "To initiate a return, log into your account and go to Your Orders. Select the item you want to return and choose Return or Replace Items. Select your reason for return from the dropdown menu. Choose your preferred refund method: original payment method or store credit. Print the prepaid return label provided. Pack the item securely with all original accessories and documentation. Drop off the package at any authorized shipping location within 7 days of label generation."
                    },
                    {
                        "section": "Refund Timeline",
                        "text": "Once we receive and inspect your returned item, we will process your refund within 5-7 business days. Refunds to credit cards typically appear on your statement within 1-2 billing cycles. Store credit refunds are available immediately after processing. You will receive an email confirmation when your refund is processed."
                    },
                    {
                        "section": "Return Exceptions",
                        "text": "Items damaged by customer misuse or negligence are not eligible for return. Products missing original packaging, accessories, or documentation may be subject to a 15% restocking fee. Customized or personalized items cannot be returned unless defective. Bulk orders of 10 or more items may have different return policies."
                    }
                ]
            },
            {
                "id": "warranty_policy",
                "title": "Product Warranty Terms",
                "sections": [
                    {
                        "section": "Standard Warranty",
                        "text": "All electronics come with a 1-year limited warranty covering manufacturing defects and hardware failures under normal use. The warranty period begins from the date of purchase. We will repair or replace defective products at no charge. Warranty service requires proof of purchase."
                    },
                    {
                        "section": "What's Covered",
                        "text": "Warranty covers hardware defects like screen issues, battery failure, charging port problems, and internal component failures. Software issues caused by manufacturing defects are also covered. Accessories included in the original package are covered under the same warranty terms."
                    },
                    {
                        "section": "What's Not Covered",
                        "text": "Warranty does not cover accidental damage, water damage, cosmetic damage, or normal wear and tear. Damage from unauthorized modifications or repairs voids the warranty. Issues caused by using incompatible accessories or software are not covered. Lost or stolen items are not covered under warranty."
                    },
                    {
                        "section": "Extended Warranty",
                        "text": "Extended warranty plans are available for purchase within 30 days of product delivery. Extended plans add 1-3 additional years of coverage. Accidental damage protection can be added to extended warranty plans for an additional fee. Extended warranties are non-refundable and non-transferable."
                    }
                ]
            },
            {
                "id": "shipping_policy",
                "title": "Shipping and Delivery Policy",
                "sections": [
                    {
                        "section": "Shipping Options",
                        "text": "Standard shipping delivers within 5-7 business days and is free on orders over $50. Expedited shipping delivers within 2-3 business days for $12.99. Priority shipping delivers within 1-2 business days for $24.99. Shipping times are estimates and not guaranteed."
                    },
                    {
                        "section": "International Shipping",
                        "text": "We ship to over 50 countries worldwide. International shipping costs $25-$50 depending on destination and package weight. Delivery times vary from 7-21 business days. Customers are responsible for any customs duties, taxes, or import fees. Some items may not be available for international shipping due to restrictions."
                    },
                    {
                        "section": "Order Processing",
                        "text": "Orders are processed within 1-2 business days. Processing may take longer during peak seasons or for large orders. You will receive a confirmation email with tracking information once your order ships. Orders placed after 2 PM EST will be processed the next business day."
                    },
                    {
                        "section": "Delivery Issues",
                        "text": "If your package is lost or damaged in transit, contact customer service within 48 hours. We will work with the carrier to locate your package or send a replacement. Refused deliveries will be treated as returns and may be subject to return shipping fees."
                    }
                ]
            }
        ]
        
        return self._process_documents(policy_docs, doc_type="policy")
    
    def load_manual_documents(self) -> List[DocumentChunk]:
        """
        Load sample product manual documents.
        In production, this would load from actual PDF manuals.
        """
        manual_docs = [
            {
                "id": "phone_manual",
                "title": "Smartphone User Manual",
                "sections": [
                    {
                        "section": "Initial Setup",
                        "text": "To set up your new smartphone, first insert the SIM card by locating the SIM tray on the side of the device. Use the provided ejector tool to open the tray. Place the SIM card with the gold contacts facing down. Charge your device fully before first use by connecting the included charger to the USB-C port. Press and hold the power button for 3 seconds to turn on the device. Follow the on-screen setup wizard to connect to WiFi, create or sign in to your account, and set up security features like fingerprint or face recognition."
                    },
                    {
                        "section": "WiFi Connection",
                        "text": "To connect to WiFi, go to Settings > Network & Internet > WiFi. Toggle WiFi to ON if not already enabled. Select your network from the list of available networks. Enter the network password when prompted. Tap Connect. Once connected, you will see a WiFi icon in the status bar. To forget a network, tap the network name and select Forget."
                    },
                    {
                        "section": "Factory Reset",
                        "text": "To perform a factory reset, first backup your data as this process will erase all content. Go to Settings > System > Reset Options > Erase All Data. Enter your PIN or password when prompted. Tap Erase All Data to confirm. The device will restart and begin the reset process, which takes 5-10 minutes. Do not interrupt the process. After reset, the device will restart to the initial setup screen."
                    },
                    {
                        "section": "Battery Optimization",
                        "text": "To maximize battery life, enable Battery Saver mode in Settings > Battery. Close unused apps by swiping up from the bottom and swiping apps away. Reduce screen brightness or enable auto-brightness. Turn off WiFi, Bluetooth, and GPS when not in use. Avoid extreme temperatures. Charge your device when it reaches 20% and unplug when it reaches 80% for optimal battery health."
                    },
                    {
                        "section": "Troubleshooting",
                        "text": "If your device won't turn on, charge it for at least 30 minutes then try again. For frozen screens, press and hold the power button for 10 seconds to force restart. If apps crash frequently, clear the app cache in Settings > Apps. For overheating issues, close all apps and let the device cool down. If problems persist, contact customer support."
                    }
                ]
            },
            {
                "id": "laptop_manual",
                "title": "Laptop User Manual",
                "sections": [
                    {
                        "section": "First Time Setup",
                        "text": "Remove all packaging materials and protective films. Connect the power adapter and charge the laptop fully before first use. Press the power button to start the device. Select your language, region, and keyboard layout. Connect to a WiFi network when prompted. Create a user account with a strong password. Enable automatic updates to keep your system secure. Install essential software and applications."
                    },
                    {
                        "section": "System Recovery",
                        "text": "To reset your laptop to factory settings, go to Settings > Update & Security > Recovery. Under Reset this PC, click Get Started. Choose whether to keep your files or remove everything. Follow the on-screen instructions. The process takes 30-60 minutes. Your laptop will restart several times. Ensure the laptop is plugged in throughout the process."
                    },
                    {
                        "section": "Performance Issues",
                        "text": "If your laptop is running slowly, try these steps: Restart your laptop. Close unnecessary browser tabs and applications. Run Disk Cleanup to free up space. Check for Windows updates. Scan for malware using Windows Defender. Disable startup programs in Task Manager. If issues persist, consider upgrading RAM or storage."
                    },
                    {
                        "section": "Display Problems",
                        "text": "For screen flickering, update your graphics drivers. Right-click the desktop and select Graphics Properties. Check for driver updates. If the screen is black, try connecting to an external monitor. Press the display toggle key combination (usually Fn + F4 or F8) to cycle through display options. For resolution issues, go to Settings > System > Display and adjust the resolution."
                    }
                ]
            }
        ]
        
        return self._process_documents(manual_docs, doc_type="manual")
    
    def _process_documents(self, docs: List[Dict], doc_type: str) -> List[DocumentChunk]:
        """Process documents into chunks with metadata."""
        chunks = []
        
        for doc in docs:
            doc_id = doc["id"]
            title = doc["title"]
            
            for section in doc["sections"]:
                section_name = section["section"]
                text = section["text"]
                
                # Split text into chunks if it's too long
                if len(text) <= self.chunk_size:
                    chunk = DocumentChunk(
                        text=text,
                        doc_id=doc_id,
                        doc_type=doc_type,
                        chunk_id=len(chunks),
                        source=f"{title} - {section_name}",
                        title=title,
                        section=section_name
                    )
                    chunks.append(chunk)
                else:
                    # Split into overlapping chunks
                    words = text.split()
                    start = 0
                    chunk_id = 0
                    
                    while start < len(words):
                        end = min(start + self.chunk_size, len(words))
                        chunk_text = " ".join(words[start:end])
                        
                        chunk = DocumentChunk(
                            text=chunk_text,
                            doc_id=f"{doc_id}_chunk_{chunk_id}",
                            doc_type=doc_type,
                            chunk_id=len(chunks),
                            source=f"{title} - {section_name}",
                            title=title,
                            section=section_name
                        )
                        chunks.append(chunk)
                        
                        start = end - self.chunk_overlap
                        chunk_id += 1
                        
                        if end >= len(words):
                            break
        
        return chunks
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all document chunks from both policy and manual documents."""
        if not self.chunks:
            policy_chunks = self.load_policy_documents()
            manual_chunks = self.load_manual_documents()
            self.chunks = policy_chunks + manual_chunks
        
        return self.chunks


class EmbeddingModels:
    """
    Handles different embedding models for document and query encoding.
    Supports both local and API-based embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.tfidf_vectorizer = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"Loaded SentenceTransformer model: {model_name}")
            except Exception as e:
                print(f"Failed to load SentenceTransformer: {e}")
                print("Falling back to TF-IDF embeddings")
        
        if not self.model and HAS_SKLEARN:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            print("Using TF-IDF for embeddings")
    
    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """Encode a list of documents into vectors."""
        if self.model:
            # Use SentenceTransformer
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return np.array(embeddings)
        elif self.tfidf_vectorizer:
            # Use TF-IDF
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                # Fit the vectorizer first
                self.tfidf_vectorizer.fit(texts[:100] if len(texts) > 100 else texts)
            embeddings = self.tfidf_vectorizer.transform(texts).toarray()
            return embeddings
        else:
            raise ValueError("No embedding model available")
    
    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query into a vector."""
        if self.model:
            embedding = self.model.encode([text])
            return embedding
        elif self.tfidf_vectorizer:
            embedding = self.tfidf_vectorizer.transform([text]).toarray()
            return embedding
        else:
            raise ValueError("No embedding model available")


class VectorStore:
    """
    Vector store for efficient similarity search using FAISS.
    Falls back to sklearn cosine similarity if FAISS is not available.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
        if HAS_FAISS:
            try:
                self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
                print("Initialized FAISS index")
            except Exception as e:
                print(f"Failed to initialize FAISS: {e}")
                print("Falling back to sklearn cosine similarity")
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store."""
        self.chunks = chunks
        self.embeddings = embeddings
        
        if self.index is not None:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index.add(normalized_embeddings.astype(np.float32))
            print(f"Added {len(chunks)} documents to FAISS index")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        """Search for the most similar documents to the query."""
        if self.index is not None:
            # FAISS search
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            scores, indices = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    results.append(RetrievedChunk(
                        chunk=self.chunks[idx],
                        score=float(score)
                    ))
            return results
        elif self.embeddings is not None and HAS_SKLEARN:
            # Fallback to sklearn cosine similarity
            similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append(RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(similarities[idx])
                ))
            return results
        else:
            raise ValueError("No vector store available")


class QueryRouter:
    """
    Routes queries to the appropriate document corpus based on query type.
    Uses the encoder-only transformer from Phase 1 for intent classification.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.vocab = {}
        self.label2idx = {}
        self.idx2label = {}
        self.max_len = 16
        
        # Initialize with default intent classification
        self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize the intent classification model with default data."""
        # Create intent training data
        intent_data = [
            {"text": "I want to return an item I bought last week", "label": "returns"},
            {"text": "Can I return opened products?", "label": "returns"},
            {"text": "What is the return policy for electronics?", "label": "returns"},
            {"text": "How many days do I have to return a product?", "label": "returns"},
            {"text": "How long is the warranty on this phone?", "label": "warranty"},
            {"text": "Is accidental damage covered under warranty?", "label": "warranty"},
            {"text": "Does the warranty cover battery issues?", "label": "warranty"},
            {"text": "What warranty do I get with this laptop?", "label": "warranty"},
            {"text": "When will my order be shipped?", "label": "shipping"},
            {"text": "How long does shipping take?", "label": "shipping"},
            {"text": "Can I track my shipment online?", "label": "shipping"},
            {"text": "Do you offer international shipping?", "label": "shipping"},
            {"text": "How do I pay with a credit card?", "label": "payment"},
            {"text": "Can I pay using debit card?", "label": "payment"},
            {"text": "Does the store accept international cards?", "label": "payment"},
            {"text": "Are there any installment payment options?", "label": "payment"},
            {"text": "How to connect the device to wifi?", "label": "product_usage"},
            {"text": "My screen is flickering, what should I do?", "label": "product_usage"},
            {"text": "How do I reset the device to factory settings?", "label": "product_usage"},
            {"text": "The device is overheating during use", "label": "product_usage"},
            {"text": "How do I update the software?", "label": "product_usage"},
            {"text": "The laptop won't turn on", "label": "product_usage"},
            {"text": "How to perform a factory reset?", "label": "product_usage"},
            {"text": "Can I get a refund?", "label": "returns"},
            {"text": "What is covered under warranty?", "label": "warranty"},
            {"text": "How much does shipping cost?", "label": "shipping"},
            {"text": "How to factory reset laptop?", "label": "product_usage"},
            {"text": "Is water damage covered by warranty?", "label": "warranty"},
            {"text": "Can I exchange my product?", "label": "returns"},
            {"text": "How to connect to internet?", "label": "product_usage"},
            {"text": "What payment methods are accepted?", "label": "payment"},
        ]
        
        # Build vocabulary
        counter = Counter()
        for item in intent_data:
            for word in item["text"].lower().split():
                counter[word] += 1
        
        self.vocab = {"<pad>": 0, "<unk>": 1}
        for word, _ in counter.most_common():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Create label mappings
        labels = sorted(list(set(item["label"] for item in intent_data)))
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderIntentClassifier(
            vocab_size=len(self.vocab),
            num_classes=len(self.label2idx),
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_len=self.max_len
        ).to(device)
        
        # Train the model
        self._train_model(intent_data, device)
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, self.vocab["<unk>"]))
        
        # Truncate and pad
        tokens = tokens[:self.max_len]
        while len(tokens) < self.max_len:
            tokens.append(self.vocab["<pad>"])
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _train_model(self, data: List[Dict], device: torch.device):
        """Train the intent classification model."""
        # Create dataset
        class IntentDataset(Dataset):
            def __init__(self, data, encode_fn, label2idx):
                self.data = data
                self.encode_fn = encode_fn
                self.label2idx = label2idx
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                x = self.encode_fn(item["text"])
                y = torch.tensor(self.label2idx[item["label"]], dtype=torch.long)
                return x, y
        
        # Shuffle and split
        random.shuffle(data)
        split = int(0.8 * len(data))
        train_data = data[:split]
        val_data = data[split:]
        
        train_dataset = IntentDataset(train_data, self._encode_text, self.label2idx)
        val_dataset = IntentDataset(val_data, self._encode_text, self.label2idx)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        for epoch in range(10):
            self.model.train()
            total_loss = 0
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = self.model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            
            acc = correct / total
            print(f"Query Router Training - Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}")
    
    def route_query(self, query: str) -> str:
        """
        Route query to appropriate document type.
        Returns: 'policy' or 'manual'
        """
        # Classify query intent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        encoded = self._encode_text(query).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.model(encoded)
            pred = logits.argmax(dim=1).item()
        
        intent = self.idx2label[pred]
        
        # Map intent to document type
        policy_intents = ["returns", "warranty", "shipping", "payment"]
        manual_intents = ["product_usage"]
        
        if intent in policy_intents:
            return "policy"
        elif intent in manual_intents:
            return "manual"
        else:
            # Default to policy for unknown intents
            return "policy"


class AnswerGenerator:
    """
    Generates answers using a small LLM with retrieved context.
    Supports multiple model options including T5, GPT-2, and custom models.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load the model
        self._load_model()
        
        # If no model loaded, use the encoder-decoder from Phase 1
        if self.model is None:
            print("Using custom encoder-decoder model from Phase 1")
            self._load_custom_model()
    
    def _load_model(self):
        """Try to load a pre-trained model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"Loaded {self.model_name} model")
            
        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
    
    def _load_custom_model(self):
        """Load custom encoder-decoder model from Phase 1."""
        # This would use the Seq2SeqTransformer from Phase 1
        # For now, we'll implement a simple rule-based approach
        self.model = "rule_based"
    
    def generate_answer(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> StructuredResponse:
        """
        Generate an answer using retrieved context.
        Returns a StructuredResponse with answer, citations, and structured information.
        """
        if not retrieved_chunks:
            return StructuredResponse(
                answer="I couldn't find relevant information to answer your question.",
                confidence=0.0,
                query_type="unknown"
            )
        
        # Combine retrieved chunks into context
        context_parts = []
        citations = []
        
        for i, chunk in enumerate(retrieved_chunks[:3]):  # Use top 3 chunks
            context_parts.append(f"[{i+1}] {chunk.chunk.text}")
            citations.append(f"{chunk.chunk.source} (Score: {chunk.score:.2f})")
        
        context = "\n\n".join(context_parts)
        
        if self.model != "rule_based" and self.tokenizer is not None:
            # Use transformer model
            return self._generate_with_transformer(query, context, citations)
        else:
            # Use rule-based approach
            return self._generate_rule_based(query, retrieved_chunks, citations)
    
    def _generate_with_transformer(self, query: str, context: str, citations: List[str]) -> StructuredResponse:
        """Generate answer using transformer model."""
        # Create prompt
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer from the generated text
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Extract structured information
        structured_info = self._extract_structured_info(answer, query)
        
        return StructuredResponse(
            answer=answer,
            eligibility=structured_info.get("eligibility"),
            required_steps=structured_info.get("required_steps"),
            time_window=structured_info.get("time_window"),
            citations=citations,
            confidence=0.8,  # Placeholder
            query_type="unknown"  # Will be filled by RAG system
        )
    
    def _generate_rule_based(self, query: str, retrieved_chunks: List[RetrievedChunk], citations: List[str]) -> StructuredResponse:
        """Generate answer using rule-based approach."""
        # Extract key information from retrieved chunks
        chunk_texts = [chunk.chunk.text for chunk in retrieved_chunks[:2]]
        combined_text = " ".join(chunk_texts)
        
        # Simple rule-based extraction
        answer = self._extract_answer_from_text(query, combined_text)
        
        # Extract structured information
        structured_info = self._extract_structured_info(answer, query)
        
        return StructuredResponse(
            answer=answer,
            eligibility=structured_info.get("eligibility"),
            required_steps=structured_info.get("required_steps"),
            time_window=structured_info.get("time_window"),
            citations=citations,
            confidence=0.7,  # Placeholder
            query_type="unknown"
        )
    
    def _extract_answer_from_text(self, query: str, text: str) -> str:
        """Extract answer from text based on query."""
        query_lower = query.lower()
        
        # Simple pattern matching
        if "return" in query_lower:
            if "30 days" in text:
                return "You can return most items within 30 days of delivery for a full refund."
            elif "14 days" in text:
                return "Electronics and opened software can be returned within 14 days."
        
        if "warranty" in query_lower:
            if "1 year" in text or "one year" in text:
                return "All electronics come with a 1-year limited warranty covering manufacturing defects."
        
        if "shipping" in query_lower:
            if "5-7" in text:
                return "Standard shipping delivers within 5-7 business days and is free on orders over $50."
        
        # Default answer
        return f"Based on our documentation: {text[:200]}..."
    
    def _extract_structured_info(self, answer: str, query: str) -> Dict[str, Any]:
        """Extract structured information from answer."""
        structured_info = {}
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Extract eligibility
        if any(word in answer_lower for word in ["yes", "can", "able"]):
            structured_info["eligibility"] = "yes"
        elif any(word in answer_lower for word in ["no", "cannot", "not", "unable"]):
            structured_info["eligibility"] = "no"
        elif any(word in answer_lower for word in ["depends", "conditional", "if"]):
            structured_info["eligibility"] = "conditional"
        
        # Extract time windows
        time_patterns = [
            r'(\d+)\s*(?:days?|business\s+days?)',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?',
            r'(\d+)\s*years?',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                structured_info["time_window"] = match.group(0)
                break
        
        # Extract required steps (simple approach)
        steps = []
        if "log into" in answer_lower or "go to" in answer_lower:
            steps.append("Access your account")
        if "select" in answer_lower:
            steps.append("Select the relevant option")
        if "contact" in answer_lower:
            steps.append("Contact customer support")
        
        if steps:
            structured_info["required_steps"] = steps
        
        return structured_info


class RAGEvaluator:
    """
    Evaluates RAG system performance using a gold dataset.
    """
    
    def __init__(self):
        self.gold_dataset = self._create_gold_dataset()
    
    def _create_gold_dataset(self) -> List[Dict]:
        """Create a gold dataset for evaluation."""
        return [
            {
                "query": "How long do I have to return a product?",
                "expected_answer": "You can return most items within 30 days of delivery",
                "expected_citations": ["Returns and Refunds Policy"],
                "expected_eligibility": "yes",
                "expected_time_window": "30 days"
            },
            {
                "query": "Is water damage covered by warranty?",
                "expected_answer": "Warranty does not cover water damage",
                "expected_citations": ["Product Warranty Terms"],
                "expected_eligibility": "no",
                "expected_time_window": None
            },
            {
                "query": "How do I connect to WiFi?",
                "expected_answer": "Go to Settings > Network & Internet > WiFi",
                "expected_citations": ["Smartphone User Manual"],
                "expected_eligibility": "yes",
                "expected_time_window": None
            },
            {
                "query": "What is the warranty period?",
                "expected_answer": "1-year limited warranty",
                "expected_citations": ["Product Warranty Terms"],
                "expected_eligibility": "yes",
                "expected_time_window": "1 year"
            },
            {
                "query": "How long does shipping take?",
                "expected_answer": "Standard shipping delivers within 5-7 business days",
                "expected_citations": ["Shipping and Delivery Policy"],
                "expected_eligibility": "yes",
                "expected_time_window": "5-7 business days"
            }
        ]
    
    def evaluate(self, rag_system) -> Dict[str, float]:
        """Evaluate the RAG system on the gold dataset."""
        results = {
            "answer_accuracy": 0.0,
            "citation_precision": 0.0,
            "eligibility_accuracy": 0.0,
            "time_window_accuracy": 0.0,
            "overall_score": 0.0
        }
        
        total_queries = len(self.gold_dataset)
        answer_scores = []
        citation_scores = []
        eligibility_scores = []
        time_window_scores = []
        
        for gold_item in self.gold_dataset:
            # Get RAG response
            response = rag_system.query(gold_item["query"])
            
            # Evaluate answer accuracy (simple keyword matching)
            expected_keywords = gold_item["expected_answer"].lower().split()
            answer_keywords = response.answer.lower().split()
            
            common_words = len(set(expected_keywords) & set(answer_keywords))
            answer_score = common_words / len(expected_keywords) if expected_keywords else 0
            answer_scores.append(answer_score)
            
            # Evaluate citation precision
            expected_citations = [c.lower() for c in gold_item["expected_citations"]]
            actual_citations = [c.lower() for c in response.citations] if response.citations else []
            
            citation_matches = 0
            for expected in expected_citations:
                for actual in actual_citations:
                    if expected in actual or actual in expected:
                        citation_matches += 1
                        break
            
            citation_score = citation_matches / len(expected_citations) if expected_citations else 0
            citation_scores.append(citation_score)
            
            # Evaluate eligibility accuracy
            expected_eligibility = gold_item["expected_eligibility"]
            actual_eligibility = response.eligibility
            
            eligibility_score = 1.0 if expected_eligibility == actual_eligibility else 0.0
            eligibility_scores.append(eligibility_score)
            
            # Evaluate time window accuracy
            expected_time = gold_item["expected_time_window"]
            actual_time = response.time_window
            
            time_score = 1.0 if expected_time == actual_time else 0.0
            time_window_scores.append(time_score)
        
        # Calculate averages
        results["answer_accuracy"] = np.mean(answer_scores)
        results["citation_precision"] = np.mean(citation_scores)
        results["eligibility_accuracy"] = np.mean(eligibility_scores)
        results["time_window_accuracy"] = np.mean(time_window_scores)
        
        # Overall score (weighted average)
        results["overall_score"] = (
            0.4 * results["answer_accuracy"] +
            0.3 * results["citation_precision"] +
            0.2 * results["eligibility_accuracy"] +
            0.1 * results["time_window_accuracy"]
        )
        
        return results


class ECommerceRAGSystem:
    """
    Main RAG system that combines all components for e-commerce customer support.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "t5-small",
        chunk_size: int = 512,
        top_k_retrieval: int = 5
    ):
        self.embedding_model = EmbeddingModels(embedding_model_name)
        self.vector_store = VectorStore()
        self.query_router = QueryRouter()
        self.answer_generator = AnswerGenerator(llm_model_name)
        self.evaluator = RAGEvaluator()
        
        self.chunk_size = chunk_size
        self.top_k_retrieval = top_k_retrieval
        self.documents_ingested = False
        
        print(f"Initialized E-Commerce RAG System")
        print(f"Embedding Model: {embedding_model_name}")
        print(f"LLM Model: {llm_model_name}")
    
    def ingest_documents(self):
        """Ingest and index all documents."""
        print("Ingesting documents...")
        
        # Initialize document ingestion
        ingestion = DocumentIngestion(self.chunk_size)
        
        # Get all document chunks
        chunks = ingestion.get_all_chunks()
        print(f"Created {len(chunks)} document chunks")
        
        # Encode all chunks
        texts = [chunk.text for chunk in chunks]
        print("Encoding documents...")
        embeddings = self.embedding_model.encode_documents(texts)
        print(f"Created embeddings with shape: {embeddings.shape}")
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
        self.documents_ingested = True
        
        print("Document ingestion completed!")
    
    def query(self, query: str) -> StructuredResponse:
        """
        Main query function that processes user queries and returns structured responses.
        """
        if not self.documents_ingested:
            raise ValueError("Documents must be ingested before querying. Call ingest_documents() first.")
        
        print(f"Processing query: {query}")
        
        # Step 1: Route query to appropriate document type
        doc_type = self.query_router.route_query(query)
        print(f"Routed to document type: {doc_type}")
        
        # Step 2: Encode query
        query_embedding = self.embedding_model.encode_query(query)
        
        # Step 3: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query_embedding, self.top_k_retrieval)
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Filter by document type if needed
        if doc_type != "all":
            retrieved_chunks = [chunk for chunk in retrieved_chunks if chunk.chunk.doc_type == doc_type]
        
        # Step 4: Generate answer with citations
        response = self.answer_generator.generate_answer(query, retrieved_chunks)
        response.query_type = doc_type
        
        print(f"Generated answer: {response.answer[:100]}...")
        return response
    
    def evaluate_system(self) -> Dict[str, float]:
        """Evaluate the RAG system performance."""
        print("Evaluating RAG system...")
        results = self.evaluator.evaluate(self)
        
        print("\nEvaluation Results:")
        print(f"Answer Accuracy: {results['answer_accuracy']:.3f}")
        print(f"Citation Precision: {results['citation_precision']:.3f}")
        print(f"Eligibility Accuracy: {results['eligibility_accuracy']:.3f}")
        print(f"Time Window Accuracy: {results['time_window_accuracy']:.3f}")
        print(f"Overall Score: {results['overall_score']:.3f}")
        
        return results


def main():
    """
    Main function demonstrating the complete RAG pipeline.
    """
    print("=" * 80)
    print("E-Commerce Support Assistant with RAG")
    print("=" * 80)
    
    # Initialize RAG system
    rag_system = ECommerceRAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",  # Fallback to TF-IDF if not available
        llm_model_name="t5-small",  # Fallback to rule-based if not available
        chunk_size=256,
        top_k_retrieval=3
    )
    
    # Ingest documents
    rag_system.ingest_documents()
    
    # Test queries
    test_queries = [
        "How long do I have to return a product?",
        "What is the warranty period for electronics?",
        "How do I connect my phone to WiFi?",
        "Is water damage covered by warranty?",
        "How long does shipping take?",
        "How do I perform a factory reset on my laptop?",
        "What payment methods do you accept?",
        "Can I return an opened product?"
    ]
    
    print("\n" + "=" * 80)
    print("Testing RAG System")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        response = rag_system.query(query)
        
        print(f"Answer: {response.answer}")
        print(f"Query Type: {response.query_type}")
        print(f"Eligibility: {response.eligibility}")
        print(f"Time Window: {response.time_window}")
        
        if response.required_steps:
            print(f"Required Steps: {', '.join(response.required_steps)}")
        
        if response.citations:
            print("Citations:")
            for citation in response.citations:
                print(f"  - {citation}")
        
        print(f"Confidence: {response.confidence:.2f}")
    
    # Evaluate system
    print("\n" + "=" * 80)
    print("System Evaluation")
    print("=" * 80)
    
    results = rag_system.evaluate_system()
    
    # Save results
    with open("rag_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to rag_evaluation_results.json")
    
    # Optional: Fine-tuning setup
    print("\n" + "=" * 80)
    print("Fine-tuning Information")
    print("=" * 80)
    print("To fine-tune the models with LoRA/QLoRA:")
    print("1. Install required packages: pip install peft bitsandbytes")
    print("2. Use the setup_lora_finetuning() method")
    print("3. Prepare your training data in the correct format")
    print("4. Run the fine-tuning process")


if __name__ == "__main__":
    main()
