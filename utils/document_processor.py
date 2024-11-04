import os
from typing import List, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import networkx as nx
import streamlit as st
from .supabase_config import SupabaseManager
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self):
        self.embedding_function = SentenceTransformer('all-MiniLM-L6-v2')
        self.supabase = SupabaseManager()
        self.graph = nx.Graph()
        
    def get_uploaded_files(self) -> Set[str]:
        """Get list of unique uploaded files from metadata"""
        try:
            chunks = self.supabase.get_document_chunks()
            unique_files = set()
            
            for chunk in chunks:
                # Extract the base filename (everything before the last underscore)
                base_name = '_'.join(chunk['id'].split('_')[:-1])
                if base_name:
                    unique_files.add(base_name)
                    
            return unique_files
        except Exception as e:
            st.toast(f"Error getting files: {str(e)}", icon="⚠️")
            return set()
            
    def remove_file(self, base_filename: str):
        """Remove a file and its chunks from Supabase"""
        try:
            success = self.supabase.delete_document_chunks(base_filename)
            if success:
                st.toast(f"Removed {base_filename}", icon="✅")
            return success
        except Exception as e:
            st.toast(f"Error removing file: {str(e)}", icon="⚠️")
            return False
        
    def process_file(self, file_path: str, original_filename: str) -> List[str]:
        """Process a file and store chunks in Supabase"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=8000,
                chunk_overlap=500,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Use original filename without extension as base_id
            base_id = os.path.splitext(original_filename)[0]
            
            # Process and store chunks
            for i, chunk in enumerate(chunks):
                try:
                    chunk_id = f"{base_id}_{i}"
                    embedding = self.embedding_function.encode([chunk.page_content])[0].tolist()
                    
                    self.supabase.store_document_chunk(
                        chunk_id=chunk_id,
                        content=chunk.page_content,
                        embedding=embedding,
                        metadata={"source": original_filename}
                    )
                    
                    # Add nodes and edges to graph
                    self.graph.add_node(chunk_id, content=chunk.page_content)
                    if i > 0:
                        self.graph.add_edge(f"{base_id}_{i-1}", chunk_id)
                except Exception as e:
                    st.toast(f"Error processing chunk {i}: {str(e)}", icon="⚠️")
                    continue
            
            return [chunk.page_content for chunk in chunks]
            
        except Exception as e:
            st.toast(f"Error processing file: {str(e)}", icon="⚠️")
            raise e

    def query_similar(self, query: str, n_results: int = 3) -> List[str]:
        """Query similar documents using Supabase vector similarity"""
        try:
            query_embedding = self.embedding_function.encode([query])[0].tolist()
            results = self.supabase.query_similar(query_embedding, n_results)
            return [result['content'] for result in results] if results else []
        except Exception as e:
            st.toast(f"Error in query_similar: {str(e)}", icon="⚠️")
            return []