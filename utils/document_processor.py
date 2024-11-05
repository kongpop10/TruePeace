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
        """Remove a file and its chunks from Supabase."""
        try:
            # Delete all chunks associated with the base filename
            success = self.supabase.delete_document_chunks(base_filename)
            if success:
                st.toast(f"Removed all chunks for {base_filename}", icon="✅")
            
            # Optionally, you can also remove the file from any other storage if applicable
            # For example, if you have a method to delete the file itself, call it here.
            
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
            if not documents:
                st.toast(f"No content found in {original_filename}.", icon="⚠️")
                return []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=150,
                chunk_overlap=30,
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
                    
                    # Log the chunk being stored
                    # print(f"Storing chunk: {chunk_id} for file: {original_filename}")  # Commented out
                    
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
            
            st.toast(f"Successfully processed {len(chunks)} chunks for {original_filename}.", icon="✅")
            return [chunk.page_content for chunk in chunks]
            
        except Exception as e:
            st.toast(f"Error processing file: {str(e)}", icon="⚠️")
            raise e

    def query_similar(self, query: str, n_results: int = 3) -> List[str]:
        """Query similar documents using Supabase vector similarity with improved retrieval."""
        try:
            # Generate the embedding for the query
            query_embedding = self.embedding_function.encode([query])[0].tolist()
            
            # Retrieve similar documents from Supabase
            results = self.supabase.query_similar(query_embedding, n_results)
            
            # If results are found, sort them based on similarity score
            if results:
                # Assuming results contain 'content' and 'similarity' fields
                sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
                return [result['content'] for result in sorted_results[:n_results]]  # Limit to n_results
            return []
        except Exception as e:
            st.toast(f"Error in query_similar: {str(e)}", icon="⚠️")
            return []