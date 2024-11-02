import os
from typing import List, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import networkx as nx
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import streamlit as st

class DocumentProcessor:
    def __init__(self, persist_dir: str = "./.chromadb"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.chroma_client = PersistentClient(path=persist_dir)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )
        self.graph = nx.Graph()
        
    def get_uploaded_files(self) -> Set[str]:
        """Get list of unique uploaded files from metadata"""
        try:
            if self.collection.count() == 0:
                return set()
            collection_data = self.collection.get()
            unique_files = set()
            
            # Group IDs by their base filename (before the underscore)
            for id in collection_data['ids']:
                # Extract the base filename (everything before the last underscore)
                base_name = '_'.join(id.split('_')[:-1])  # Join in case filename contains underscores
                if base_name:
                    unique_files.add(base_name)
                    
            return unique_files
        except Exception as e:
            st.toast(f"Error getting files: {str(e)}", icon="⚠️")
            return set()
            
    def remove_file(self, base_filename: str):
        """Remove a file and its chunks from the collection"""
        try:
            if self.collection.count() == 0:
                return False
            
            collection_data = self.collection.get()
            all_ids = collection_data['ids']
            
            # Find all IDs that start with the base filename
            ids_to_remove = [
                id for id in all_ids 
                if id.startswith(base_filename + '_')
            ]
            
            if ids_to_remove:
                try:
                    self.collection.delete(ids=ids_to_remove)
                    # Remove nodes from graph
                    for id in ids_to_remove:
                        if self.graph.has_node(id):
                            self.graph.remove_node(id)
                    st.toast(f"Removed {base_filename}", icon="✅")
                    return True
                except Exception as e:
                    st.toast(f"Error during deletion: {str(e)}", icon="⚠️")
                    return False
            else:
                st.toast(f"No chunks found for {base_filename}", icon="ℹ️")
                return False
        except Exception as e:
            st.toast(f"Error removing file: {str(e)}", icon="⚠️")
            return False
        
    def process_file(self, file_path: str, original_filename: str) -> List[str]:
        """Process a file and return chunks"""
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
            
            # Add chunks to ChromaDB
            for i, chunk in enumerate(chunks):
                try:
                    chunk_id = f"{base_id}_{i}"
                    self.collection.add(
                        documents=[chunk.page_content],
                        metadatas=[{"source": original_filename}],
                        ids=[chunk_id]
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
        """Query similar documents"""
        try:
            if self.collection.count() == 0:
                return []
                
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            return results['documents'][0]
        except Exception as e:
            st.toast(f"Error in query_similar: {str(e)}", icon="⚠️")
            return [] 