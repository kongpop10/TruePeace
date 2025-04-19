import os
from typing import List, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import networkx as nx
import streamlit as st
import numpy as np
from fastembed import TextEmbedding

# Import the new database manager instead of directly using Supabase
from .database_manager import DatabaseManager

class SimpleDocumentProcessor:
    def __init__(self, db_path: str = "data/local_db.sqlite", show_notifications: bool = False):
        # Use the hybrid database manager instead of directly using Supabase
        self.db = DatabaseManager(db_path)
        self.graph = nx.Graph()
        # Initialize the FastEmbed model
        try:
            # Use the default model (BAAI/bge-small-en-v1.5) which is known to work
            self.embedding_model = TextEmbedding()
            if show_notifications:
                st.toast(f"FastEmbed model loaded successfully: {self.embedding_model.model_name}", icon="✅")
        except Exception as e:
            if show_notifications:
                st.toast(f"Error loading FastEmbed model: {str(e)}", icon="⚠️")
            # Fallback to simple embedding if FastEmbed fails
            self.embedding_model = None

        # Display database status (only show toasts if requested)
        self._show_db_status(show_toast=show_notifications)

    def _show_db_status(self, show_toast=False):
        """Display the current database status.

        Args:
            show_toast: Whether to show toast notifications (for admin users)
        """
        status = self.db.get_sync_status()
        if show_toast:
            if status['supabase_available']:
                st.toast("Connected to Supabase and local database", icon="✅")
            else:
                st.toast("Using local database only (Supabase not available)", icon="ℹ️")
        return status

    def get_uploaded_files(self) -> Set[str]:
        """Get list of unique uploaded files from metadata"""
        try:
            chunks = self.db.get_document_chunks()
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
        """Remove a file and its chunks from the database."""
        try:
            # Delete all chunks associated with the base filename
            success = self.db.delete_document_chunks(base_filename)
            if success:
                st.toast(f"Removed all chunks for {base_filename}", icon="✅")

            return success
        except Exception as e:
            st.toast(f"Error removing file: {str(e)}", icon="⚠️")
            return False

    def sync_database(self, show_toast=True):
        """Sync local database with Supabase if available.

        Args:
            show_toast: Whether to show toast notifications

        Returns:
            bool: True if any chunks were synced, False otherwise
        """
        # First check if Supabase is available
        if not self.db.supabase_available:
            if show_toast:
                st.toast("Supabase not available. Cannot sync.", icon="⚠️")
            return False

        # Sync local changes to Supabase
        synced, total, status = self.db.sync_to_supabase()
        if show_toast:
            st.toast(f"Sync status: {status}", icon="ℹ️")
        return synced > 0

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using FastEmbed or fallback to simple method"""
        if self.embedding_model is not None:
            try:
                # FastEmbed returns a generator, convert to list first
                embeddings = list(self.embedding_model.embed([text]))
                if embeddings and len(embeddings) > 0:
                    # Convert numpy array to list
                    return embeddings[0].tolist()
                else:
                    st.toast("No embeddings generated, using fallback", icon="⚠️")
                    return self.simple_embedding(text)
            except Exception as e:
                st.toast(f"Error generating embedding: {str(e)}", icon="⚠️")
                # Fallback to simple embedding if FastEmbed fails
                return self.simple_embedding(text)
        else:
            return self.simple_embedding(text)

    def simple_embedding(self, text: str) -> List[float]:
        """A very simple embedding function that creates a random vector.
        Used as fallback if FastEmbed fails."""
        # Create a random vector of length 384 (same as the original model)
        # Use a seed based on the text to make it deterministic
        seed = sum(ord(c) for c in text)
        np.random.seed(seed)
        return np.random.rand(384).tolist()

    def process_file(self, file_path: str, original_filename: str) -> List[str]:
        """Process a file and store chunks in the database"""
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
                    embedding = self.get_embedding(chunk.page_content)

                    # Use the database manager instead of directly using Supabase
                    self.db.store_document_chunk(
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

            # Try to sync with Supabase if available, but don't show toasts for regular users
            if self.db.supabase_available:
                self.sync_database(show_toast=False)

            # Only show success toast for admin users
            if hasattr(self, 'show_notifications') and self.show_notifications:
                st.toast(f"Successfully processed {len(chunks)} chunks for {original_filename}.", icon="✅")
            return [chunk.page_content for chunk in chunks]

        except Exception as e:
            st.toast(f"Error processing file: {str(e)}", icon="⚠️")
            raise e

    def query_similar(self, query: str, n_results: int = 5) -> List[str]:
        """Query documents for keyword matches, with similarity search as fallback."""
        try:
            # Generate embedding using FastEmbed
            query_embedding = self.get_embedding(query)

            # Search for matches in document_chunks using the database manager
            st.write(f"\nSearching for content containing: {query}")
            results = self.db.query_similar(query, query_embedding, n_results)

            # Extract and return content from matches
            if results:
                return [result['content'] for result in results]

            st.write("No matching content found.")
            return []

        except Exception as e:
            st.toast(f"Error in query_similar: {str(e)}", icon="⚠️")
            return []

    def import_from_supabase(self, show_toast=True):
        """Import all data from Supabase to local database.

        Args:
            show_toast: Whether to show toast notifications

        Returns:
            bool: True if any chunks were imported, False otherwise
        """
        if not self.db.supabase_available:
            if show_toast:
                st.toast("Supabase not available. Cannot import data.", icon="⚠️")
            return False

        synced, total, status = self.db.sync_from_supabase()
        if show_toast:
            st.toast(f"Import status: {status}", icon="ℹ️")
        return synced > 0
