import streamlit as st
import time
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .supabase_config import SupabaseManager
from .sqlite_manager import SQLiteManager

class DatabaseManager:
    """Hybrid database manager that can use either Supabase or SQLite."""

    def __init__(self, db_path: str = "data/local_db.sqlite", show_notifications: bool = False):
        """Initialize the database manager.

        Args:
            db_path: Path to SQLite database file
            show_notifications: Whether to show toast notifications
        """
        self.sqlite = SQLiteManager(db_path)
        self.supabase = None
        self.supabase_available = False
        self.show_notifications = show_notifications

        # Try to initialize Supabase if credentials are available
        try:
            if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
                self.supabase = SupabaseManager()
                self.check_supabase_connection()
            elif self.show_notifications:
                st.toast("Supabase credentials not found. Using local database only.", icon="ℹ️")
        except Exception as e:
            if self.show_notifications:
                st.toast(f"Error initializing Supabase: {str(e)}. Using local database only.", icon="⚠️")

    def check_supabase_connection(self) -> bool:
        """Check if Supabase is available.

        Returns:
            bool: True if Supabase is available, False otherwise
        """
        if not self.supabase:
            self.supabase_available = False
            return False

        try:
            # Try to make a simple query to Supabase
            response = self.supabase.supabase.table('document_chunks').select('id').limit(1).execute()

            # If we get here, Supabase is available
            self.supabase_available = True
            return True
        except Exception as e:
            if self.show_notifications:
                st.toast(f"Supabase connection failed: {str(e)}. Using local database.", icon="⚠️")
            self.supabase_available = False
            return False

    def store_document_chunk(self, chunk_id: str, content: str, embedding: list, metadata: dict) -> bool:
        """Store document chunk in the appropriate database.

        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content of the chunk
            embedding: Vector embedding of the chunk
            metadata: Additional metadata for the chunk

        Returns:
            bool: True if successful, False otherwise
        """
        # Always store in local SQLite
        sqlite_success = self.sqlite.store_document_chunk(chunk_id, content, embedding, metadata)

        # If Supabase is available, store there too and mark as synced in SQLite
        if self.supabase_available and self.supabase:
            try:
                supabase_success = self.supabase.store_document_chunk(chunk_id, content, embedding, metadata)
                if supabase_success and sqlite_success:
                    self.sqlite.mark_as_synced(chunk_id)
                return sqlite_success and supabase_success
            except Exception as e:
                if self.show_notifications:
                    st.toast(f"Error storing in Supabase: {str(e)}. Stored locally only.", icon="⚠️")
                return sqlite_success

        return sqlite_success

    def get_document_chunks(self) -> List[Dict[str, Any]]:
        """Get document chunks from the appropriate database.

        Returns:
            List of document chunks
        """
        if self.supabase_available and self.supabase:
            try:
                return self.supabase.get_document_chunks()
            except Exception as e:
                if self.show_notifications:
                    st.toast(f"Error getting chunks from Supabase: {str(e)}. Using local database.", icon="⚠️")
                self.supabase_available = False

        return self.sqlite.get_document_chunks()

    def delete_document_chunks(self, base_filename: str) -> bool:
        """Delete document chunks from the appropriate database.

        Args:
            base_filename: Base filename to match for deletion

        Returns:
            bool: True if successful, False otherwise
        """
        # Always delete from local SQLite
        sqlite_success = self.sqlite.delete_document_chunks(base_filename)

        # If Supabase is available, delete there too
        if self.supabase_available and self.supabase:
            try:
                supabase_success = self.supabase.delete_document_chunks(base_filename)
                return sqlite_success and supabase_success
            except Exception as e:
                if self.show_notifications:
                    st.toast(f"Error deleting from Supabase: {str(e)}. Deleted locally only.", icon="⚠️")
                self.supabase_available = False

        return sqlite_success

    def search_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for keyword in the appropriate database.

        Args:
            keyword: Keyword to search for
            limit: Maximum number of results to return

        Returns:
            List of matching document chunks
        """
        if self.supabase_available and self.supabase:
            try:
                return self.supabase.search_keyword(keyword, limit)
            except Exception as e:
                if self.show_notifications:
                    st.toast(f"Error searching Supabase: {str(e)}. Using local database.", icon="⚠️")
                self.supabase_available = False

        return self.sqlite.search_keyword(keyword, limit)

    def query_similar(self, query: str, query_embedding: list, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query similar documents from the appropriate database.

        Args:
            query: Query text
            query_embedding: Vector embedding of the query
            n_results: Maximum number of results to return

        Returns:
            List of matching document chunks
        """
        if self.supabase_available and self.supabase:
            try:
                return self.supabase.query_similar(query, query_embedding, n_results)
            except Exception as e:
                if self.show_notifications:
                    st.toast(f"Error querying Supabase: {str(e)}. Using local database.", icon="⚠️")
                self.supabase_available = False

        return self.sqlite.query_similar(query, query_embedding, n_results)

    def sync_to_supabase(self) -> Tuple[int, int, str]:
        """Sync local database to Supabase.

        Returns:
            Tuple of (chunks_synced, total_chunks, status_message)
        """
        if not self.supabase:
            return 0, 0, "Supabase not configured"

        # Check if Supabase is available
        if not self.check_supabase_connection():
            return 0, 0, "Supabase not available"

        # Get unsynced chunks
        unsynced_chunks = self.sqlite.get_unsynced_chunks()
        total_chunks = len(unsynced_chunks)

        if total_chunks == 0:
            self.sqlite.update_sync_status(int(time.time()), "All chunks synced")
            return 0, 0, "All chunks already synced"

        # Sync chunks to Supabase
        synced_count = 0
        for chunk in unsynced_chunks:
            try:
                success = self.supabase.store_document_chunk(
                    chunk_id=chunk['id'],
                    content=chunk['content'],
                    embedding=chunk['embedding'],
                    metadata=chunk['metadata']
                )

                if success:
                    self.sqlite.mark_as_synced(chunk['id'])
                    synced_count += 1
            except Exception as e:
                st.toast(f"Error syncing chunk {chunk['id']}: {str(e)}", icon="⚠️")

        # Update sync status
        status = f"Synced {synced_count}/{total_chunks} chunks"
        self.sqlite.update_sync_status(int(time.time()), status)

        return synced_count, total_chunks, status

    def sync_from_supabase(self) -> Tuple[int, int, str]:
        """Sync from Supabase to local database.

        Returns:
            Tuple of (chunks_synced, total_chunks, status_message)
        """
        if not self.supabase:
            return 0, 0, "Supabase not configured"

        # Check if Supabase is available
        if not self.check_supabase_connection():
            return 0, 0, "Supabase not available"

        try:
            # Get all chunks from Supabase
            supabase_chunks = self.supabase.get_document_chunks()
            total_chunks = len(supabase_chunks)

            if total_chunks == 0:
                return 0, 0, "No chunks in Supabase"

            # Store chunks in local database
            synced_count = 0
            for chunk in supabase_chunks:
                try:
                    success = self.sqlite.store_document_chunk(
                        chunk_id=chunk['id'],
                        content=chunk['content'],
                        embedding=chunk['embedding'],
                        metadata=chunk['metadata']
                    )

                    if success:
                        self.sqlite.mark_as_synced(chunk['id'])
                        synced_count += 1
                except Exception as e:
                    st.toast(f"Error syncing chunk {chunk['id']} from Supabase: {str(e)}", icon="⚠️")

            # Update sync status
            status = f"Synced {synced_count}/{total_chunks} chunks from Supabase"
            self.sqlite.update_sync_status(int(time.time()), status)

            return synced_count, total_chunks, status
        except Exception as e:
            st.toast(f"Error syncing from Supabase: {str(e)}", icon="⚠️")
            return 0, 0, f"Error: {str(e)}"

    def get_sync_status(self) -> Dict[str, Any]:
        """Get the current sync status.

        Returns:
            Dictionary with sync status information
        """
        sqlite_status = self.sqlite.get_sync_status()

        return {
            'last_sync': sqlite_status['last_sync'],
            'status': sqlite_status['status'],
            'supabase_available': self.supabase_available
        }

    def close(self):
        """Close database connections."""
        self.sqlite.close()
