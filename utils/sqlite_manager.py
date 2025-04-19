import sqlite3
import json
import os
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional
import time

class SQLiteManager:
    """SQLite database manager for storing document chunks and embeddings locally."""

    def __init__(self, db_path: str = "data/local_db.sqlite"):
        """Initialize SQLite database manager.

        Args:
            db_path: Path to SQLite database file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Connect to SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Configure to return rows as dictionaries
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            # Use print instead of st.error to avoid UI notifications
            print(f"Error connecting to SQLite database: {str(e)}")
            return False

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            # Create document_chunks table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                embedding BLOB,  -- Store embedding as binary blob
                metadata TEXT,   -- Store metadata as JSON string
                last_updated INTEGER,  -- Unix timestamp for sync purposes
                synced INTEGER DEFAULT 0  -- 0 = not synced, 1 = synced
            )
            ''')

            # Create sync_status table to track last sync time
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_status (
                id INTEGER PRIMARY KEY,
                last_sync INTEGER,  -- Unix timestamp of last sync
                status TEXT         -- Status message
            )
            ''')

            # Insert initial sync status if not exists
            self.cursor.execute('''
            INSERT OR IGNORE INTO sync_status (id, last_sync, status)
            VALUES (1, 0, 'Never synced')
            ''')

            self.conn.commit()
            return True
        except Exception as e:
            # Use print instead of st.error to avoid UI notifications
            print(f"Error creating tables: {str(e)}")
            return False

    def store_document_chunk(self, chunk_id: str, content: str, embedding: list, metadata: dict) -> bool:
        """Store document chunk and its embedding in SQLite.

        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content of the chunk
            embedding: Vector embedding of the chunk
            metadata: Additional metadata for the chunk

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert numpy array to bytes for storage
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # Convert embedding list to bytes
            embedding_bytes = json.dumps(embedding).encode()

            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)

            # Current timestamp
            timestamp = int(time.time())

            # Insert or replace document chunk
            self.cursor.execute('''
            INSERT OR REPLACE INTO document_chunks
            (id, content, embedding, metadata, last_updated, synced)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (chunk_id, content, embedding_bytes, metadata_json, timestamp, 0))

            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error storing document chunk in SQLite: {str(e)}")
            return False

    def get_document_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all document chunks from SQLite.

        Returns:
            List of document chunks as dictionaries
        """
        try:
            self.cursor.execute('SELECT id, content, embedding, metadata FROM document_chunks')
            rows = self.cursor.fetchall()

            result = []
            for row in rows:
                # Convert embedding bytes back to list
                embedding_bytes = row['embedding']
                embedding = json.loads(embedding_bytes.decode())

                # Convert metadata JSON string back to dict
                metadata_json = row['metadata']
                metadata = json.loads(metadata_json)

                result.append({
                    'id': row['id'],
                    'content': row['content'],
                    'embedding': embedding,
                    'metadata': metadata
                })

            return result
        except Exception as e:
            print(f"Error retrieving document chunks from SQLite: {str(e)}")
            return []

    def delete_document_chunks(self, base_filename: str) -> bool:
        """Delete all chunks for a given file from SQLite.

        Args:
            base_filename: Base filename to match for deletion

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cursor.execute(
                "DELETE FROM document_chunks WHERE id LIKE ?",
                (f'{base_filename}_%',)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting document chunks from SQLite: {str(e)}")
            return False

    def search_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Direct keyword search in document_chunks content field.

        Args:
            keyword: Keyword to search for
            limit: Maximum number of results to return

        Returns:
            List of matching document chunks
        """
        try:
            self.cursor.execute(
                "SELECT content FROM document_chunks WHERE content LIKE ? LIMIT ?",
                (f'%{keyword}%', limit)
            )
            rows = self.cursor.fetchall()

            matches = [{'content': row['content']} for row in rows]
            if matches:
                st.write(f"Found {len(matches)} chunks containing '{keyword}'")

            return matches
        except Exception as e:
            print(f"Error in SQLite keyword search: {str(e)}")
            return []

    def query_similar(self, query: str, query_embedding: list, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query documents using keyword matching in SQLite.

        Note: This is a simplified version that doesn't do true vector similarity search,
        as SQLite doesn't have built-in vector operations. For a production app, consider
        using an extension like sqlite-vss or a more sophisticated approach.

        Args:
            query: Query text
            query_embedding: Vector embedding of the query
            n_results: Maximum number of results to return

        Returns:
            List of matching document chunks
        """
        try:
            # First try keyword search
            matches = self.search_keyword(query, n_results)

            # If no keyword matches and we have less than n_results,
            # fall back to returning the most recent chunks
            if not matches or len(matches) < n_results:
                remaining = n_results - len(matches)
                self.cursor.execute(
                    "SELECT content FROM document_chunks ORDER BY last_updated DESC LIMIT ?",
                    (remaining,)
                )
                rows = self.cursor.fetchall()
                additional_matches = [{'content': row['content']} for row in rows]
                matches.extend(additional_matches)

            return matches
        except Exception as e:
            print(f"Error querying documents from SQLite: {str(e)}")
            return []

    def mark_as_synced(self, chunk_id: str) -> bool:
        """Mark a document chunk as synced with Supabase.

        Args:
            chunk_id: ID of the chunk to mark as synced

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cursor.execute(
                "UPDATE document_chunks SET synced = 1 WHERE id = ?",
                (chunk_id,)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error marking chunk as synced: {str(e)}")
            return False

    def get_unsynced_chunks(self) -> List[Dict[str, Any]]:
        """Get all document chunks that haven't been synced with Supabase.

        Returns:
            List of unsynced document chunks
        """
        try:
            self.cursor.execute('''
            SELECT id, content, embedding, metadata
            FROM document_chunks
            WHERE synced = 0
            ''')
            rows = self.cursor.fetchall()

            result = []
            for row in rows:
                # Convert embedding bytes back to list
                embedding_bytes = row['embedding']
                embedding = json.loads(embedding_bytes.decode())

                # Convert metadata JSON string back to dict
                metadata_json = row['metadata']
                metadata = json.loads(metadata_json)

                result.append({
                    'id': row['id'],
                    'content': row['content'],
                    'embedding': embedding,
                    'metadata': metadata
                })

            return result
        except Exception as e:
            print(f"Error retrieving unsynced chunks: {str(e)}")
            return []

    def update_sync_status(self, timestamp: int, status: str) -> bool:
        """Update the last sync timestamp and status.

        Args:
            timestamp: Unix timestamp of the sync
            status: Status message

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cursor.execute('''
            UPDATE sync_status
            SET last_sync = ?, status = ?
            WHERE id = 1
            ''', (timestamp, status))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating sync status: {str(e)}")
            return False

    def get_sync_status(self) -> Dict[str, Any]:
        """Get the current sync status.

        Returns:
            Dictionary with last_sync timestamp and status message
        """
        try:
            self.cursor.execute('SELECT last_sync, status FROM sync_status WHERE id = 1')
            row = self.cursor.fetchone()

            if row:
                return {
                    'last_sync': row['last_sync'],
                    'status': row['status']
                }
            else:
                return {
                    'last_sync': 0,
                    'status': 'Unknown'
                }
        except Exception as e:
            print(f"Error getting sync status: {str(e)}")
            return {
                'last_sync': 0,
                'status': f'Error: {str(e)}'
            }

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
