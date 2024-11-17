from supabase import create_client
import streamlit as st
import numpy as np
import json

class SupabaseManager:
    def __init__(self):
        if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
            st.error("Please set SUPABASE_URL and SUPABASE_KEY in your Streamlit secrets.")
            st.stop()
            
        self.supabase_url = st.secrets["SUPABASE_URL"]
        self.supabase_key = st.secrets["SUPABASE_KEY"]
        self.supabase = create_client(self.supabase_url, self.supabase_key)

    def store_document_chunk(self, chunk_id: str, content: str, embedding: list, metadata: dict):
        """Store document chunk and its embedding"""
        try:
            # Convert numpy array to list if necessary
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
                
            data = {
                'id': chunk_id,
                'content': content,
                'embedding': embedding,  # Store directly as list
                'metadata': json.dumps(metadata)
            }
            self.supabase.table('document_chunks').upsert(data).execute()
            return True
        except Exception as e:
            st.error(f"Error storing document chunk: {str(e)}")
            st.error(f"Full error details: {str(e)}")  
            return False

    def get_document_chunks(self):
        """Retrieve all document chunks"""
        try:
            response = self.supabase.table('document_chunks').select('*').execute()
            return response.data
        except Exception as e:
            st.error(f"Error retrieving document chunks: {str(e)}")
            return []

    def delete_document_chunks(self, base_filename: str):
        """Delete all chunks for a given file"""
        try:
            self.supabase.table('document_chunks')\
                .delete()\
                .like('id', f'{base_filename}_%')\
                .execute()
            return True
        except Exception as e:
            st.error(f"Error deleting document chunks: {str(e)}")
            return False

    def search_keyword(self, keyword: str, limit: int = 10):
        """Direct keyword search in document_chunks content field"""
        try:
            # Search for content containing the keyword
            response = self.supabase.table('document_chunks')\
                .select('content')\
                .ilike('content', f'%{keyword}%')\
                .limit(limit)\
                .execute()
            
            matches = response.data if response.data else []
            if matches:
                st.write(f"Found {len(matches)} chunks containing '{keyword}'")
            
            return matches
        except Exception as e:
            st.error(f"Error in keyword search: {str(e)}")
            return []

    def query_similar(self, query: str, query_embedding: list, n_results: int = 10):
        """Query documents using keyword matching"""
        try:
            # First try keyword search
            matches = self.search_keyword(query, n_results)
            
            # If no keyword matches, try similarity search
            if not matches:
                similarity_response = self.supabase.rpc(
                    'match_documents_similarity_v1',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': 0.5,
                        'match_count': n_results
                    }
                ).execute()
                matches = similarity_response.data if similarity_response.data else []
            
            return matches
            
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return []