# AI Chat Assistant

A Streamlit-based chat application that uses Together AI's API for responses and supports document upload for context-aware conversations. The application uses Supabase for persistent storage of document embeddings.

## Setup

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Set up Streamlit secrets with your API keys:
   ```toml
   # .streamlit/secrets.toml
   TOGETHER_API_KEY = "your_together_api_key"
   SUPABASE_URL = "your_supabase_url"
   SUPABASE_KEY = "your_supabase_key"
   ```
4. Set up Supabase:
   - Create a new project in Supabase
   - Run the following SQL in the SQL editor:
   ```sql
   create extension if not exists vector;

   create table if not exists document_chunks (
       id text primary key,
       content text,
       embedding vector(384),
       metadata jsonb
   );

   create or replace function match_documents (
       query_embedding vector(384),
       match_count int
   )
   returns table (
       id text,
       content text,
       similarity float
   )
   language plpgsql
   as $$
   begin
       return query
       select
           id,
           content,
           1 - (document_chunks.embedding <=> query_embedding) as similarity
       from document_chunks
       order by document_chunks.embedding <=> query_embedding
       limit match_count;
   end;
   $$;
   ```

5. Run the app: `streamlit run chat_app.py`

## Features

- Document upload and processing
- Context-aware conversations
- Vector similarity search
- Persistent storage with Supabase
- Real-time chat interface

## Environment Variables

Required secrets for Streamlit deployment:
- `TOGETHER_API_KEY`: Your Together AI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key

## Deployment

1. Set up a new project on Streamlit Cloud
2. Add the required secrets in the Streamlit Cloud dashboard
3. Deploy the application

## Security Notes

- Never commit `.streamlit/secrets.toml` to version control
- Keep your API keys secure
- Use environment variables for local development