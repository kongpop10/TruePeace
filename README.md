# Beyond Path Chat Application

A Streamlit-based chat application that uses Groq's LLM API for intelligent responses and Supabase for persistent vector storage. The application supports document upload for context-aware conversations.

## Features

- 💬 Context-Aware Chat with Groq LLM
  - Intelligent responses using Groq's Llama 3.1 8B Instant model
  - Rate limit handling with automatic retries
  - Document-based context integration
  - Thai language support
  
- 📄 Document Management (Admin Only)
  - Support for text and markdown files
  - Automatic document chunking and embedding
  - Real-time processing status
  
- 🔍 Vector Search
  - Semantic similarity search using Supabase
  - Fast and accurate document retrieval

## Setup

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd beyond-path
   pip install -r requirements.txt
   ```

2. **Configure Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   SUPABASE_URL = "your_supabase_url"
   SUPABASE_KEY = "your_supabase_key"
   ADMIN_PASSWORD = "your_admin_password"
   ```

3. **Set up Supabase**
   - Create a new Supabase project
   - Enable the Vector extension
   - Run the setup SQL (see Supabase Setup section)

4. **Run the Application**
   ```bash
   streamlit run chat_app.py
   ```

## Supabase Setup

```sql
-- Enable vector extension
create extension if not exists vector;

-- Create documents table
create table if not exists document_chunks (
    id text primary key,
    content text,
    embedding vector(384),
    metadata jsonb
);

-- Create similarity search function
create or replace function match_documents_similarity_v1 (
    query_embedding vector(384),
    match_threshold float DEFAULT 0.78,
    match_count int DEFAULT 5
)
returns table (
    id text,
    content text,
    similarity float,
    metadata jsonb
)
language plpgsql
as $$
begin
    return query
    select
        dc.id,
        dc.content,
        1 - (dc.embedding <=> query_embedding) as similarity,
        dc.metadata
    from document_chunks dc
    where 1 - (dc.embedding <=> query_embedding) > match_threshold
    order by dc.embedding <=> query_embedding
    limit match_count;
end;
$$;
```

## Rate Limits

The application handles Groq API rate limits automatically:
- Displays warnings when approaching limits
- Shows wait times when limits are reached
- Automatically retries requests after required wait time
- Handles both request-based and token-based limits

## Usage

### Regular Users
- Access the chat interface
- Ask questions and receive AI responses
- Benefit from context-aware answers

### Admin Users
1. Access admin interface via sidebar
2. Upload and manage documents
3. Reindex documents as needed

## Security Notes

- Keep your `.streamlit/secrets.toml` secure
- Never commit secrets to version control
- Regularly rotate API keys and credentials
- Use strong admin passwords

## License

[Your License Here]