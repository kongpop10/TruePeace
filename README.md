# Beyond Path Chat Application

A Streamlit-based chat application that uses Google's Gemini API for intelligent responses and a hybrid database system (local SQLite + Supabase) for persistent vector storage. The application provides Thai language support and context-aware conversations through document-based knowledge.

## Hybrid Database System

The application now features a robust hybrid database system that:
- Uses SQLite for local storage when Supabase is unavailable
- Automatically syncs with Supabase when it becomes available
- Provides seamless fallback to local storage during connectivity issues
- Includes admin tools for database management and synchronization

## Features

- 💬 Thai Language Chat with Google Gemini
  - Intelligent responses using Google's Gemini 2.0 Flash model
  - Enhanced Thai language keyword matching
  - Document-based context integration
  - Smart keyword extraction for better relevance
  - Automatic rate limit handling

- 📄 Document Management (Admin Only)
  - Support for text and markdown files
  - Automatic document chunking and embedding
  - Real-time processing status
  - Secure admin interface

- 🔍 Advanced Search
  - Hybrid search combining keyword and semantic matching
  - Up to 10 most relevant context chunks
  - Improved Thai language search relevance

## Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/kongpop10/TruePeace.git
   cd TruePeace
   pip install -r requirements.txt
   ```

2. **Configure Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key"
   SUPABASE_URL = "your_supabase_url"  # Optional - app works without Supabase
   SUPABASE_KEY = "your_supabase_key"  # Optional - app works without Supabase
   ADMIN_PASSWORD = "your_admin_password"
   ```

3. **Set up Supabase (Optional)**
   - Create a new Supabase project
   - Enable the Vector extension
   - Run the setup SQL (see Supabase Setup section)
   - The app will work without Supabase using the local SQLite database

4. **Run the Application**
   ```bash
   streamlit run chat_app.py
   ```

   The application will automatically create a local SQLite database in the `data/` directory.

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
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
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

## Search Features

The application uses a sophisticated search approach:
- Extracts keywords from user questions
- Performs direct keyword matching in document chunks
- Falls back to semantic similarity search if needed
- Returns up to 10 most relevant chunks for comprehensive answers
- Optimized for Thai language queries

## Rate Limits

The application handles Gemini API rate limits automatically:
- Displays warnings when limits are reached
- Automatically retries requests after required wait time
- Implements a simplified rate limit handling mechanism
- Gracefully recovers from rate limit errors

## Usage

### Regular Users
- Access the chat interface
- Ask questions and receive AI responses
- Benefit from context-aware answers

### Admin Users
1. Access admin interface via sidebar login
2. Upload and manage documents
3. Monitor document processing status
4. Reindex documents as needed

## Security Notes

- Keep your `.streamlit/secrets.toml` secure
- Never commit secrets to version control
- Regularly rotate API keys and credentials
- Use strong admin passwords
- Admin login includes error handling and password protection

## License

MIT License