# AI Chat Assistant

A Streamlit-based chat application that leverages Together AI's API for intelligent responses and Supabase for persistent vector storage. The application supports document upload for context-aware conversations, making it ideal for document-based Q&A.

## Features

- 📄 Document Upload & Processing (Admin Only)
  - Support for text and markdown files
  - Automatic document chunking and embedding
  - Real-time processing status
  - Secure admin-only access
  
- 💬 Context-Aware Chat
  - Intelligent responses using Together AI's LLM
  - Document-based context integration
  - Persistent chat history
  - Token-optimized responses
  
- 🔍 Vector Search
  - Semantic similarity search using Supabase vector storage
  - Fast and accurate document retrieval
  - Configurable similarity thresholds
  
- 💾 Persistent Storage
  - Supabase backend for document storage
  - Vector embeddings for efficient search
  - Secure and scalable data management

## Quick Start

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd ai-chat-assistant
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   - Copy the secrets template:
     ```bash
     cp .streamlit/secrets.toml.example .streamlit/secrets.toml
     ```
   - Update `.streamlit/secrets.toml` with your credentials:
     ```toml
     TOGETHER_API_KEY = "your_together_api_key"
     SUPABASE_URL = "your_supabase_url"
     SUPABASE_KEY = "your_supabase_key"
     ADMIN_PASSWORD = "your_admin_password"
     ```

4. **Set up Supabase**
   - Create a new project in [Supabase](https://supabase.com)
   - Enable the Vector extension in Database Settings
   - Run the following SQL in the SQL editor:
   ```sql
   -- Enable vector extension
   create extension if not exists vector;

   -- Create the documents table
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

5. **Run the Application**
   ```bash
   streamlit run chat_app.py
   ```

## Usage

### Normal Users
- Access the chat interface directly
- Ask questions and receive AI responses
- Benefit from context-aware answers based on uploaded documents

### Admin Users
1. Click "Admin Login" in the sidebar
2. Enter the admin password
3. Access document management features:
   - Upload new documents
   - View uploaded files
   - Delete documents
4. Logout when done

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Add all required secrets in Streamlit Cloud settings:
   - `TOGETHER_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `ADMIN_PASSWORD`
5. Deploy the application

### Local Deployment

For local deployment, ensure you have:
- Python 3.8 or higher
- Required packages installed
- Valid API keys for Together AI and Supabase
- Proper network connectivity
- Admin password configured

## Security Considerations

- Never commit `.streamlit/secrets.toml` to version control
- Keep API keys and admin password secure
- Rotate credentials regularly
- Use strong admin passwords
- Monitor Supabase usage and set appropriate limits
- Regularly backup your vector database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Together AI](https://together.ai) for the LLM API
- [Supabase](https://supabase.com) for vector storage
- [Streamlit](https://streamlit.io) for the web framework
- [Sentence Transformers](https://www.sbert.net) for embeddings