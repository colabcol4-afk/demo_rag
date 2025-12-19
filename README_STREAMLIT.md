# 🤖 LangGraph RAG Agent - Streamlit UI

A professional, beautiful web interface for interacting with your LangGraph RAG (Retrieval-Augmented Generation) agent.

## ✨ Features

### 📁 Document Management
- **Upload Documents**: Upload PDF, TXT, DOC, DOCX, or MD files (max 2 files, 3MB each)
- **View All Files**: See all documents in the data folder with file sizes and metadata
- **Delete Files**: Remove unwanted documents with a single click
- **Real-time Stats**: Monitor document count, parsed files, and chunk folders

### 🔄 Pipeline Controls
Three powerful buttons to manage your RAG pipeline:

1. **🚀 Parse Files**:
   - Runs the ingestion script to parse and embed documents
   - Shows live terminal output during processing
   - Updates parsed files and chunks automatically

2. **🧹 Clear Files**:
   - Clears all parsed text files from `src/parsed_text/`
   - Removes all chunk folders from `src/chunks/`
   - Perfect for starting fresh

3. **💥 Clear Vectors**:
   - Clears all vectors from the Qdrant vector store
   - Uses collection name: `document_chunks`
   - Maintains collection structure

### 💬 Chat Interface
- **Interactive Chat**: Talk to your documents using natural language
- **Weather Tool**: Ask about weather in any city
- **RAG Retrieval**: Search through your ingested documents
- **Beautiful UI**: Professional gradient design with smooth animations
- **Message History**: Full conversation context maintained

### 📊 Status Tracking
- **Documents**: Count of uploaded files
- **Parsed Files**: Number of processed documents in `src/parsed_text/`
- **Chunk Folders**: Number of chunk directories in `src/chunks/`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install with RAG support
pip install -e ".[rag]"

# Or install everything
pip install -e .
```

### 2. Set Environment Variables

Make sure your `.env` file contains:

```bash
# OpenWeatherMap API Key
OPENWEATHERMAP_API_KEY=your_key_here

# Groq API Key (for LLM)
GROQ_API_KEY=your_key_here

# Qdrant Configuration (optional if using defaults)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key

# LangSmith (optional, for tracing)
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=react-agent
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Workflow

1. **Upload Documents**
   - Click "Choose files to upload" in the sidebar
   - Select 1-2 documents (PDF, TXT, DOC, DOCX, or MD)
   - Click "💾 Save Files"

2. **Parse & Ingest**
   - Click the "🚀 Parse" button
   - Wait for the ingestion to complete
   - View the terminal output in the expandable log

3. **Chat with Your Documents**
   - Type your question in the chat input
   - The agent will automatically use the RAG tool for document questions
   - Or ask about weather: "What's the weather in London?"

4. **Manage Your Pipeline**
   - View parsed files and chunks in the expandable sections
   - Clear intermediate files with "🧹 Clear Files"
   - Reset the vector store with "💥 Clear Vectors"

## 🎨 UI Features

### Professional Design
- **Gradient Headers**: Beautiful purple-blue gradients
- **Card-Based Layout**: Clean, modern file cards with hover effects
- **Color-Coded Messages**: User messages in gradient purple, agent responses in white
- **Responsive Stats**: Real-time updates of document counts
- **Smooth Animations**: Subtle transitions and hover effects

### Organized Layout
- **Sidebar**: All document management and controls
- **Main Area**: Clean chat interface
- **Collapsible Sections**: Keep the UI clean and organized
- **Status Indicators**: Clear success/error messages

## 🛠️ Configuration

### Change LLM Model

In the sidebar, expand "⚙️ Settings" → "LLM Configuration":
- Default: `groq/openai/gpt-oss-120b`
- Format: `provider/model-name`
- Examples:
  - `groq/llama3-8b-8192`
  - `openai/gpt-4`
  - `anthropic/claude-3-sonnet`

### File Upload Limits

Edit in `app.py`:
```python
MAX_FILE_SIZE_MB = 3      # Maximum file size in MB
MAX_FILES_UPLOAD = 2      # Maximum files per upload
```

### Vector Store Collection

The Qdrant collection name is set to `document_chunks`. To change it:
```python
QDRANT_COLLECTION_NAME = "your_collection_name"
```

## 📂 Directory Structure

```
agent_demo/
├── app.py                      # Streamlit UI (main file)
├── src/
│   ├── data/                   # Uploaded documents
│   ├── parsed_text/            # Parsed text files
│   ├── chunks/                 # Document chunks (folders)
│   ├── rag/
│   │   ├── ingestion.py        # Ingestion script
│   │   └── retriever.py        # RAG retriever
│   ├── react_agent/            # LangGraph agent
│   │   ├── tools.py            # Weather & RAG tools
│   │   └── graph.py            # Agent graph
│   └── utils/
│       └── clear_store.py      # Vector store cleanup
└── .env                        # Environment variables
```

## 🔧 Troubleshooting

### Issue: "RAG retriever not available"
**Solution**: Install RAG dependencies:
```bash
pip install -e ".[rag]"
```

### Issue: "Collection 'document_chunks' not found"
**Solution**: Run the ingestion script first to create the collection:
```bash
python src/rag/ingestion.py
```

### Issue: "OpenWeatherMap API key not configured"
**Solution**: Add your API key to `.env`:
```bash
OPENWEATHERMAP_API_KEY=your_key_here
```

### Issue: Parse button shows no output
**Solution**: Check that `src/rag/ingestion.py` exists and runs successfully:
```bash
python src/rag/ingestion.py
```

## 💡 Tips

1. **Start Fresh**: Use "Clear Files" and "Clear Vectors" to reset everything
2. **Monitor Progress**: Expand the ingestion log to see what's happening
3. **Small Files**: Keep documents under 3MB for faster processing
4. **Check Stats**: The stats row shows counts for docs, parsed files, and chunks
5. **Model Selection**: Try different models for better responses

## 🎯 Advanced Usage

### Running in Production

For production deployment, consider:
- Using a production ASGI server (not `streamlit run`)
- Setting proper environment variables for API keys
- Configuring proper Qdrant instance (not free tier)
- Adjusting rate limits and timeouts

### Custom Tools

To add more tools to the agent:
1. Edit `src/react_agent/tools.py`
2. Add your tool function (must be async)
3. Append to the `TOOLS` list
4. The agent will automatically discover it!

## 📝 License

MIT License - see LICENSE file for details

---

**Built with ❤️ using Streamlit, LangGraph, and Claude**
