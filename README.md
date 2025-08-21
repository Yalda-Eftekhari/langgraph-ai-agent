# Advisory Firms AI Agent

A sophisticated AI agent built with LangGraph that can interact with a SQLite database to perform complex data operations based on natural language requests.

## 🚀 Features

- **Natural Language Interface**: Users can ask questions in plain English
- **AI-Powered Query Generation**: Automatically converts natural language to SQL
- **Database Schema Understanding**: The agent automatically understands database structure
- **Multi-Table Operations**: Supports complex queries across advisory firms, clients, and consultants
- **Error Handling**: Robust error handling with fallback mechanisms
- **Conversational Memory**: Maintains context across multiple interactions
- **Modern Web Interface**: Clean, responsive chatbot interface

## 🏗️ Technology Stack

- **Backend**: FastAPI (Python 3.11)
- **AI Framework**: LangChain + Custom AI Agent (LangGraph-inspired)
- **LLM**: Ollama (local deployment)
- **Database**: SQLite
- **ORM**: SQLAlchemy

## 🤔 Why Ollama?

Local LLM deployment ensures data privacy, eliminates API costs, and avoids regional restrictions that can occur with cloud-based services like OpenAI and Anthropic.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- Make
- Ollama (for local LLM deployment)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd langgraph
```

### 2. Setup Ollama

First, you need to set up Ollama for local LLM deployment:

```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama2:7b  # or any other model you prefer
```

### 3. Complete Setup (One Command!)

```bash
make setup
```

This will:
- Create virtual environment
- Install all dependencies
- Prepare everything for launch

### 4. Start the Application

```bash
make start
```

### 5. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 💬 Usage Examples

The AI agent can handle various types of queries:

### Data Retrieval
- "Show me all advisory firms"
- "Find consultants with more than 10 years of experience"
- "List clients in the technology industry"

### Data Creation
- "Add a new advisory firm called TechAdvisors"
- "Create a new client project for ABC Corp"

### Data Updates
- "Update the revenue for McKinsey to 13000 million"
- "Change the headquarters of Bain to San Francisco"

### Complex Queries
- "Show me firms with revenue over 5000 million and more than 10000 employees"
- "Find consultants with PMP certification working for management consulting firms"

## 🔧 Development

### Makefile Commands

The project includes a comprehensive Makefile for easy management:

```bash
# Setup & Installation
make setup          # Complete setup (dependencies + Ollama)
make install        # Install Python dependencies only
make setup-ollama   # Setup Ollama with required model

# Application Management
make start          # Start all services
make stop           # Stop all services
make restart        # Restart all services
make logs           # View application logs

# Testing & Monitoring
make test           # Run complete system tests
make test-ai        # Test AI agent functionality
make status         # Check service status
make health         # Health check of all services

# Development
make build          # Build Docker image
make clean          # Clean up containers and images
make deploy         # Deploy to production (placeholder)

# Help
make help           # Show all available commands
```

### Project Structure

```
langgraph/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── database.py          # Database models and connection
│   ├── ai_agent.py          # AI agent (LangGraph-inspired)
│   └── templates/
│       └── chatbot.html     # Web interface
├── tests/                   # Test files
│   ├── test_database_operations.py
│   ├── test_direct_db_operations.py
│   └── test_system.py
├── requirements.txt          # Python dependencies
├── Makefile                 # Project management commands
├── .gitignore              # Git ignore rules
└── .github/workflows/       # CI/CD pipeline
```

### Running Tests

```bash
# Test the complete system
make test

# Test just the AI agent
make test-ai

# Check system status
make status
```

### Code Quality

The project follows SOLID principles and includes:
- Type hints throughout
- Comprehensive error handling
- Clean separation of concerns
- Modular architecture

## 🐳 Docker

### Building the Image

```bash
docker build -t advisory-firms-ai-agent .
```

### Running with Make

```bash
# Start all services
make start

# View logs
make logs

# Stop services
make stop

# Restart services
make restart

# Check status
make status

# Health check
make health
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:
1. **Tests**: Runs unit tests against SQLite
2. **Lints**: Checks code quality with flake8
3. **Builds**: Creates Docker images
4. **Deploys**: Pushes to Docker Hub (on main branch)

## 🛠️ Configuration

### Troubleshooting

If you encounter issues:

1. **Virtual Environment Problems**:
   ```bash
   make clean
   make setup
   ```

2. **Ollama Issues**:
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Restart Ollama service
   sudo systemctl restart ollama
   
   # Or manually start
   ollama serve
   
   # Check model availability
   ollama pull llama2:7b
   ```

3. **Database Issues**:
   ```bash
   # SQLite database is automatically created
   # Check if database file exists
   ls -la *.db
   ```

4. **Check System Status**:
   ```bash
   make status
   make health
   ```

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `OLLAMA_BASE_URL`: Ollama service URL

### Database Configuration

Default SQLite setup:
- Database: `advisory_firms.db`
- File-based storage
- No authentication required
- Automatic creation on first run

## 📊 Performance

- **Response Time**: < 2 seconds for most queries
- **Concurrent Users**: Supports multiple simultaneous users
- **Database**: Optimized queries with proper indexing
- **Memory**: Efficient conversation state management

## 🔒 Security

- Input validation and sanitization
- SQL injection prevention
- Restricted database operations (no DROP/TRUNCATE)
- Environment-based configuration

## 🚧 Future Enhancements

- User authentication and authorization
- Advanced analytics and reporting
- Integration with external data sources
- Real-time notifications
- Mobile application

