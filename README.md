# Advisory Firms AI Agent

A sophisticated AI agent built with LangGraph that can interact with a SQLite database to perform complex data operations based on natural language requests. This project demonstrates advanced AI development, database integration, and modern software engineering practices.

## 🚀 Features

- **Natural Language Interface**: Users can ask questions in plain English
- **AI-Powered Query Generation**: Automatically converts natural language to SQL
- **Database Schema Understanding**: The agent automatically understands database structure
- **Multi-Table Operations**: Supports complex queries across advisory firms, clients, and consultants
- **Error Handling**: Robust error handling with fallback mechanisms
- **Conversational Memory**: Maintains context across multiple interactions
- **Modern Web Interface**: Clean, responsive chatbot interface

## 🏗️ Architecture

### Technology Stack

- **Backend**: FastAPI (Python 3.11)
- **AI Framework**: LangChain + Custom AI Agent (LangGraph-inspired)
- **LLM**: Ollama (local deployment)
- **Database**: SQLite
- **ORM**: SQLAlchemy
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Project Management**: Make

### Why These Choices?

1. **FastAPI**: Chosen for its high performance, automatic API documentation, and modern async support
2. **Custom AI Agent**: Implements LangGraph-inspired workflow patterns with natural language understanding and SQL generation
3. **SQLite**: Lightweight, file-based database with excellent support for complex queries and relationships
4. **Ollama**: Local LLM deployment ensures data privacy and eliminates API costs
5. **Docker**: Ensures consistent deployment across environments

### System Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI App   │    │   SQLite        │
│   (HTML/JS)    │◄──►│   + LangGraph   │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Ollama LLM    │
                       │   (Local)       │
                       └─────────────────┘
```

## 🗄️ Database Schema

The system includes three main tables with realistic advisory firm data:

- **advisory_firms**: Company information, revenue, services, etc.
- **clients**: Client projects and relationships
- **consultants**: Professional profiles and specializations

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- Make

### 1. Clone and Setup

```bash
git clone <repository-url>
cd langgraph
```

### 2. Complete Setup (One Command!)

```bash
make setup
```

This will:
- Create virtual environment
- Install all dependencies
- Setup Ollama with required model
- Prepare everything for launch

**Note**: Make sure you have extracted the `ollama-linux-amd64.tgz.1` file to the `ollama/` directory first.

### 3. Start the Application

```bash
make start
```

### 4. Access the Application

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
   make stop
   make setup-ollama
   make start
   ```

3. **Database Issues**:
   ```bash
   make stop
   docker-compose down -v
   make start
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

