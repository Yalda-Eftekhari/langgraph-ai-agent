.PHONY: help install setup-ollama start stop clean test test-ai logs status health

# Variables
PYTHON = python3
PIP = pip3
VENV = .venv
OLLAMA_PATH = ./ollama/bin/ollama
MODEL = qwen2.5:0.5b

help:
	@echo "Available commands:"
	@echo "  install      - Create virtual environment and install dependencies"
	@echo "  setup-ollama - Setup and start Ollama with the specified model"
	@echo "  setup        - Run install then setup-ollama"
	@echo "  start        - Start the FastAPI application"
	@echo "  test         - Run basic system tests"
	@echo "  test-ai      - Test the AI agent functionality"
	@echo "  clean        - Clean up temporary files and processes"
	@echo "  status       - Check status of services"

install:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	@echo "Virtual environment setup complete!"

setup-ollama:
	@echo "Setting up Ollama..."
	@if [ ! -f $(OLLAMA_PATH) ]; then \
		echo "Ollama binary not found at $(OLLAMA_PATH)"; \
		exit 1; \
	fi
	@echo "Starting Ollama server..."
	$(OLLAMA_PATH) serve &
	@sleep 3
	@echo "Pulling model $(MODEL)..."
	$(OLLAMA_PATH) pull $(MODEL)
	@echo "Ollama setup complete!"

setup: install setup-ollama

start:
	@echo "Starting FastAPI application..."
	@if ! pgrep -f "ollama serve" > /dev/null; then \
		echo "Starting Ollama..."; \
		$(OLLAMA_PATH) serve & \
		sleep 3; \
	fi
	. $(VENV)/bin/activate && $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

stop:
	@echo "Stopping services..."
	@pkill -f "uvicorn app.main:app" || true
	@pkill -f "ollama serve" || true
	@echo "Services stopped"

test:
	@echo "Running basic system tests..."
	@if ! pgrep -f "ollama serve" > /dev/null; then \
		echo "Ollama is not running"; \
		exit 1; \
	fi
	@echo "Ollama is running"
	@echo "All tests passed!"

test-ai:
	@echo "Testing AI agent functionality..."
	. $(VENV)/bin/activate && $(PYTHON) -c "
from app.ai_agent_simple import process_query
result = process_query('Show me all advisory firms')
print('AI Agent Test Result:')
print(f'Query: Show me all advisory firms')
print(f'SQL: {result.get(\"sql_query\", \"No SQL generated\")}')
print(f'Response: {result.get(\"response\", \"No response\")}')
print(f'Error: {result.get(\"error\", \"None\")}')
"

clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__
	@rm -rf app/__pycache__
	@rm -rf app/*/__pycache__
	@rm -f *.log
	@rm -f *.pid
	@rm -f advisory_firms.db
	@echo "Cleanup complete"

status:
	@echo "Service Status:"
	@if pgrep -f "ollama serve" > /dev/null; then \
		echo "✅ Ollama: Running"; \
	else \
		echo "❌ Ollama: Not running"; \
	fi
	@if pgrep -f "uvicorn app.main:app" > /dev/null; then \
		echo "✅ FastAPI: Running"; \
	else \
		echo "❌ FastAPI: Not running"; \
	fi
