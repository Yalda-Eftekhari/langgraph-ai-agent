#!/usr/bin/env python3
"""
Simple test script to verify system components
"""

import requests
import time
import sys

def test_ollama():
    """Test if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print("❌ Ollama responded with status:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Ollama is not running:", e)
        return False

def test_database():
    """Test if database is accessible"""
    try:
        response = requests.get("http://localhost:8000/api/schema", timeout=5)
        if response.status_code == 200:
            print("✅ Database is accessible")
            return True
        else:
            print("❌ Database responded with status:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Database is not accessible:", e)
        return False

def test_ai_agent():
    """Test if AI agent is working"""
    try:
        response = requests.post(
            "http://localhost:8000/api/query",
            json={"query": "Show me all advisory firms"},
            timeout=10
        )
        if response.status_code == 200:
            print("✅ AI Agent is working")
            return True
        else:
            print("❌ AI Agent responded with status:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ AI Agent is not accessible:", e)
        return False

def main():
    print("🧪 Testing Advisory Firms AI Agent System...")
    print("=" * 50)
    
    # Test Ollama
    ollama_ok = test_ollama()
    
    # Test Database
    database_ok = test_database()
    
    # Test AI Agent
    ai_agent_ok = test_ai_agent()
    
    print("=" * 50)
    if all([ollama_ok, database_ok, ai_agent_ok]):
        print("🎉 All systems are working correctly!")
        print("🌐 Access the application at: http://localhost:8000")
    else:
        print("⚠️  Some systems are not working correctly.")
        print("💡 Check the logs with: docker-compose logs -f")
        sys.exit(1)

if __name__ == "__main__":
    main()
