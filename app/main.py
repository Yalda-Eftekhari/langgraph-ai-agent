from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any
import json
import os
import re

from .database import get_db, get_database_schema, init_database
from .ai_agent import process_query

app = FastAPI(title="Advisory Firms AI Agent", version="1.0.0")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database with sample data on startup"""
    init_database()

# Templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chatbot interface"""
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.post("/api/query")
async def process_user_query(
    request: Request,
    db: Session = Depends(get_db)
):
    """Process a natural language query through the AI agent and execute database operations"""
    try:
        body = await request.json()
        user_query = body.get("query", "")
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Process the query through the AI agent
        result = await process_query(user_query)
        
        # If there's an error in the AI agent, return it
        if result.get("error"):
            return {
                "response": f"I couldn't process your request: {result['error']}",
                "sql_query": "",
                "error": result["error"],
                "data": [],
                "conversation": result.get("conversation", [])
            }
        
        # If there's a SQL query, try to execute it
        if result.get("sql_query"):
            try:
                sql_query = result["sql_query"]
                
                # Additional security validation
                if not is_safe_query(sql_query):
                    return {
                        "response": "This operation is not allowed for security reasons.",
                        "sql_query": sql_query,
                        "error": "Security validation failed",
                        "data": [],
                        "conversation": result.get("conversation", [])
                    }
                
                # Execute the SQL query
                db_result = db.execute(text(sql_query))
                
                # Handle different query types
                if sql_query.strip().upper().startswith("SELECT"):
                    # Fetch results for SELECT queries
                    columns = db_result.keys()
                    rows = [dict(zip(columns, row)) for row in db_result.fetchall()]
                    
                    if rows:
                        result["data"] = rows
                        result["response"] = f"Query executed successfully! Found {len(rows)} results."
                    else:
                        result["data"] = []
                        result["response"] = "Query executed successfully, but no results were found."
                        
                elif sql_query.strip().upper().startswith("INSERT"):
                    # For INSERT queries, commit the transaction
                    db.commit()
                    result["data"] = []
                    result["response"] = "Record inserted successfully!"
                    
                elif sql_query.strip().upper().startswith("UPDATE"):
                    # For UPDATE queries, commit the transaction
                    db.commit()
                    result["data"] = []
                    result["response"] = "Record(s) updated successfully!"
                    
                elif sql_query.strip().upper().startswith("DELETE"):
                    # For DELETE queries, commit the transaction
                    db.commit()
                    result["data"] = []
                    result["response"] = "Record(s) deleted successfully!"
                    
                else:
                    # For other query types
                    db.commit()
                    result["data"] = []
                    result["response"] = "Query executed successfully!"
                
                result["error"] = ""
                
            except Exception as e:
                db.rollback()
                error_msg = str(e)
                
                # Provide user-friendly error messages
                if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
                    result["response"] = "I couldn't find the table you're referring to. Please check the table name."
                elif "column" in error_msg.lower() and "does not exist" in error_msg.lower():
                    result["response"] = "I couldn't find the column you're referring to. Please check the column name."
                elif "syntax error" in error_msg.lower():
                    result["response"] = "There was a syntax error in the generated query. Please try rephrasing your request."
                elif "foreign key" in error_msg.lower():
                    result["response"] = "This operation would violate database constraints. Please check your request."
                else:
                    result["response"] = f"I encountered an error executing the query: {error_msg}"
                
                result["error"] = f"Database execution error: {error_msg}"
                result["data"] = []
        else:
            # No SQL query generated
            result["data"] = []
            result["error"] = ""
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def is_safe_query(sql_query: str) -> bool:
    """Check if the SQL query is safe to execute"""
    sql_upper = sql_query.upper()
    
    # Block dangerous operations
    dangerous_keywords = [
        "DROP", "TRUNCATE", "ALTER", "CREATE TABLE", "CREATE DATABASE", 
        "CREATE SCHEMA", "GRANT", "REVOKE", "EXECUTE", "EXEC"
    ]
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False
    
    # Block operations without WHERE clauses for UPDATE/DELETE
    if sql_upper.startswith("UPDATE") and "WHERE" not in sql_upper:
        return False
    
    if sql_upper.startswith("DELETE") and "WHERE" not in sql_upper:
        return False
    
    return True

@app.get("/api/schema")
async def get_schema():
    """Get the database schema information"""
    return get_database_schema()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

