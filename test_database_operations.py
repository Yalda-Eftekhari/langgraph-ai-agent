#!/usr/bin/env python3
"""
Comprehensive Database Operations Test Script
Tests all CRUD operations with the advisory firms database
"""

import asyncio
import json
from app.database import get_db, get_database_schema
from app.ai_agent import process_query

async def test_database_operations():
    """Test all database operations"""
    print("🚀 Starting Comprehensive Database Operations Test\n")
    
    # Test 1: Get database schema
    print("=== TEST 1: Database Schema ===")
    schema = get_database_schema()
    print(f"Available tables: {list(schema.keys())}")
    print(f"Advisory firms columns: {[col['name'] for col in schema.get('advisory_firms', {}).get('columns', [])]}")
    print(f"Clients columns: {[col['name'] for col in schema.get('clients', {}).get('columns', [])]}")
    print(f"Consultants columns: {[col['name'] for col in schema.get('consultants', {}).get('columns', [])]}")
    print()
    
    # Test 2: Get all advisory firms
    print("=== TEST 2: Get All Advisory Firms ===")
    try:
        result = await process_query("give me advisory_firms")
        print(f"Response: {result['response']}")
        if result.get('data'):
            print(f"Found {len(result['data'])} advisory firms")
            for firm in result['data']:
                print(f"  - {firm['name']} ({firm['industry']})")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test 3: Get all clients
    print("=== TEST 3: Get All Clients ===")
    try:
        result = await process_query("give me clients")
        print(f"Response: {result['response']}")
        if result.get('data'):
            print(f"Found {len(result['data'])} clients")
            for client in result['data']:
                print(f"  - {client['name']} ({client['project_type']})")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test 4: Get all consultants
    print("=== TEST 4: Get All Consultants ===")
    try:
        result = await process_query("give me consultants")
        print(f"Response: {result['response']}")
        if result.get('data'):
            print(f"Found {len(result['data'])} consultants")
            for consultant in result['data']:
                print(f"  - {consultant['name']} ({consultant['specialization']})")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test 5: Filter advisory firms by industry
    print("=== TEST 5: Filter Advisory Firms by Industry ===")
    try:
        result = await process_query("find advisory_firms with industry Management Consulting")
        print(f"Response: {result['response']}")
        if result.get('data'):
            print(f"Found {len(result['data'])} firms in Management Consulting")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test 6: Filter clients by project value
    print("=== TEST 6: Filter Clients by Project Value ===")
    try:
        result = await process_query("find clients with value_millions greater than 3")
        print(f"Response: {result['response']}")
        if result.get('data'):
            print(f"Found {len(result['data'])} clients with value > 3M")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test 7: Add a new client
    print("=== TEST 7: Add New Client ===")
    try:
        result = await process_query("add a new client named TestCorp with industry Technology, project_type Research, value_millions 2.5")
        print(f"Response: {result['response']}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test 8: Update a client
    print("=== TEST 8: Update Client ===")
    try:
        result = await process_query("update client TestCorp set value_millions to 3.0")
        print(f"Response: {result['response']}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test 9: Delete a client
    print("=== TEST 9: Delete Client ===")
    try:
        result = await process_query("delete client with name TestCorp")
        print(f"Response: {result['response']}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test 10: Aggregation - count clients per firm
    print("=== TEST 10: Count Clients per Firm ===")
    try:
        result = await process_query("count clients per firm")
        print(f"Response: {result['response']}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    print("✅ Database operations test completed!")

if __name__ == "__main__":
    asyncio.run(test_database_operations())
