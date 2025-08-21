#!/usr/bin/env python3
"""
Direct Database Operations Test Script
Tests database operations directly without AI agent
"""

from app.database import get_db, get_database_schema
from sqlalchemy import text

def test_direct_database_operations():
    """Test database operations directly"""
    print("�� Starting Direct Database Operations Test\n")
    
    # Get database session
    db = next(get_db())
    
    try:
        # Test 1: Get database schema
        print("=== TEST 1: Database Schema ===")
        schema = get_database_schema()
        print(f"Available tables: {list(schema.keys())}")
        print()
        
        # Test 2: Get all advisory firms
        print("=== TEST 2: Get All Advisory Firms ===")
        result = db.execute(text("SELECT * FROM advisory_firms LIMIT 10"))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        print(f"Found {len(rows)} advisory firms:")
        for firm in rows:
            print(f"  - {firm['name']} ({firm['industry']}) - Founded: {firm['founded_year']}")
        print()
        
        # Test 3: Get all clients
        print("=== TEST 3: Get All Clients ===")
        result = db.execute(text("SELECT * FROM clients LIMIT 10"))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        print(f"Found {len(rows)} clients:")
        for client in rows:
            print(f"  - {client['name']} ({client['project_type']}) - Value: ${client['value_millions']}M")
        print()
        
        # Test 4: Get all consultants
        print("=== TEST 4: Get All Consultants ===")
        result = db.execute(text("SELECT * FROM consultants LIMIT 10"))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        print(f"Found {len(rows)} consultants:")
        for consultant in rows:
            print(f"  - {consultant['name']} ({consultant['specialization']}) - Experience: {consultant['experience_years']} years")
        print()
        
        # Test 5: Filter advisory firms by industry
        print("=== TEST 5: Filter Advisory Firms by Industry ===")
        result = db.execute(text("SELECT * FROM advisory_firms WHERE industry = 'Management Consulting' LIMIT 10"))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        print(f"Found {len(rows)} firms in Management Consulting:")
        for firm in rows:
            print(f"  - {firm['name']} - Revenue: ${firm['revenue_millions']}M")
        print()
        
        # Test 6: Filter clients by project value
        print("=== TEST 6: Filter Clients by Project Value ===")
        result = db.execute(text("SELECT * FROM clients WHERE value_millions > 3 LIMIT 10"))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        print(f"Found {len(rows)} clients with value > $3M:")
        for client in rows:
            print(f"  - {client['name']} - Value: ${client['value_millions']}M")
        print()
        
        # Test 7: Add a new client
        print("=== TEST 7: Add New Client ===")
        try:
            insert_result = db.execute(text("""
                INSERT INTO clients (name, industry, firm_id, project_type, value_millions) 
                VALUES ('TestCorp Inc', 'Technology', 1, 'Research', 2.5)
            """))
            db.commit()
            print("✅ New client 'TestCorp Inc' added successfully!")
        except Exception as e:
            print(f"❌ Error adding client: {e}")
            db.rollback()
        print()
        
        # Test 8: Update a client
        print("=== TEST 8: Update Client ===")
        try:
            update_result = db.execute(text("""
                UPDATE clients SET value_millions = 3.0 WHERE name = 'TestCorp Inc'
            """))
            db.commit()
            print(f"✅ Client 'TestCorp Inc' updated successfully! Rows affected: {update_result.rowcount}")
        except Exception as e:
            print(f"❌ Error updating client: {e}")
            db.rollback()
        print()
        
        # Test 9: Verify the update
        print("=== TEST 9: Verify Update ===")
        result = db.execute(text("SELECT * FROM clients WHERE name = 'TestCorp Inc'"))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        if rows:
            client = rows[0]
            print(f"✅ Client found: {client['name']} - Value: ${client['value_millions']}M")
        else:
            print("❌ Client not found after update")
        print()
        
        # Test 10: Delete the test client
        print("=== TEST 10: Delete Test Client ===")
        try:
            delete_result = db.execute(text("DELETE FROM clients WHERE name = 'TestCorp Inc'"))
            db.commit()
            print(f"✅ Client 'TestCorp Inc' deleted successfully! Rows affected: {delete_result.rowcount}")
        except Exception as e:
            print(f"❌ Error deleting client: {e}")
            db.rollback()
        print()
        
        # Test 11: Aggregation - count clients per firm
        print("=== TEST 11: Count Clients per Firm ===")
        result = db.execute(text("""
            SELECT af.name as firm_name, COUNT(c.id) as client_count
            FROM advisory_firms af
            LEFT JOIN clients c ON af.id = c.firm_id
            GROUP BY af.id, af.name
            ORDER BY client_count DESC
        """))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        print("Client count per firm:")
        for row in rows:
            print(f"  - {row['firm_name']}: {row['client_count']} clients")
        print()
        
        # Test 12: Complex query with JOINs
        print("=== TEST 12: Complex Query with JOINs ===")
        result = db.execute(text("""
            SELECT 
                c.name as client_name,
                c.project_type,
                c.value_millions,
                af.name as firm_name,
                af.industry as firm_industry
            FROM clients c
            JOIN advisory_firms af ON c.firm_id = af.id
            WHERE c.value_millions > 0
            ORDER BY c.value_millions DESC
            LIMIT 5
        """))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        print("Top clients by project value:")
        for row in rows:
            print(f"  - {row['client_name']} ({row['project_type']}) - ${row['value_millions']}M - {row['firm_name']}")
        print()
        
        print("✅ All direct database operations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during database operations: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    test_direct_database_operations()
