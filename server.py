import os
import re
import json
import asyncio
from typing import List, Optional, Any, Dict

import psycopg
from psycopg.rows import dict_row
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

import vwcd

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP
mcp_host = os.getenv("MCP_HOST", "0.0.0.0")
mcp_port = int(os.getenv("MCP_PORT", "8000"))
mcp = FastMCP(name="NetworkDataAnalyser", host=mcp_host, port=mcp_port)

# Database connection management
async def get_db_connection():
    """
    Establishes an asynchronous connection to the PostgreSQL database using credentials from environment variables.
    Returns a psycopg AsyncConnection object with a dictionary row factory.
    """
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "password")

    conn_info = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"
    return await psycopg.AsyncConnection.connect(conn_info, row_factory=dict_row)

# --- Category A: Introspection Tools ---

@mcp.tool(description="Lists all available metric tables and their numeric columns in the database.")
async def list_available_metrics() -> str:
    """
    Lists all available metric tables and their numeric columns in the database.
    
    This tool scans the 'monitoramento' schema for tables ending in '_metrics' and identifies 
    columns with numeric data types (integer, bigint, double precision, etc.).
    
    Returns:
        A JSON string where keys are table names and values are lists of numeric column names.
    """
    conn = await get_db_connection()
    try:
        async with conn.cursor() as cur:
            # Find tables ending in _metrics in monitoramento schema
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'monitoramento' 
                AND table_name LIKE '%_metrics'
            """
            await cur.execute(tables_query)
            tables_records = await cur.fetchall()
            tables = [r['table_name'] for r in tables_records]
            
            result = {}
            for table in tables:
                # Find numeric columns for each table
                cols_query = """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'monitoramento' 
                    AND table_name = %s 
                    AND data_type IN ('integer', 'bigint', 'double precision', 'real', 'numeric', 'smallint')
                """
                await cur.execute(cols_query, (table,))
                cols_records = await cur.fetchall()
                result[table] = [r['column_name'] for r in cols_records]
                
            return json.dumps(result, indent=2)
    finally:
        await conn.close()

@mcp.tool(description="Retrieves the schema information (column names and data types) for a specific table.")
async def describe_table_schema(table_name: str) -> str:
    """
    Retrieves the schema information (column names and data types) for a specific table.
    
    Args:
        table_name: The name of the table to describe (must be in the 'monitoramento' schema).
        
    Returns:
        A markdown-formatted table listing the column names and their data types.
    """
    conn = await get_db_connection()
    try:
        async with conn.cursor() as cur:
            # Check if table exists to avoid SQL injection in schema queries implicitly
            query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'monitoramento' 
                AND table_name = %s
            """
            await cur.execute(query, (table_name,))
            records = await cur.fetchall()
            
            if not records:
                return f"Table '{table_name}' not found in schema 'monitoramento'."
                
            df = pd.DataFrame([dict(r) for r in records])
            return df.to_markdown(index=False)
    finally:
        await conn.close()

# --- Category B: Relationship & Inventory Tools ---

@mcp.tool(description="Retrieves a summary of all routers in the network inventory.")
async def get_network_inventory() -> str:
    """
    Retrieves a summary of all routers in the network inventory.
    
    This tool queries the 'monitoramento.routers' table to get details like location and model 
    of all registered routers.
    
    Returns:
        A markdown-formatted table of the routers inventory.
    """
    conn = await get_db_connection()
    try:
        async with conn.cursor() as cur:
            query = "SELECT * FROM monitoramento.routers"
            try:
                await cur.execute(query)
                records = await cur.fetchall()
                df = pd.DataFrame([dict(r) for r in records])
                if df.empty:
                    return "No routers found in inventory."
                return df.to_markdown(index=False)
            except psycopg.errors.UndefinedTable:
                return "Table 'monitoramento.routers' does not exist."
    finally:
        await conn.close()

@mcp.tool(description="Lists all devices connected to a specific router.")
async def list_router_devices(router_mac: str) -> str:
    """
    Lists all devices connected to a specific router.
    
    Args:
        router_mac: The MAC address of the router.
        
    Returns:
        A markdown-formatted table of devices connected to the specified router.
    """
    conn = await get_db_connection()
    try:
        async with conn.cursor() as cur:
            query = "SELECT * FROM monitoramento.devices WHERE router_mac = %s"
            try:
                await cur.execute(query, (router_mac,))
                records = await cur.fetchall()
                
                if not records:
                    return f"No devices found for router {router_mac} (or router not found)."
                    
                df = pd.DataFrame([dict(r) for r in records])
                return df.to_markdown(index=False)
            except psycopg.errors.UndefinedColumn:
                return "Column 'router_mac' does not exist in 'monitoramento.devices'. Please check schema."
            except psycopg.errors.UndefinedTable:
                return "Table 'monitoramento.devices' does not exist."
    finally:
        await conn.close()

@mcp.tool(description="Retrieves detailed information about a specific device, including its parent router details.")
async def get_device_full_info(device_mac: str) -> str:
    """
    Retrieves detailed information about a specific device, including its parent router details.
    
    Args:
        device_mac: The MAC address of the device to query.
        
    Returns:
        A JSON string containing the device's metadata (OS, vendor, etc.) and its connected router's details.
    """
    conn = await get_db_connection()
    try:
        async with conn.cursor() as cur:
            # Joining devices and routers
            query = """
                SELECT d.*, r.model as router_model, r.location as router_location, r.isp as router_isp
                FROM monitoramento.devices d
                JOIN monitoramento.routers r ON d.router_mac = r.mac_address
                WHERE d.mac_address = %s
            """
            
            await cur.execute(query, (device_mac,))
            record = await cur.fetchone()
            
            if not record:
                return f"Device {device_mac} not found or not linked to a known router."
                
            return json.dumps(dict(record), indent=2, default=str)
    finally:
        await conn.close()

# --- Category C: Time-Series Analysis Tools ---

@mcp.tool(description="Computes aggregated metrics (average and maximum) for a given device over a specified time window.")
async def get_metrics_aggregate(
    table_name: str, 
    column_names: List[str], 
    mac_address: str, 
    start_time: str, 
    end_time: str, 
    interval: str
) -> str:
    """
    Computes aggregated metrics (average and maximum) for a given device over a specified time window.
    
    Args:
        table_name: The name of the metrics table (must end in '_metrics').
        column_names: A list of numeric columns to aggregate.
        mac_address: The MAC address of the target device/router.
        start_time: The start of the time window (ISO 8601 string).
        end_time: The end of the time window (ISO 8601 string).
        interval: The aggregation interval (e.g., '1 hour', '15 minutes').
        
    Returns:
        A markdown-formatted table containing the time buckets and aggregated values.
    """
    # Validation
    if not table_name.endswith('_metrics'):
        return "Error: table_name must end with '_metrics'"
    
    # Safe guard against SQL injection in table/column names
    if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
        return "Invalid table name."
    for col in column_names:
        if not re.match(r'^[a-zA-Z0-9_]+$', col):
            return f"Invalid column name: {col}"

    conn = await get_db_connection()
    try:
        async with conn.cursor() as cur:
            # Check column existence for MAC address
            check_col_query = """
                SELECT column_name FROM information_schema.columns 
                WHERE table_schema = 'monitoramento' AND table_name = %s 
                AND column_name IN ('mac_address', 'router_mac', 'device_mac')
            """
            await cur.execute(check_col_query, (table_name,))
            mac_col_record = await cur.fetchone()
            if not mac_col_record:
                return f"Could not determine MAC address column for table {table_name}."
            target_mac_col = mac_col_record['column_name']

            # Construct Aggregates
            aggs = ", ".join([f"AVG({col}) as avg_{col}, MAX({col}) as max_{col}" for col in column_names])
            
            # Using %s for parameters
            query = f"""
                SELECT time_bucket(%s, timestamp) AS bucket, {aggs}
                FROM monitoramento.{table_name}
                WHERE {target_mac_col} = %s
                AND timestamp BETWEEN %s::timestamp AND %s::timestamp
                GROUP BY bucket
                ORDER BY bucket
            """
            
            await cur.execute(query, (interval, mac_address, start_time, end_time))
            records = await cur.fetchall()
            
            if not records:
                return "No data found for the specified criteria."
                
            df = pd.DataFrame([dict(r) for r in records])
            return df.to_markdown(index=False)
        
    except Exception as e:
        return f"Error executing aggregate query: {str(e)}"
    finally:
        await conn.close()

@mcp.tool(description="Execute a SQL query and detect change points on a specific metric column using the VWCD algorithm.")
async def analyze_change_points_from_sql(sql_query: str, metric_column: str) -> str:
    """
    Execute a SQL query and detect change points on a specific metric column using the VWCD algorithm.
    
    Safety:
        - Rejects queries containing modification keywords (DROP, DELETE, UPDATE, etc.).
        - Enforces a read-only transaction.
        - Sets a 30-second statement timeout.
        
    Args:
        sql_query: The SQL query to fetch the data.
        metric_column: The name of the column containing the metric to analyze.
        
    Returns:
        A JSON string containing the change points and segments statistics.
    """
    # Safeguard: Regex Blocklist (Same as execute_readonly_query)
    forbidden_patterns = [
        r'\bDROP\b', r'\bDELETE\b', r'\bUPDATE\b', r'\bINSERT\b', 
        r'\bTRUNCATE\b', r'\bALTER\b', r'\bGRANT\b', r'\bREVOKE\b'
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, sql_query, re.IGNORECASE):
            return "Query rejected: Contains forbidden keywords (DROP, DELETE, UPDATE, etc.)."

    conn = await get_db_connection()
    try:
        # Safeguard: SET TRANSACTION READ ONLY and timeout
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION READ ONLY;")
                await cur.execute("SET statement_timeout = '30s';")
                
                await cur.execute(sql_query)
                records = await cur.fetchall()
                
                if not records:
                    return "No data found."
                
                df = pd.DataFrame([dict(r) for r in records])
                
                if metric_column not in df.columns:
                    return f"Metric column '{metric_column}' not found. Available columns: {list(df.columns)}"
                
                # Ensure metric is numeric and drop NaNs
                data = pd.to_numeric(df[metric_column], errors='coerce').dropna().tolist()
                
                if not data:
                    return "Error: No valid numeric data found in the specified column."

                # Convert to numpy array for VWCD
                X = np.array(data)
                
                # Run VWCD
                # Since vwcd is CPU bound and synchronous, strictly speaking we might want to run it in a thread 
                # if it blocks the event loop for too long, but for an MCP tool this might be acceptable for now.
                CP, _, _, elapsed = vwcd.vwcd(X)
                
                # Get segments with statistical description
                segments = vwcd.get_segments(X, CP)
                    
                return json.dumps({
                    "change_points": [int(cp) for cp in CP],
                    "segments": segments,
                    "elapsed_time_ms": elapsed * 1000
                })

    except psycopg.Error as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error analyzing change points: {str(e)}"
    finally:
        await conn.close()

# --- Category D: Query Sandbox ---

@mcp.tool(description="Executes a read-only SQL query provided by the user.")
async def execute_readonly_query(query: str) -> str:
    """
    Executes a read-only SQL query provided by the user.
    
    Safety:
        - Rejects queries containing modification keywords (DROP, DELETE, UPDATE, etc.).
        - Enforces a read-only transaction.
        - Sets a 30-second statement timeout.
        
    Args:
        query: The SQL SELECT query string to execute.
        
    Returns:
        A markdown-formatted table of the query results.
    """
    # Safeguard 2: Regex Blocklist
    forbidden_patterns = [
        r'\bDROP\b', r'\bDELETE\b', r'\bUPDATE\b', r'\bINSERT\b', 
        r'\bTRUNCATE\b', r'\bALTER\b', r'\bGRANT\b', r'\bREVOKE\b'
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return "Query rejected: Contains forbidden keywords (DROP, DELETE, UPDATE, etc.)."

    conn = await get_db_connection()
    try:
        # Safeguard 1: SET TRANSACTION READ ONLY
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION READ ONLY;")
                
                # Safeguard 3: statement_timeout (30s)
                await cur.execute("SET statement_timeout = '30s';")
                
                # Executing query
                await cur.execute(query)
                records = await cur.fetchall()
                
                if not records:
                    return "Query returned no results."
                    
                df = pd.DataFrame([dict(r) for r in records])
                return df.to_markdown(index=False)

    except psycopg.Error as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        await conn.close()

# --- Category E: Prompts ---

@mcp.prompt()
def audit_network_prompt() -> str:
    """
    Generates a prompt to audit the entire network inventory and available metrics.
    """
    return """I would like to perform a comprehensive audit of the network. 
Please start by:
1. Retrieving the full network inventory (routers).
2. Listing all available metrics tables in the database.
3. Based on the findings, suggest what key performance indicators we should check."""

@mcp.prompt()
def diagnose_router_prompt(router_mac: str) -> str:
    """
    Generates a prompt to diagnose a specific router and its connected devices.
    """
    return f"""I need to diagnose issues with router {router_mac}.
Please execute the following steps:
1. Retrieve details for this router.
2. List all devices connected to this router.
3. If metric tables are available, check the aggregate metrics for this router for the last 1 hour."""

@mcp.prompt()
def metric_analysis_prompt(table_name: str, mac_address: str) -> str:
    """
    Generates a prompt to analyze specific metrics for a device over time.
    """
    return f"""Please analyze the metrics in table '{table_name}' for device '{mac_address}'.
I need to see the average and maximum values aggregated by hour for the last 24 hours. 
Please use the 'get_metrics_aggregate' tool to fetch this data."""

if __name__ == "__main__":
    mcp.run(transport='sse')
