import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import server

# Mocking the connection context manager
class MockCursor:
    def __init__(self):
        self.execute = AsyncMock()
        self.fetchall = AsyncMock(return_value=[])
        self.fetchone = AsyncMock(return_value=None)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockConnection:
    def __init__(self):
        self.cursor = MagicMock(return_value=MockCursor())
        self.close = AsyncMock()
        self.transaction = MagicMock()
    
    async def __aenter__(self):
        return self # For transaction
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# We need to mock get_db_connection in server.py
@pytest.fixture
def mock_db_conn():
    with patch('server.get_db_connection', new_callable=AsyncMock) as mock:
        mock_conn = MockConnection()
        # Setup transaction context manager
        mock_conn.transaction.return_value = mock_conn
        
        # Setup cursor context manager return
        mock_cursor = MockCursor()
        mock_conn.cursor.return_value = mock_cursor
        
        mock.return_value = mock_conn
        yield mock_conn, mock_cursor

@pytest.mark.asyncio
async def test_execute_readonly_query_blocklist():
    # Test forbidden keywords
    forbidden = ["DROP TABLE x", "DELETE FROM y", "UPDATE z SET a=1", "INSERT INTO a VALUES(1)", "TRUNCATE table", "GRANT ALL", "REVOKE ALL", "ALTER TABLE"]
    for q in forbidden:
        result = await server.execute_readonly_query(q)
        assert "Query rejected" in result, f"Query '{q}' was not rejected"

@pytest.mark.asyncio
async def test_execute_readonly_query_enforces_readonly(mock_db_conn):
    conn, cursor = mock_db_conn
    
    # Valid query
    await server.execute_readonly_query("SELECT * FROM routers")
    
    # Check if READ ONLY was set
    cursor.execute.assert_any_call("SET TRANSACTION READ ONLY;")
    cursor.execute.assert_any_call("SET statement_timeout = '30s';")

@pytest.mark.asyncio
async def test_get_metrics_aggregate_injection_protection():
    # Bad table name (passes suffix check, fails regex check)
    result = await server.get_metrics_aggregate("router; DROP_metrics", ["cpu_usage"], "00:00:00:00:00:00", "2023-01-01", "2023-01-02", "1 hour")
    assert "Invalid table name" in result

    # Bad column name
    result = await server.get_metrics_aggregate("router_metrics", ["cpu_usage; --"], "00:00:00:00:00:00", "2023-01-01", "2023-01-02", "1 hour")
    assert "Invalid column name" in result

@pytest.mark.asyncio
async def test_get_metrics_aggregate_success(mock_db_conn):
    conn, cursor = mock_db_conn
    
    # Mock finding the mac column
    # For router_metrics, schema says 'router_mac'
    cursor.fetchone.return_value = {'column_name': 'router_mac'}
    # Mock data return
    cursor.fetchall.return_value = [{'bucket': '2023-01-01 10:00:00', 'avg_cpu_usage': 10, 'max_cpu_usage': 20}]
    
    result = await server.get_metrics_aggregate("router_metrics", ["cpu_usage"], "00:11:22:33:44:55", "2023-01-01T00:00:00", "2023-01-02T00:00:00", "1 hour")
    
    # Verify the SQL construction (checking parameter usage)
    call_args = cursor.execute.call_args_list[-1]
    sql_executed = call_args[0][0]
    
    assert "SELECT time_bucket(%s, timestamp)" in sql_executed
    assert "avg_cpu_usage" in sql_executed
    assert "max_cpu_usage" in sql_executed
    assert "router_mac =" in sql_executed
    assert "monitoramento.router_metrics" in sql_executed

@pytest.mark.asyncio
async def test_list_available_metrics_real_schema(mock_db_conn):
    conn, cursor = mock_db_conn
    
    # Mock tables query
    # Based on schema provided
    cursor.fetchall.side_effect = [
        # First call: list tables ending in _metrics
        [
            {'table_name': 'device_metrics'}, 
            {'table_name': 'lan_metrics'}, 
            {'table_name': 'optical_metrics'},
            {'table_name': 'router_metrics'},
            {'table_name': 'wan_metrics'},
            {'table_name': 'wifi_2_metrics'},
            {'table_name': 'wifi_5_metrics'}
        ], 
        # Subsequent calls: list numeric columns for each table (simplified for test)
        [{'column_name': 'bytes_up'}, {'column_name': 'bytes_down'}], # device_metrics
        [{'column_name': 'bytes_up'}, {'column_name': 'bytes_down'}], # lan_metrics
        [{'column_name': 'txpower'}, {'column_name': 'rxpower'}],     # optical_metrics
        [{'column_name': 'cpu_usage'}, {'column_name': 'memory'}],    # router_metrics
        [{'column_name': 'bytes_up'}, {'column_name': 'bytes_down'}], # wan_metrics
        [{'column_name': 'bytes_up'}, {'column_name': 'bytes_down'}], # wifi_2_metrics
        [{'column_name': 'bytes_up'}, {'column_name': 'bytes_down'}], # wifi_5_metrics
    ]
    
    result = await server.list_available_metrics()
    data = json.loads(result)
    
    assert 'router_metrics' in data
    assert 'cpu_usage' in data['router_metrics']
    assert 'wan_metrics' in data
    assert 'bytes_up' in data['wan_metrics']
    assert len(data) == 7

@pytest.mark.asyncio
async def test_describe_table_schema_real_columns(mock_db_conn):
    conn, cursor = mock_db_conn
    
    # Mocking response for 'routers' table
    cursor.fetchall.return_value = [
        {'column_name': 'mac_address', 'data_type': 'macaddr'},
        {'column_name': 'manufacturer', 'data_type': 'character varying'},
        {'column_name': 'model', 'data_type': 'text'}
    ]
    
    result = await server.describe_table_schema("routers")
    assert "mac_address" in result
    assert "macaddr" in result
    assert "manufacturer" in result

@pytest.mark.asyncio
async def test_get_network_inventory(mock_db_conn):
    conn, cursor = mock_db_conn
    cursor.fetchall.return_value = [
        {'mac_address': '00:00:00:00:00:01', 'model': 'FiberHome', 'location': 'Office'}
    ]
    result = await server.get_network_inventory()
    assert "FiberHome" in result
    assert "00:00:00:00:00:01" in result

@pytest.mark.asyncio
async def test_list_router_devices(mock_db_conn):
    conn, cursor = mock_db_conn
    cursor.fetchall.return_value = [{'device_mac': '11:11:11:11:11:11', 'type': 'Phone', 'host_name': 'iPhone'}]
    result = await server.list_router_devices("00:00:00:00:00:01")
    assert "iPhone" in result

@pytest.mark.asyncio
async def test_get_device_full_info(mock_db_conn):
    conn, cursor = mock_db_conn
    cursor.fetchone.return_value = {
        'device_mac': '11:11:11:11:11:11', 
        'router_model': 'FiberHome', 
        'router_location': 'Office',
        'os': 'iOS'
    }
    result = await server.get_device_full_info("11:11:11:11:11:11")
    assert "FiberHome" in result
    assert "iOS" in result

@pytest.mark.asyncio
async def test_analyze_change_points_from_sql_success(mock_db_conn):
    conn, cursor = mock_db_conn
    
    # Mock data return: a simple step function
    # 0,0,0,0,0, 10,10,10,10,10
    data = [{'val': 0} for _ in range(30)] + [{'val': 10} for _ in range(30)]
    cursor.fetchall.return_value = data
    
    # We need to mock vwcd because it's computationally intensive and we want to test the server integration logic
    with patch('vwcd.vwcd') as mock_vwcd, patch('vwcd.get_segments') as mock_segments:
        mock_vwcd.return_value = ([29], [], [], 0.01)
        mock_segments.return_value = [{'segment_index': 0}]
        
        result = await server.analyze_change_points_from_sql("SELECT val FROM t", "val")
        
        # Verify result structure
        res_json = json.loads(result)
        assert res_json['change_points'] == [29]
        assert len(res_json['segments']) == 1
        
        # Verify SQL safety was called
        cursor.execute.assert_any_call("SET TRANSACTION READ ONLY;")

@pytest.mark.asyncio
async def test_analyze_change_points_from_sql_safety(mock_db_conn):
    result = await server.analyze_change_points_from_sql("DELETE FROM t", "val")
    assert "Query rejected" in result