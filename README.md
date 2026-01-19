# Network Data Analyser MCP Server

This is a **Model Context Protocol (MCP)** server designed to expose network monitoring data from a PostgreSQL database to LLM clients (like Claude or other MCP-compliant agents). It provides tools for introspection, network inventory, and time-series metric analysis.

## Features

- **Introspection:** Discover available metric tables and schema details.
- **Inventory:** List routers and connected devices.
- **Analysis:** specific aggregations (time-series) for network metrics (e.g., bandwidth, CPU usage).
- **Security:** Read-only access enforcement, SQL injection protection, and blocked administrative commands.

## Prerequisites

- **Python:** 3.12+ recommended.
- **PostgreSQL:** A running PostgreSQL database with the `monitoramento` schema populated (TimescaleDB extension recommended for `time_bucket`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd RoutersMCP
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Create a `.env` file in the root directory (you can copy the example below):
    ```ini
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=postgres
    DB_USER=postgres
    DB_PASSWORD=password
    MCP_HOST=0.0.0.0
    MCP_PORT=8000
    ```

2.  Update the values to match your PostgreSQL configuration.

## Usage

Start the MCP server:

```bash
python server.py
```

The server will start listening on the configured host and port (default: `0.0.0.0:8000`) using the SSE (Server-Sent Events) transport.

## Testing

This project uses `pytest` for testing. The tests include security checks (SQL injection, read-only enforcement) and logic verification using mocks.

To run the tests:

```bash
python -m pytest test_server.py
```

## Available Tools

-   `list_available_metrics()`: List all metric tables and their numeric columns.
-   `describe_table_schema(table_name)`: Get the column schema for a specific table.
-   `get_network_inventory()`: List all routers.
-   `list_router_devices(router_mac)`: List devices connected to a router.
-   `get_device_full_info(device_mac)`: Get detailed info for a device (and its parent router).
-   `get_metrics_aggregate(...)`: Aggregated time-series data (requires TimescaleDB).
-   `execute_readonly_query(query)`: Execute a raw SQL query (Strictly Read-Only).

## Available Prompts

-   `audit_network_prompt`: Guides a full network audit.
-   `diagnose_router_prompt(router_mac)`: Diagnosis steps for a specific router.
-   `metric_analysis_prompt(table_name, mac_address)`: Time-series analysis request.
