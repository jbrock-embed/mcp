# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""awslabs Athena MCP Server implementation."""

from __future__ import annotations

import asyncio
import boto3
import functools
import time
from . import __version__
from .models import (
    ColumnInfo,
    DatabaseInfo,
    DataCatalogSummary,
    DataCatalogType,
    ErrorResponse,
    ListDatabasesResponse,
    ListDataCatalogsResponse,
    ListTablesResponse,
    ListWorkgroupsResponse,
    QueryResults,
    TableInfo,
    TableSummary,
    TableType,
    WorkgroupDetailsResponse,
    WorkgroupState,
    WorkgroupSummary,
)
from .validation import validate_query
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Annotated, Any


DEFAULT_QUERY_TIMEOUT_SECONDS = 60


mcp = FastMCP(
    'awslabs.athena-mcp-server',
    instructions="""AWS Athena MCP Server provides tools to execute read-only SQL queries via AWS Athena.

Supports complex analytical queries, joins, aggregations, and window functions.

## Example Workflows

### Data Discovery
1. List available databases and data catalogs with `list_databases` and `list_data_catalogs`
2. Explore tables within databases using `list_tables`
3. Get detailed table metadata including columns and partitions with `get_table_metadata`
4. Use DESCRIBE or SHOW statements via `execute_query` to explore data structure

### Query Execution
1. Execute queries with `execute_query`
2. For large result sets, use pagination via `get_query_results`
3. Monitor query execution time and data scanned from `execute_query` response
""",
    dependencies=[
        'pydantic',
        'loguru',
        'boto3',
        'sqlglot',
    ],
)


@functools.lru_cache
def _get_athena_client(region: str = ''):
    """Get configured Athena client."""
    try:
        config = Config(user_agent_extra=f'awslabs/mcp/athena-mcp-server/{__version__}')
        if region.strip():
            return boto3.client('athena', region_name=region, config=config)
        else:
            return boto3.client('athena', config=config)
    except Exception as e:
        logger.error(f'Error creating Athena client: {e}')
        raise


def _extract_columns_and_rows(
    response: dict[str, Any],
) -> tuple[list[ColumnInfo], list[dict[str, str]]]:
    """Parse AWS Athena query results response into column info and rows.

    Args:
        response: Raw AWS Athena get_query_results response

    Returns:
        Tuple of (column_info, rows) where rows are dicts with column names as keys
    """
    column_info = []
    # Column metadata is in ResultSet.ResultSetMetadata.ColumnInfo
    result_set = response.get('ResultSet', {})
    metadata = result_set.get('ResultSetMetadata', {})
    if 'ColumnInfo' in metadata:
        for col in metadata['ColumnInfo']:
            # Handle nullable field - AWS returns 'UNKNOWN', 'NULLABLE', 'NOT_NULL' as strings
            nullable_str = col.get('Nullable')
            nullable = None
            if nullable_str == 'NULLABLE':
                nullable = True
            elif nullable_str == 'NOT_NULL':
                nullable = False
            # For 'UNKNOWN' or any other value, leave as None
            column_info.append(
                ColumnInfo(
                    name=col['Name'],
                    type=col['Type'],
                    nullable=nullable,
                    precision=col.get('Precision'),
                    scale=col.get('Scale'),
                )
            )
    rows = []
    if 'Rows' in response['ResultSet']:
        # Get column names from metadata
        column_names = [col.name for col in column_info]
        for row in response['ResultSet']['Rows'][1:]:  # Skip header row
            row_data = [col.get('VarCharValue', '') for col in row['Data']]
            # Convert to dict format with column names as keys
            if column_names:
                row_dict = dict(zip(column_names, row_data))
                rows.append(row_dict)
            else:
                # Fallback: use generic column names if metadata is missing
                row_dict = {f'column_{i}': val for i, val in enumerate(row_data)}
                rows.append(row_dict)
    return column_info, rows


def _handle_athena_error(error: Exception) -> ErrorResponse:
    """Handle and format Athena errors."""
    if isinstance(error, ClientError):
        error_code = error.response['Error']['Code']
        error_message = error.response['Error']['Message']
        return ErrorResponse(
            error_code=error_code,
            error_message=error_message,
            error_type='ClientError',
        )
    else:
        return ErrorResponse(
            error_code='InternalError',
            error_message=str(error),
            error_type=type(error).__name__,
        )


@mcp.tool()
async def execute_query(
    query_string: Annotated[
        str,
        Field(
            min_length=1,
            max_length=262144,
            description='SQL query to execute. Allowed statements: SELECT, VALUES, DESCRIBE, SHOW, EXPLAIN.',
        ),
    ],
    workgroup: Annotated[
        str,
        Field(
            max_length=128,
            description='Athena workgroup to use for query execution. If not specified, uses default workgroup.',
        ),
    ] = '',
    database: Annotated[
        str,
        Field(
            max_length=255,
            description='Default database for the query context. If not specified, queries must include fully qualified table names.',
        ),
    ] = '',
    output_location: Annotated[
        str,
        Field(
            description='S3 location for query results. If not specified, uses workgroup default location.'
        ),
    ] = '',
    timeout_seconds: Annotated[
        int,
        Field(
            ge=0,
            description='Maximum time to wait for query completion in seconds. Use 0 for no timeout.',
        ),
    ] = DEFAULT_QUERY_TIMEOUT_SECONDS,
    region: Annotated[
        str,
        Field(
            description='AWS region to use for the query. If not specified, uses boto3 default region resolution.',
        ),
    ] = '',
) -> QueryResults:
    """Execute a read-only SQL query in Athena and return the results.

    - Only allows SELECT, VALUES, DESCRIBE, SHOW, and EXPLAIN operations.
    - Non-read-only operations such as INSERT, UPDATE, DELETE, CREATE, DROP, and ALTER are not allowed.
    - When appropriate, use LIMIT clauses, WHERE filters, and select specific columns
    - Leverage partitioning when appropriate.

    Args:
        query_string: SQL query to execute (SELECT, VALUES, DESCRIBE, SHOW, EXPLAIN)
        workgroup: Athena workgroup to use
        database: Default database for the query context
        output_location: S3 location for query results
        timeout_seconds: Maximum time to wait for query completion (0 for no timeout)
        region: AWS region to use for the query

    Returns:
        Query results data containing:
        - column_info: List of column metadata with name, type, nullable status, precision, scale
        - rows: List of dictionaries where each dict represents a row with column names as keys
        - total_rows: Number of data rows returned (excluding headers)
        - query_execution_id: AWS Athena execution ID for reference or pagination
        - next_token: Pagination token if more results are available (use with get_query_results)
        - data_scanned_in_bytes: Amount of data scanned by the query (for cost analysis)
        - execution_time_in_millis: Query execution time in milliseconds (performance metrics)
    """
    validate_query(query_string)

    try:
        client = _get_athena_client(region)
        params: dict[str, Any] = {'QueryString': query_string}
        if workgroup.strip():
            params['WorkGroup'] = workgroup
        if output_location.strip():
            params['ResultConfiguration'] = {'OutputLocation': output_location}
        if database.strip():
            params['QueryExecutionContext'] = {'Database': database}
        response = client.start_query_execution(**params)
        query_execution_id = response['QueryExecutionId']

        # Poll for completion
        start_time = time.time()
        while True:
            execution_response = client.get_query_execution(QueryExecutionId=query_execution_id)
            execution = execution_response['QueryExecution']
            state = execution['Status']['State']
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            # Check timeout (skip if timeout_seconds is 0)
            if timeout_seconds > 0 and time.time() - start_time > timeout_seconds:
                try:
                    client.stop_query_execution(QueryExecutionId=query_execution_id)
                except Exception:
                    pass  # Ignore errors when cancelling
                raise TimeoutError(f'Query timed out after {timeout_seconds} seconds')
            await asyncio.sleep(2)

        if state == 'FAILED':
            error_reason = execution['Status'].get('StateChangeReason', 'Query failed')
            raise RuntimeError(f'Query failed: {error_reason}')
        elif state == 'CANCELLED':
            raise RuntimeError('Query was cancelled')

        results_response = client.get_query_results(
            QueryExecutionId=query_execution_id, MaxResults=1000
        )

        column_info, rows = _extract_columns_and_rows(results_response)
        statistics = execution.get('Statistics', {})
        return QueryResults(
            column_info=column_info,
            rows=rows,
            total_rows=len(rows),
            next_token=results_response.get('NextToken'),
            query_execution_id=query_execution_id,
            data_scanned_in_bytes=statistics.get('DataScannedInBytes'),
            execution_time_in_millis=statistics.get('EngineExecutionTimeInMillis'),
        )
    except Exception as e:
        logger.error(f'Error executing query: {e}')
        error_response = _handle_athena_error(e)
        raise RuntimeError(f'{error_response.error_code}: {error_response.error_message}')


@mcp.tool()
async def get_query_results(
    query_execution_id: Annotated[
        str,
        Field(
            description='Query execution ID from execute_query. Used to retrieve additional results from a completed query (mainly for pagination).'
        ),
    ],
    next_token: Annotated[
        str,
        Field(
            description='Token for pagination. Use this to get additional result pages if more results are available.'
        ),
    ] = '',
    region: Annotated[
        str,
        Field(
            description='AWS region to use for the query. If not specified, uses boto3 default region resolution.',
        ),
    ] = '',
) -> QueryResults:
    """Get results from a completed query execution.

    This is mainly used for pagination of large result sets since execute_query returns results
    directly.

    Args:
        query_execution_id: Query execution ID from execute_query
        next_token: Token for pagination
        region: AWS region to use for the query

    Returns:
        Paginated query results containing:
        - column_info: List of column metadata with name, type, nullable status, precision, scale
        - rows: List of dictionaries where each dict represents a row with column names as keys
        - total_rows: Number of data rows returned in this page
        - query_execution_id: Same execution ID passed in (for reference)
        - next_token: Token for next page if more results available, None if this is the last page
        - data_scanned_in_bytes: Amount of data scanned by the original query
        - execution_time_in_millis: Original query execution time in milliseconds
    """
    try:
        client = _get_athena_client(region)
        params = {
            'QueryExecutionId': query_execution_id,
            'MaxResults': 1000,
        }
        if next_token.strip():
            params['NextToken'] = next_token
        response = client.get_query_results(**params)
        column_info, rows = _extract_columns_and_rows(response)
        return QueryResults(
            column_info=column_info,
            rows=rows,
            total_rows=len(rows),
            next_token=response.get('NextToken'),
            query_execution_id=query_execution_id,
            data_scanned_in_bytes=None,  # Not available in get_query_results
            execution_time_in_millis=None,  # Not available in get_query_results
        )
    except Exception as e:
        logger.error(f'Error getting query results: {e}')
        error_response = _handle_athena_error(e)
        raise RuntimeError(f'{error_response.error_code}: {error_response.error_message}')


@mcp.tool()
async def list_databases(
    catalog_name: Annotated[
        str,
        Field(
            description='Data catalog name to list databases from. Use "AwsDataCatalog" for AWS Glue Data Catalog.'
        ),
    ] = 'AwsDataCatalog',
    next_token: Annotated[
        str, Field(description='Token for pagination to get additional database pages.')
    ] = '',
    region: Annotated[
        str,
        Field(
            description='AWS region to use for the query. If not specified, uses boto3 default region resolution.',
        ),
    ] = '',
) -> ListDatabasesResponse:
    """List databases in the specified data catalog.

    Args:
        catalog_name: Data catalog name
        next_token: Token for pagination
        region: AWS region to use for the query

    Returns:
        Database listing containing:
        - databases: List of database dictionaries with 'name', 'description', and 'parameters'
        - next_token: Pagination token for retrieving additional databases, None if no more pages
    """
    try:
        client = _get_athena_client(region)
        params = {
            'CatalogName': catalog_name,
            'MaxResults': 50,
        }
        if next_token.strip():
            params['NextToken'] = next_token
        response = client.list_databases(**params)
        databases = []
        for db in response.get('DatabaseList', []):
            databases.append(
                DatabaseInfo(
                    name=db['Name'],
                    description=db.get('Description'),
                    parameters=db.get('Parameters'),
                ).model_dump()
            )
        return ListDatabasesResponse(
            databases=databases,
            next_token=response.get('NextToken'),
        )
    except Exception as e:
        logger.error(f'Error listing databases: {e}')
        error_response = _handle_athena_error(e)
        raise RuntimeError(f'{error_response.error_code}: {error_response.error_message}')


@mcp.tool()
async def list_tables(
    database_name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=255,
            description='Database name to list tables from. Must be a valid database name from the data catalog.',
        ),
    ],
    catalog_name: Annotated[
        str, Field(description='Data catalog name. Use AwsDataCatalog for AWS Glue Data Catalog.')
    ] = 'AwsDataCatalog',
    expression: Annotated[
        str,
        Field(
            description='Regular expression to filter table names. Only tables matching this pattern will be returned. If not specified, returns all tables.'
        ),
    ] = '',
    next_token: Annotated[
        str, Field(description='Token for pagination to get additional table pages.')
    ] = '',
    region: Annotated[
        str,
        Field(
            description='AWS region to use for the query. If not specified, uses boto3 default region resolution.',
        ),
    ] = '',
) -> ListTablesResponse:
    """List tables in the specified database.

    Args:
        database_name: Database name
        catalog_name: Data catalog name
        expression: Regular expression to filter table names
        next_token: Token for pagination
        region: AWS region to use for the query

    Returns:
        Table listing containing:
        - tables: List of table summaries with name, table_type, create_time, last_access_time, columns_count, partition_keys_count
        - next_token: Pagination token for retrieving additional tables, None if no more pages
    """
    try:
        client = _get_athena_client(region)
        params = {
            'CatalogName': catalog_name,
            'DatabaseName': database_name,
            'MaxResults': 50,
        }
        if expression.strip():
            params['Expression'] = expression
        if next_token.strip():
            params['NextToken'] = next_token
        response = client.list_table_metadata(**params)
        tables = []
        for table in response.get('TableMetadataList', []):
            tables.append(
                TableSummary(
                    name=table['Name'],
                    table_type=TableType(table['TableType']) if table.get('TableType') else None,
                    create_time=table.get('CreateTime'),
                    last_access_time=table.get('LastAccessTime'),
                    columns_count=len(table.get('Columns', [])),
                    partition_keys_count=len(table.get('PartitionKeys', [])),
                )
            )
        return ListTablesResponse(
            tables=tables,
            next_token=response.get('NextToken'),
        )
    except Exception as e:
        logger.error(f'Error listing tables: {e}')
        error_response = _handle_athena_error(e)
        raise RuntimeError(f'{error_response.error_code}: {error_response.error_message}')


@mcp.tool()
async def get_table_metadata(
    database_name: Annotated[
        str,
        Field(description='Database name containing the table. Must exist in the data catalog.'),
    ],
    table_name: Annotated[
        str,
        Field(description='Table name to get metadata for. Must exist in the specified database.'),
    ],
    catalog_name: Annotated[
        str,
        Field(description='Data catalog name. Use "AwsDataCatalog" for AWS Glue Data Catalog.'),
    ] = 'AwsDataCatalog',
    region: Annotated[
        str,
        Field(
            description='AWS region to use for the query. If not specified, uses boto3 default region resolution.',
        ),
    ] = '',
) -> TableInfo:
    """Get detailed metadata for a specific table.

    Args:
        database_name: Database name
        table_name: Table name
        catalog_name: Data catalog name
        region: AWS region to use for the query

    Returns:
        Detailed table metadata containing:
        - name: Table name
        - table_type: Type (EXTERNAL_TABLE, MANAGED_TABLE, VIRTUAL_VIEW)
        - create_time, last_access_time: Timestamps for table lifecycle
        - columns: List of column definitions with detailed schema information
        - partition_keys: List of partition column definitions
        - location: S3 path where table data is stored
        - input_format, output_format: Hadoop input/output format classes
        - serde_info: Serialization/deserialization configuration
        - parameters: Additional table properties and metadata
    """
    try:
        client = _get_athena_client(region)
        response = client.get_table_metadata(
            CatalogName=catalog_name,
            DatabaseName=database_name,
            TableName=table_name,
        )
        table = response['TableMetadata']
        columns = []
        for col in table.get('Columns', []):
            columns.append(
                ColumnInfo(
                    name=col['Name'],
                    type=col['Type'],
                    nullable=col.get('Nullable'),
                )
            )
        partition_keys = []
        for pk in table.get('PartitionKeys', []):
            partition_keys.append(
                ColumnInfo(
                    name=pk['Name'],
                    type=pk['Type'],
                )
            )
        return TableInfo(
            name=table['Name'],
            table_type=table.get('TableType'),
            create_time=table.get('CreateTime'),
            last_access_time=table.get('LastAccessTime'),
            columns=columns,
            partition_keys=partition_keys,
            location=table.get('StorageDescriptor', {}).get('Location'),
            input_format=table.get('StorageDescriptor', {}).get('InputFormat'),
            output_format=table.get('StorageDescriptor', {}).get('OutputFormat'),
            serde_info=table.get('StorageDescriptor', {}).get('SerdeInfo'),
            parameters=table.get('Parameters'),
        )
    except Exception as e:
        logger.error(f'Error getting table metadata: {e}')
        error_response = _handle_athena_error(e)
        raise RuntimeError(f'{error_response.error_code}: {error_response.error_message}')


@mcp.tool()
async def list_work_groups(
    next_token: Annotated[
        str, Field(description='Token for pagination to get additional workgroup pages.')
    ] = '',
    region: Annotated[
        str,
        Field(
            description='AWS region to use for the query. If not specified, uses boto3 default region resolution.',
        ),
    ] = '',
) -> ListWorkgroupsResponse:
    """List available Athena workgroups.

    Args:
        next_token: Token for pagination
        region: AWS region to use for the query

    Returns:
        Workgroup listing containing:
        - workgroups: List of workgroup summaries with name, state, description, creation_time
        - next_token: Pagination token for retrieving additional workgroups, None if no more pages
    """
    try:
        client = _get_athena_client(region)
        params: dict[str, Any] = {'MaxResults': 50}
        if next_token.strip():
            params['NextToken'] = next_token
        response = client.list_work_groups(**params)
        workgroups = []
        for wg in response.get('WorkGroups', []):
            workgroups.append(
                WorkgroupSummary(
                    name=wg['Name'],
                    state=WorkgroupState(wg['State']),
                    description=wg.get('Description'),
                    creation_time=wg.get('CreationTime'),
                )
            )
        return ListWorkgroupsResponse(
            workgroups=workgroups,
            next_token=response.get('NextToken'),
        )
    except Exception as e:
        logger.error(f'Error listing workgroups: {e}')
        error_response = _handle_athena_error(e)
        raise RuntimeError(f'{error_response.error_code}: {error_response.error_message}')


@mcp.tool()
async def get_work_group(
    workgroup_name: Annotated[
        str,
        Field(
            description='Name of the workgroup to get details for. Must be a valid workgroup name that you have access to.'
        ),
    ],
    region: Annotated[
        str,
        Field(
            description='AWS region to use for the query. If not specified, uses boto3 default region resolution.',
        ),
    ] = '',
) -> WorkgroupDetailsResponse:
    """Get detailed workgroup configuration and settings.

    Args:
        workgroup_name: Name of the workgroup
        region: AWS region to use for the query

    Returns:
        Detailed workgroup information containing:
        - name: Workgroup name
        - state: Current state (ENABLED/DISABLED)
        - description: Optional workgroup description
        - creation_time: When the workgroup was created
        - configuration: Complete workgroup settings including result location, encryption, cost controls
    """
    try:
        client = _get_athena_client(region)

        response = client.get_work_group(WorkGroup=workgroup_name)
        workgroup = response['WorkGroup']
        config = workgroup.get('Configuration', {})
        result_config = config.get('ResultConfiguration', {})
        return WorkgroupDetailsResponse(
            name=workgroup['Name'],
            state=WorkgroupState(workgroup['State']),
            description=workgroup.get('Description'),
            creation_time=workgroup.get('CreationTime'),
            configuration={
                'result_configuration': {
                    'output_location': result_config.get('OutputLocation'),
                    'encryption_configuration': result_config.get('EncryptionConfiguration'),
                },
                'enforce_work_group_configuration': config.get('EnforceWorkGroupConfiguration'),
                'publish_cloud_watch_metrics': config.get('PublishCloudWatchMetrics'),
                'bytes_scanned_cutoff_per_query': config.get('BytesScannedCutoffPerQuery'),
                'requester_pays_enabled': config.get('RequesterPaysEnabled'),
                'engine_version': config.get('EngineVersion'),
            },
        )
    except Exception as e:
        logger.error(f'Error getting workgroup details: {e}')
        error_response = _handle_athena_error(e)
        raise RuntimeError(f'{error_response.error_code}: {error_response.error_message}')


@mcp.tool()
async def list_data_catalogs(
    next_token: Annotated[
        str,
        Field(description='Token for pagination to get additional data catalog pages.'),
    ] = '',
    region: Annotated[
        str,
        Field(
            description='AWS region to use for the query. If not specified, uses boto3 default region resolution.',
        ),
    ] = '',
) -> ListDataCatalogsResponse:
    """List available data catalogs.

    Args:
        next_token: Token for pagination
        region: AWS region to use for the query

    Returns:
        Data catalog listing containing:
        - data_catalogs: List of catalog summaries with catalog_name and type (GLUE, HIVE, LAMBDA)
        - next_token: Pagination token for retrieving additional catalogs, None if no more pages
    """
    try:
        client = _get_athena_client(region)
        params: dict[str, Any] = {'MaxResults': 50}
        if next_token.strip():
            params['NextToken'] = next_token
        response = client.list_data_catalogs(**params)
        catalogs = []
        for catalog in response.get('DataCatalogsSummary', []):
            catalogs.append(
                DataCatalogSummary(
                    catalog_name=catalog['CatalogName'],
                    type=DataCatalogType(catalog['Type']),
                )
            )
        return ListDataCatalogsResponse(
            data_catalogs=catalogs,
            next_token=response.get('NextToken'),
        )
    except Exception as e:
        logger.error(f'Error listing data catalogs: {e}')
        error_response = _handle_athena_error(e)
        raise RuntimeError(f'{error_response.error_code}: {error_response.error_message}')


def main():
    """Run the MCP server with CLI argument support."""
    mcp.run()


if __name__ == '__main__':
    main()
