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

"""Test the Athena MCP server implementation."""

import pytest
from awslabs.athena_mcp_server.models import (
    ErrorResponse,
    QueryResults,
    TableInfo,
)
from awslabs.athena_mcp_server.server import (
    _get_athena_client,
    _handle_athena_error,
    execute_query,
    get_query_results,
    get_table_metadata,
    get_work_group,
    list_data_catalogs,
    list_databases,
    list_tables,
    list_work_groups,
)
from botocore.exceptions import ClientError
from datetime import datetime


@pytest.fixture
def mock_athena_client(mocker):
    """Create a mock Athena client."""
    mock_client = mocker.MagicMock()
    mocker.patch('awslabs.athena_mcp_server.server.boto3.client', return_value=mock_client)

    yield mock_client

    # Clear the LRU cache after mocked tests finish to ensure live integration
    # tests and subsequent tests get fresh, real boto3 clients instead of
    # cached mock clients that would cause validation errors
    _get_athena_client.cache_clear()


@pytest.fixture
def sample_query_execution():
    """Sample query execution response."""
    return {
        'QueryExecutionId': 'test-execution-id-123',
        'Query': 'SELECT * FROM test_table LIMIT 10',
        'Status': {
            'State': 'SUCCEEDED',
            'StateChangeReason': 'Query completed successfully',
            'SubmissionDateTime': datetime(2024, 1, 1, 12, 0, 0),
            'CompletionDateTime': datetime(2024, 1, 1, 12, 0, 30),
        },
        'Statistics': {
            'DataScannedInBytes': 1024,
            'EngineExecutionTimeInMillis': 5000,
        },
        'WorkGroup': 'primary',
        'ResultConfiguration': {
            'OutputLocation': 's3://test-bucket/results/',
        },
    }


@pytest.fixture
def sample_query_results():
    """Sample query results response."""
    return {
        'ResultSet': {
            'ResultSetMetadata': {
                'ColumnInfo': [
                    {'Name': 'id', 'Type': 'bigint', 'Nullable': 'NULLABLE'},
                    {'Name': 'name', 'Type': 'varchar', 'Nullable': 'NULLABLE'},
                    {
                        'Name': 'score',
                        'Type': 'double',
                        'Precision': 10,
                        'Scale': 2,
                        'Nullable': 'NULLABLE',
                    },
                ]
            },
            'Rows': [
                {
                    'Data': [
                        {'VarCharValue': 'id'},
                        {'VarCharValue': 'name'},
                        {'VarCharValue': 'score'},
                    ]
                },
                {
                    'Data': [
                        {'VarCharValue': '1'},
                        {'VarCharValue': 'Alice'},
                        {'VarCharValue': '95.5'},
                    ]
                },
                {
                    'Data': [
                        {'VarCharValue': '2'},
                        {'VarCharValue': 'Bob'},
                        {'VarCharValue': '87.3'},
                    ]
                },
            ],
        },
        'NextToken': 'next-page-token',
    }


class TestExecuteQuery:
    """Test query execution functionality."""

    @pytest.mark.asyncio
    async def test_execute_query_success(
        self, mock_athena_client, sample_query_execution, sample_query_results
    ):
        """Test successful query execution."""
        mock_athena_client.start_query_execution.return_value = {
            'QueryExecutionId': 'test-execution-id-123'
        }
        mock_athena_client.get_query_execution.return_value = {
            'QueryExecution': sample_query_execution
        }
        mock_athena_client.get_query_results.return_value = sample_query_results

        result = await execute_query('SELECT * FROM test_table LIMIT 10')

        assert isinstance(result, QueryResults)
        assert len(result.column_info) == 3
        assert result.column_info[0].name == 'id'
        assert result.column_info[0].type == 'bigint'
        assert len(result.rows) == 2  # Excluding header row
        assert result.rows[0] == {'id': '1', 'name': 'Alice', 'score': '95.5'}
        assert result.total_rows == 2
        assert result.next_token == 'next-page-token'
        # Test execution metadata
        assert result.query_execution_id == 'test-execution-id-123'
        assert result.data_scanned_in_bytes == 1024
        assert result.execution_time_in_millis == 5000

        mock_athena_client.start_query_execution.assert_called_once()
        mock_athena_client.get_query_execution.assert_called_once()
        mock_athena_client.get_query_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_with_options(
        self, mock_athena_client, sample_query_execution, sample_query_results
    ):
        """Test query execution with all options."""
        mock_athena_client.start_query_execution.return_value = {
            'QueryExecutionId': 'test-execution-id-123'
        }
        mock_athena_client.get_query_execution.return_value = {
            'QueryExecution': sample_query_execution
        }
        mock_athena_client.get_query_results.return_value = sample_query_results

        result = await execute_query(
            query_string='SELECT COUNT(*) FROM test_table',
            workgroup='test-workgroup',
            database='test_db',
            output_location='s3://my-bucket/results/',
        )

        assert isinstance(result, QueryResults)
        mock_athena_client.start_query_execution.assert_called_once_with(
            QueryString='SELECT COUNT(*) FROM test_table',
            WorkGroup='test-workgroup',
            QueryExecutionContext={'Database': 'test_db'},
            ResultConfiguration={'OutputLocation': 's3://my-bucket/results/'},
        )

    @pytest.mark.asyncio
    async def test_execute_query_client_error(self, mock_athena_client):
        """Test query execution with client error."""
        mock_athena_client.start_query_execution.side_effect = ClientError(
            {'Error': {'Code': 'InvalidRequestException', 'Message': 'Invalid SQL syntax'}},
            'StartQueryExecution',
        )

        with pytest.raises(RuntimeError, match='InvalidRequestException.*Invalid SQL syntax'):
            await execute_query('SELECT * FROM invalid_table')

    @pytest.mark.asyncio
    async def test_execute_query_failed_query(self, mock_athena_client):
        """Test query execution with failed query."""
        failed_execution = {
            'QueryExecutionId': 'test-execution-id-123',
            'Query': 'SELECT * FROM nonexistent_table',
            'Status': {
                'State': 'FAILED',
                'StateChangeReason': 'Table not found',
            },
        }

        mock_athena_client.start_query_execution.return_value = {
            'QueryExecutionId': 'test-execution-id-123'
        }
        mock_athena_client.get_query_execution.return_value = {'QueryExecution': failed_execution}

        with pytest.raises(RuntimeError, match='Query failed: Table not found'):
            await execute_query('SELECT * FROM nonexistent_table')

    @pytest.mark.asyncio
    async def test_execute_query_timeout(self, mock_athena_client):
        """Test query execution timeout."""
        running_execution = {
            'QueryExecutionId': 'test-execution-id-123',
            'Query': 'SELECT * FROM large_table',
            'Status': {
                'State': 'RUNNING',
            },
        }

        mock_athena_client.start_query_execution.return_value = {
            'QueryExecutionId': 'test-execution-id-123'
        }
        mock_athena_client.get_query_execution.return_value = {'QueryExecution': running_execution}

        with pytest.raises(RuntimeError, match='Query timed out after 1 seconds'):
            await execute_query('SELECT * FROM large_table', timeout_seconds=1)
        mock_athena_client.stop_query_execution.assert_called_once_with(
            QueryExecutionId='test-execution-id-123'
        )


class TestGetQueryResults:
    """Test query results retrieval."""

    @pytest.mark.asyncio
    async def test_get_query_results_success(self, mock_athena_client, sample_query_results):
        """Test successful query results retrieval."""
        mock_athena_client.get_query_results.return_value = sample_query_results

        result = await get_query_results('test-execution-id-123')

        assert isinstance(result, QueryResults)
        assert len(result.column_info) == 3
        assert result.column_info[0].name == 'id'
        assert result.column_info[0].type == 'bigint'
        assert result.column_info[2].precision == 10
        assert result.column_info[2].scale == 2
        assert len(result.rows) == 2  # Excluding header row
        assert result.rows[0] == {'id': '1', 'name': 'Alice', 'score': '95.5'}
        assert result.total_rows == 2
        assert result.next_token == 'next-page-token'
        # Test execution metadata (None for get_query_results)
        assert result.query_execution_id == 'test-execution-id-123'
        assert result.data_scanned_in_bytes is None
        assert result.execution_time_in_millis is None

    @pytest.mark.asyncio
    async def test_get_query_results_with_pagination(
        self, mock_athena_client, sample_query_results
    ):
        """Test query results with pagination parameters."""
        mock_athena_client.get_query_results.return_value = sample_query_results

        result = await get_query_results(
            query_execution_id='test-execution-id-123',
            next_token='previous-token',
            max_results=500,
        )

        mock_athena_client.get_query_results.assert_called_once_with(
            QueryExecutionId='test-execution-id-123',
            NextToken='previous-token',
            MaxResults=500,
        )
        assert isinstance(result, QueryResults)


class TestListDatabases:
    """Test database listing functionality."""

    @pytest.mark.asyncio
    async def test_list_databases_success(self, mock_athena_client):
        """Test successful database listing."""
        mock_athena_client.list_databases.return_value = {
            'DatabaseList': [
                {'Name': 'default', 'Description': 'Default database'},
                {
                    'Name': 'analytics',
                    'Description': 'Analytics database',
                    'Parameters': {'owner': 'team'},
                },
            ],
            'NextToken': 'next-db-token',
        }

        result = await list_databases()

        assert hasattr(result, 'databases')
        assert hasattr(result, 'next_token')
        assert len(result.databases) == 2
        assert result.databases[0]['name'] == 'default'
        assert result.databases[1]['parameters'] == {'owner': 'team'}
        assert result.next_token == 'next-db-token'


class TestListTables:
    """Test table listing functionality."""

    @pytest.mark.asyncio
    async def test_list_tables_success(self, mock_athena_client):
        """Test successful table listing."""
        mock_athena_client.list_table_metadata.return_value = {
            'TableMetadataList': [
                {
                    'Name': 'users',
                    'TableType': 'EXTERNAL_TABLE',
                    'CreateTime': datetime(2024, 1, 1),
                    'Columns': [{'Name': 'id', 'Type': 'bigint'}],
                    'PartitionKeys': [],
                },
                {
                    'Name': 'orders',
                    'TableType': 'EXTERNAL_TABLE',
                    'CreateTime': datetime(2024, 1, 2),
                    'Columns': [{'Name': 'order_id', 'Type': 'varchar'}],
                    'PartitionKeys': [{'Name': 'date', 'Type': 'string'}],
                },
            ],
        }

        result = await list_tables('test_database')

        assert hasattr(result, 'tables')
        assert len(result.tables) == 2
        assert result.tables[0].name == 'users'
        assert result.tables[0].columns_count == 1
        assert result.tables[0].partition_keys_count == 0
        assert result.tables[1].partition_keys_count == 1


class TestGetTableMetadata:
    """Test table metadata retrieval."""

    @pytest.mark.asyncio
    async def test_get_table_metadata_success(self, mock_athena_client):
        """Test successful table metadata retrieval."""
        mock_athena_client.get_table_metadata.return_value = {
            'TableMetadata': {
                'Name': 'users',
                'TableType': 'EXTERNAL_TABLE',
                'CreateTime': datetime(2024, 1, 1),
                'Columns': [
                    {'Name': 'id', 'Type': 'bigint', 'Nullable': False},
                    {'Name': 'name', 'Type': 'varchar', 'Nullable': True},
                ],
                'PartitionKeys': [{'Name': 'year', 'Type': 'string'}],
                'StorageDescriptor': {
                    'Location': 's3://data-bucket/users/',
                    'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
                    'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                    'SerdeInfo': {
                        'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
                    },
                },
                'Parameters': {'classification': 'csv'},
            }
        }

        result = await get_table_metadata('test_db', 'users')

        assert isinstance(result, TableInfo)
        assert result.name == 'users'
        assert result.table_type == 'EXTERNAL_TABLE'
        assert result.columns is not None
        assert len(result.columns) == 2
        assert result.partition_keys is not None
        assert len(result.partition_keys) == 1
        assert result.location == 's3://data-bucket/users/'
        assert result.parameters == {'classification': 'csv'}


class TestErrorHandling:
    """Test error handling functionality."""

    def test_handle_athena_error_client_error(self):
        """Test handling of AWS client errors."""
        client_error = ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'GetQueryExecution',
        )

        error_response = _handle_athena_error(client_error)

        assert isinstance(error_response, ErrorResponse)
        assert error_response.error_code == 'ThrottlingException'
        assert error_response.error_message == 'Rate exceeded'
        assert error_response.error_type == 'ClientError'

    def test_handle_athena_error_generic_error(self):
        """Test handling of generic errors."""
        generic_error = ValueError('Something went wrong')

        error_response = _handle_athena_error(generic_error)

        assert isinstance(error_response, ErrorResponse)
        assert error_response.error_code == 'InternalError'
        assert error_response.error_message == 'Something went wrong'
        assert error_response.error_type == 'ValueError'


class TestWorkgroupOperations:
    """Test workgroup management functionality."""

    @pytest.mark.asyncio
    async def test_list_work_groups_success(self, mock_athena_client):
        """Test successful workgroup listing."""
        mock_athena_client.list_work_groups.return_value = {
            'WorkGroups': [
                {
                    'Name': 'primary',
                    'State': 'ENABLED',
                    'Description': 'Default workgroup',
                    'CreationTime': datetime(2024, 1, 1),
                },
                {
                    'Name': 'analytics-team',
                    'State': 'ENABLED',
                    'Description': 'Analytics team workgroup',
                    'CreationTime': datetime(2024, 1, 2),
                },
            ],
            'NextToken': 'next-workgroup-token',
        }

        result = await list_work_groups()

        assert hasattr(result, 'workgroups')
        assert hasattr(result, 'next_token')
        assert len(result.workgroups) == 2
        assert result.workgroups[0].name == 'primary'
        assert result.workgroups[0].state == 'ENABLED'
        assert result.workgroups[1].name == 'analytics-team'
        assert result.next_token == 'next-workgroup-token'

    @pytest.mark.asyncio
    async def test_get_work_group_success(self, mock_athena_client):
        """Test successful workgroup details retrieval."""
        mock_athena_client.get_work_group.return_value = {
            'WorkGroup': {
                'Name': 'analytics-team',
                'State': 'ENABLED',
                'Description': 'Analytics team workgroup',
                'CreationTime': datetime(2024, 1, 1),
                'Configuration': {
                    'ResultConfiguration': {
                        'OutputLocation': 's3://analytics-results/',
                        'EncryptionConfiguration': {
                            'EncryptionOption': 'SSE_S3',
                        },
                    },
                    'EnforceWorkGroupConfiguration': True,
                    'PublishCloudWatchMetrics': True,
                    'BytesScannedCutoffPerQuery': 1000000000,
                    'RequesterPaysEnabled': False,
                    'EngineVersion': {
                        'SelectedEngineVersion': 'Athena engine version 3',
                    },
                },
            }
        }

        result = await get_work_group('analytics-team')

        assert result.name == 'analytics-team'
        assert result.state == 'ENABLED'
        assert result.description == 'Analytics team workgroup'
        assert (
            result.configuration['result_configuration']['output_location']
            == 's3://analytics-results/'
        )
        assert result.configuration['enforce_work_group_configuration'] is True
        assert result.configuration['bytes_scanned_cutoff_per_query'] == 1000000000


class TestDataCatalogOperations:
    """Test data catalog functionality."""

    @pytest.mark.asyncio
    async def test_list_data_catalogs_success(self, mock_athena_client):
        """Test successful data catalog listing."""
        mock_athena_client.list_data_catalogs.return_value = {
            'DataCatalogsSummary': [
                {
                    'CatalogName': 'AwsDataCatalog',
                    'Type': 'GLUE',
                },
                {
                    'CatalogName': 'custom-catalog',
                    'Type': 'HIVE',
                },
                {
                    'CatalogName': 'external-catalog',
                    'Type': 'LAMBDA',
                },
            ],
            'NextToken': 'next-catalog-token',
        }

        result = await list_data_catalogs()

        assert hasattr(result, 'data_catalogs')
        assert hasattr(result, 'next_token')
        assert len(result.data_catalogs) == 3
        assert result.data_catalogs[0].catalog_name == 'AwsDataCatalog'
        assert result.data_catalogs[0].type == 'GLUE'
        assert result.data_catalogs[1].catalog_name == 'custom-catalog'
        assert result.data_catalogs[1].type == 'HIVE'
        assert result.next_token == 'next-catalog-token'


@pytest.mark.live
class TestLiveIntegration:
    """Integration tests that require live AWS credentials and resources."""

    @pytest.mark.asyncio
    async def test_list_databases_live(self):
        """Test live database listing (requires AWS credentials)."""
        # This test will be skipped unless run with -m live
        result = await list_databases()
        assert hasattr(result, 'databases')
        # Should at least have the default database
        database_names = [db['name'] for db in result.databases]
        assert 'default' in database_names


class TestQueryValidationIntegration:
    """Test SQL query validation integration with server functions."""

    @pytest.mark.asyncio
    async def test_execute_query_validation_integration(self, mock_athena_client):
        """Test that execute_query properly validates queries."""
        # Should block mutation attempts before hitting AWS API
        with pytest.raises(ValueError, match='not permitted'):
            await execute_query('DROP TABLE users')

        # AWS client should not be called for blocked queries
        mock_athena_client.start_query_execution.assert_not_called()
