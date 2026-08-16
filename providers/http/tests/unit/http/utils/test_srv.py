# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

from unittest import mock

import dns.exception
import dns.resolver
import pytest

from airflow.models import Connection
from airflow.providers.common.compat.sdk import AirflowException
from airflow.providers.http.exceptions import SRVResolutionError
from airflow.providers.http.utils.srv import (
    _weighted_order,
    resolve_http_connection_endpoint,
    resolve_srv_records,
)


class MockSRVRecord:
    """Mock DNS SRV record for testing."""

    def __init__(self, priority: int, weight: int, port: int, target: str):
        self.priority = priority
        self.weight = weight
        self.port = port
        self.target = target


class TestResolveSrvRecords:
    """Tests for resolve_srv_records function."""

    @mock.patch("airflow.providers.http.utils.srv.dns.resolver.Resolver", autospec=True)
    def test_basic_srv_resolution(self, mock_resolver_cls):
        mock_resolver = mock_resolver_cls.return_value
        mock_resolver.resolve.return_value = [
            MockSRVRecord(10, 50, 8080, "server1.example.com."),
            MockSRVRecord(20, 100, 9000, "backup.example.com."),
        ]

        targets = resolve_srv_records("_http._tcp.myservice.local")

        assert len(targets) == 2
        mock_resolver.resolve.assert_called_once_with("_http._tcp.myservice.local", "SRV")

    @mock.patch("airflow.providers.http.utils.srv.dns.resolver.Resolver", autospec=True)
    def test_srv_resolution_respects_priority_order(self, mock_resolver_cls):
        mock_resolver = mock_resolver_cls.return_value
        mock_resolver.resolve.return_value = [
            MockSRVRecord(20, 100, 9000, "backup.example.com."),
            MockSRVRecord(10, 50, 8080, "primary.example.com."),
            MockSRVRecord(30, 50, 7000, "tertiary.example.com."),
        ]

        targets = resolve_srv_records("_http._tcp.myservice.local")

        assert targets[0][0] == "primary.example.com"
        assert targets[-1][0] == "tertiary.example.com"

    @mock.patch("airflow.providers.http.utils.srv.dns.resolver.Resolver", autospec=True)
    def test_srv_resolution_strips_trailing_dot(self, mock_resolver_cls):
        mock_resolver = mock_resolver_cls.return_value
        mock_resolver.resolve.return_value = [
            MockSRVRecord(10, 50, 8080, "server.example.com."),
        ]

        targets = resolve_srv_records("_http._tcp.myservice.local")

        assert targets[0] == ("server.example.com", 8080)

    @mock.patch("airflow.providers.http.utils.srv.dns.resolver.Resolver", autospec=True)
    def test_srv_resolution_custom_timeout(self, mock_resolver_cls):
        mock_resolver = mock_resolver_cls.return_value
        mock_resolver.resolve.return_value = [
            MockSRVRecord(10, 50, 8080, "server.example.com."),
        ]

        resolve_srv_records("_http._tcp.myservice.local", timeout=15.0)

        assert mock_resolver.lifetime == 15.0

    @pytest.mark.parametrize(
        "exception_cls",
        [
            dns.resolver.NXDOMAIN,
            dns.resolver.NoAnswer,
            dns.resolver.NoNameservers,
            dns.exception.Timeout,
        ],
    )
    @mock.patch("airflow.providers.http.utils.srv.dns.resolver.Resolver", autospec=True)
    def test_srv_resolution_raises_on_dns_errors(self, mock_resolver_cls, exception_cls):
        mock_resolver = mock_resolver_cls.return_value
        mock_resolver.resolve.side_effect = exception_cls()

        with pytest.raises(SRVResolutionError, match="No SRV records found"):
            resolve_srv_records("_http._tcp.nonexistent.local")


class TestWeightedOrder:
    """Tests for _weighted_order function (RFC 2782 weighted selection)."""

    def test_weighted_order_returns_all_records(self):
        records = [
            (50, "server1.example.com", 8080),
            (30, "server2.example.com", 8081),
            (20, "server3.example.com", 8082),
        ]

        ordered = _weighted_order(records)

        assert len(ordered) == 3
        assert set(r[1] for r in ordered) == {"server1.example.com", "server2.example.com", "server3.example.com"}

    def test_weighted_order_handles_zero_weights(self):
        records = [
            (0, "server1.example.com", 8080),
            (0, "server2.example.com", 8081),
        ]

        ordered = _weighted_order(records)

        assert len(ordered) == 2

    def test_weighted_order_handles_single_record(self):
        records = [(100, "server.example.com", 8080)]

        ordered = _weighted_order(records)

        assert ordered == [(100, "server.example.com", 8080)]

    def test_weighted_order_handles_empty_list(self):
        ordered = _weighted_order([])

        assert ordered == []


class TestResolveHttpConnectionEndpoint:
    """Tests for resolve_http_connection_endpoint function."""

    def test_passthrough_without_srv_enabled(self):
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host="api.example.com",
            port=443,
        )

        host, port, failover = resolve_http_connection_endpoint(conn)

        assert host == "api.example.com"
        assert port == 443
        assert failover == []

    def test_passthrough_uses_default_host(self):
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host=None,
            port=8080,
        )

        host, port, failover = resolve_http_connection_endpoint(conn, default_host="default.example.com")

        assert host == "default.example.com"
        assert port == 8080
        assert failover == []

    @mock.patch("airflow.providers.http.utils.srv.resolve_srv_records")
    def test_srv_resolution_returns_first_target(self, mock_resolve):
        mock_resolve.return_value = [
            ("server1.example.com", 8080),
            ("server2.example.com", 8081),
        ]
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host="_http._tcp.myservice.local",
            extra='{"srv": true}',
        )

        host, port, failover = resolve_http_connection_endpoint(conn)

        assert host == "server1.example.com"
        assert port == 8080
        mock_resolve.assert_called_once_with("_http._tcp.myservice.local", timeout=5.0)

    @mock.patch("airflow.providers.http.utils.srv.resolve_srv_records")
    def test_srv_resolution_with_failover_enabled(self, mock_resolve):
        mock_resolve.return_value = [
            ("server1.example.com", 8080),
            ("server2.example.com", 8081),
            ("server3.example.com", 8082),
        ]
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host="_http._tcp.myservice.local",
            extra='{"srv": true, "srv_failover": true}',
        )

        host, port, failover = resolve_http_connection_endpoint(conn)

        assert host == "server1.example.com"
        assert port == 8080
        assert failover == [("server2.example.com", 8081), ("server3.example.com", 8082)]

    @mock.patch("airflow.providers.http.utils.srv.resolve_srv_records")
    def test_srv_resolution_without_failover_returns_empty_list(self, mock_resolve):
        mock_resolve.return_value = [
            ("server1.example.com", 8080),
            ("server2.example.com", 8081),
        ]
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host="_http._tcp.myservice.local",
            extra='{"srv": true, "srv_failover": false}',
        )

        host, port, failover = resolve_http_connection_endpoint(conn)

        assert failover == []

    @mock.patch("airflow.providers.http.utils.srv.resolve_srv_records")
    def test_srv_resolution_custom_timeout(self, mock_resolve):
        mock_resolve.return_value = [("server.example.com", 8080)]
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host="_http._tcp.myservice.local",
            extra='{"srv": true, "srv_timeout": 15}',
        )

        resolve_http_connection_endpoint(conn)

        mock_resolve.assert_called_once_with("_http._tcp.myservice.local", timeout=15.0)

    def test_srv_with_port_raises_error(self):
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host="_http._tcp.myservice.local",
            port=8080,
            extra='{"srv": true}',
        )

        with pytest.raises(AirflowException, match="should not specify a port"):
            resolve_http_connection_endpoint(conn)

    def test_srv_without_host_raises_error(self):
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host=None,
            extra='{"srv": true}',
        )

        with pytest.raises(AirflowException, match="require the host field"):
            resolve_http_connection_endpoint(conn)

    @mock.patch("airflow.providers.http.utils.srv.resolve_srv_records")
    def test_srv_resolution_error_propagates(self, mock_resolve):
        mock_resolve.side_effect = SRVResolutionError("No SRV records found for '_http._tcp.bad.local'")
        conn = Connection(
            conn_id="test",
            conn_type="http",
            host="_http._tcp.bad.local",
            extra='{"srv": true}',
        )

        with pytest.raises(AirflowException, match="No SRV records found"):
            resolve_http_connection_endpoint(conn)
