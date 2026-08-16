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

import random
from collections import defaultdict
from typing import TYPE_CHECKING

import dns.exception
import dns.resolver

from airflow.providers.common.compat.sdk import AirflowException
from airflow.providers.http.exceptions import SRVResolutionError

if TYPE_CHECKING:
    from airflow.models import Connection


def resolve_srv_records(
    srv_name: str,
    *,
    timeout: float = 5.0,
) -> list[tuple[str, int]]:
    """
    Resolve SRV records per RFC 2782.

    Returns an ordered list of (hostname, port) tuples. Records with lower priority
    values come first; within the same priority, records are ordered using the
    RFC 2782 weighted selection algorithm.
    """
    resolver = dns.resolver.Resolver()
    resolver.lifetime = timeout

    try:
        answers = resolver.resolve(srv_name, "SRV")
    except (
        dns.resolver.NXDOMAIN,
        dns.resolver.NoAnswer,
        dns.resolver.NoNameservers,
        dns.exception.Timeout,
    ) as e:
        raise SRVResolutionError(f"No SRV records found for {srv_name!r}") from e

    by_priority: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for rdata in answers:
        by_priority[rdata.priority].append(
            (rdata.weight, str(rdata.target).rstrip("."), rdata.port)
        )

    result: list[tuple[str, int]] = []
    for priority in sorted(by_priority):
        for _, host, port in _weighted_order(by_priority[priority]):
            result.append((host, port))

    if not result:
        raise SRVResolutionError(f"No usable SRV records for {srv_name!r}")

    return result


def _weighted_order(records: list[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    """Return records in RFC 2782 weighted selection order."""
    pool = list(records)
    ordered: list[tuple[int, str, int]] = []

    while pool:
        total_weight = sum(max(weight, 0) for weight, _, _ in pool) or len(pool)
        pick = random.uniform(0, total_weight)
        cumulative = 0
        for index, (weight, host, port) in enumerate(pool):
            cumulative += max(weight, 0) or 1
            if cumulative >= pick:
                ordered.append(pool.pop(index))
                break

    return ordered


def resolve_http_connection_endpoint(
    connection: Connection,
    *,
    default_host: str = "",
) -> tuple[str, int | None, list[tuple[str, int]]]:
    """
    Resolve the host and port for an HTTP connection.

    When ``srv`` is enabled in the connection extra, ``connection.host`` is treated as
    an SRV record name and resolved to a concrete hostname and port.
    """
    extra = connection.extra_dejson
    host = connection.host or default_host
    port = connection.port

    if not extra.get("srv"):
        return host, port, []

    if port:
        raise AirflowException(
            "SRV HTTP connections should not specify a port; it is resolved from the SRV record"
        )
    if not host:
        raise AirflowException(
            "SRV HTTP connections require the host field to contain the SRV record name"
        )

    timeout = float(extra.get("srv_timeout", 5))
    try:
        targets = resolve_srv_records(host, timeout=timeout)
    except SRVResolutionError as e:
        raise AirflowException(str(e)) from e

    resolved_host, resolved_port = targets[0]
    failover_targets = targets[1:] if extra.get("srv_failover") else []
    return resolved_host, resolved_port, failover_targets
