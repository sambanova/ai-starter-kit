# Connection Pooling Test

This test measures the latency difference between requests that establish the first HTTPS connection and subsequent requests that reuse the same HTTPS connection (HTTP connection pooling).

## Overview

Connection pooling allows multiple HTTP requests to reuse the same underlying TCP/TLS connection, reducing the overhead of establishing new connections for each request. This test demonstrates the performance benefits of connection pooling by comparing the latency of the initial HTTPS request (which establishes a new connection) with the latency of subsequent requests (which reuse the existing connection).

## How to Run

1. Visit the [Evaluations repository](https://github.sambanovasystems.com/perf/evaluations/tree/main).
2. Locate the **Connection Pooling Test** in the repository.
3. Follow the instructions in the repository to run the test and observe the latency differences.

## What to Expect

- **First Request:** Higher latency due to the overhead of establishing a new HTTPS connection.
- **Subsequent Requests:** Lower latency as the connection is reused, demonstrating the effectiveness of HTTP connection pooling.