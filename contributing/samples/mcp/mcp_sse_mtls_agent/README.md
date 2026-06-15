# MCP SSE Agent with mTLS

This sample demonstrates how to configure an ADK agent to connect to an MCP server using **mutual TLS (mTLS)** over SSE (HTTPS).

## Prerequisites

To test mTLS locally, you need to generate local certificates (CA, Server, and Client) and configure your environment to trust them.

### 1. Generate Certificates

Run the helper script in this directory to generate a local CA and sign the server and client certificates:

```bash
./generate_mtls_certs.sh
```

This will generate:

- `ca.crt`, `ca.key` (Local CA)
- `server.crt`, `server.key` (Server certificate/key)
- `client.crt`, `client.key` (Client certificate/key)
- `certificate_config.json` (Workload certificate configuration for `google-auth`)

______________________________________________________________________

## Running the Sample

### Step 1: Start the MCP Server

Start the server in this directory. We configure it to trust our local CA so it can verify the client certificate:

```bash
# Point to the certificate config
export GOOGLE_API_CERTIFICATE_CONFIG=$(pwd)/certificate_config.json

# Tell the server to trust our test CA for client verification
export SSL_CA_CERTS=$(pwd)/ca.crt

# Run the server
python filesystem_server.py
```

*(The server will run on `https://localhost:3000`)*

### Step 2: Run the ADK Agent (Client)

In a second terminal, navigate to the open-source workspace root and run the client.

```bash
cd third_party/py/google/adk/open_source_workspace
source .venv/bin/activate

# 1. Combine system CAs with our test CA so the client trusts the server cert
cat /usr/lib/ssl/cert.pem contributing/samples/mcp/mcp_sse_mtls_agent/ca.crt > combined_ca.pem
export SSL_CERT_FILE=$(pwd)/combined_ca.pem

# 2. Point google-auth to our simulated workload config
export GOOGLE_API_CERTIFICATE_CONFIG=$(pwd)/contributing/samples/mcp/mcp_sse_mtls_agent/certificate_config.json

# 3. Enable client certificate usage
export GOOGLE_API_USE_CLIENT_CERTIFICATE=true

# 4. Set your LLM credentials (e.g. source your env file)
source test/.env

# 5. Run the agent
adk run contributing/samples/mcp/mcp_sse_mtls_agent
```

______________________________________________________________________

## How it works

1. **Client Certificate (mTLS):** The `google-auth` library (used by ADK) reads `GOOGLE_API_CERTIFICATE_CONFIG` to load the client certificate (`client.crt`) and key (`client.key`) as a simulated Workload Certificate.
1. **Server Verification:** The server loads the CA (`ca.crt`) via `SSL_CA_CERTS` and requires the client to present a certificate signed by this CA (`ssl_cert_reqs=ssl.CERT_REQUIRED`).
1. **Client Verification:** The client trusts the server certificate (`server.crt`) because it is signed by the same CA, which we added to `SSL_CERT_FILE`.
