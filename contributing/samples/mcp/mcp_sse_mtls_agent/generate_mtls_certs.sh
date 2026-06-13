#!/bin/bash
set -e

# Directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Generating certificates in $DIR..."

# 1. Create CA
openssl req -x509 -new -nodes -newkey rsa:2048 -keyout ca.key -sha256 -days 365 -out ca.crt -subj '/CN=TestCA'

# 2. Create Server Cert
openssl req -new -nodes -newkey rsa:2048 -keyout server.key -out server.csr -subj '/CN=localhost'
# Sign with CA
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -sha256

# 3. Create Client Cert
openssl req -new -nodes -newkey rsa:2048 -keyout client.key -out client.csr -subj '/CN=TestClient'
# Sign with CA
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365 -sha256

# Clean up CSRs and serial file
rm -f server.csr client.csr ca.srl

# 4. Create certificate_config.json
cat <<EOF > certificate_config.json
{
  "cert_configs": {
    "workload": {
      "cert_path": "$DIR/client.crt",
      "key_path": "$DIR/client.key"
    }
  }
}
EOF

echo "Done! Generated:"
echo "  - ca.crt, ca.key (CA)"
echo "  - server.crt, server.key (Server cert)"
echo "  - client.crt, client.key (Client cert)"
echo "  - certificate_config.json (Workload config for google-auth)"
