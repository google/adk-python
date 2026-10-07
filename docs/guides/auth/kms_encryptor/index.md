# Session Credentials Encryption Guide

To prevent sensitive OAuth 2 credentials (like access tokens, refresh tokens, and client secrets) from being stored in plaintext inside the session state database, ADK tools using `BaseGoogleCredentialsConfig` support encrypting them using Google Cloud KMS with **Envelope Encryption**.

## How It Works

1. **Envelope Encryption Pattern**:
   * **Data Encryption Key (DEK)**: A local 256-bit key (Fernet, deriving a 128-bit AES encryption key and a 128-bit HMAC key) is generated locally to encrypt sensitive fields (`access_token`, `refresh_token`, `client_secret`).
   * **Key Encryption Key (KEK)**: The Google Cloud KMS key acts as the KEK and is used to encrypt (wrap) the local DEK.
   * **Storage**: The session stores the locally encrypted credentials, the public reference of the KMS key (`kms_key_name`), and the encrypted DEK (`wrapped_dek`).
2. **In-Memory Caching (Zero Latency)**:
   * To prevent performing a slow GCP KMS network request on every field encryption or decryption, the resolved plaintext DEK and its corresponding `wrapped_dek` are cached in-memory.
   * On deserialization, KMS is called **once per process** to unwrap the DEK, and subsequent decryptions are processed locally in-memory (instantaneous). On serialization, the newly generated DEK is wrapped with KMS on first use, and subsequent serializations reuse the cached wrapped DEK with zero KMS calls.
3. **Backward Compatibility**: If no KMS key is configured or the stored credentials do not contain a `wrapped_dek`, ADK automatically falls back to loading/saving them in plaintext without raising errors.

---

## Installation

Session credentials encryption requires Google Cloud KMS support. Install it with the `gcp` extra:

```bash
pip install "google-adk[gcp]"
```

---

## Configuration

Set the environment variable `GOOGLE_CREDENTIAL_KMS_KEY` to point to your GCP KMS CryptoKey:

```bash
export GOOGLE_CREDENTIAL_KMS_KEY="projects/{project_id}/locations/{location}/keyRings/{key_ring_name}/cryptoKeys/{key_name}"
```

Alternatively, you can configure it programmatically on any `CredentialsConfig` (like `BigQueryCredentialsConfig`):

```python
oauth_credentials_config = BigQueryCredentialsConfig(
    client_id=client_id,
    client_secret=client_secret,
    scopes=scopes,
    kms_key_name="projects/{project_id}/locations/{location}/keyRings/{key_ring_name}/cryptoKeys/{key_name}",
)
```

---

## Required IAM Permissions

The Service Account running the ADK Agent / Runner must be granted the appropriate permissions to call the Cloud KMS API.

### KMS Permissions
* **Role**: `Cloud KMS CryptoKey Encrypter/Decrypter` (`roles/cloudkms.cryptoKeyEncrypterDecrypter`)
* **Scope**: Must be granted on the specified CryptoKey or KeyRing.

Example `gcloud` command to grant access:

```bash
gcloud kms keys add-iam-policy-binding {key_name} \
    --location={location} \
    --keyring={key_ring_name} \
    --member="serviceAccount:{agent_service_account}@{project_id}.iam.gserviceaccount.com" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

---

## Key Rotation

* **CryptoKey Version Rotation**: Rotating a CryptoKey in Cloud KMS keeps previous key versions enabled for decryption. Cloud KMS `decrypt` takes a CryptoKey and determines the key version directly from the ciphertext metadata, so rotating key versions does not invalidate stored credentials.
* **Repointing to a Different CryptoKey**: If `kms_key_name` or `GOOGLE_CREDENTIAL_KMS_KEY` is changed to point to a *different* CryptoKey, credentials whose `wrapped_dek` was encrypted under the old key cannot be decrypted. When attempting to load them, ADK safely discards the stale cached credentials (logging a warning) and prompts the user to re-authenticate via OAuth, then encrypts newly generated credentials using the new key.
