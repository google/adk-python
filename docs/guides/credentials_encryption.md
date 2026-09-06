# Session Credentials Encryption Guide

To prevent sensitive OAuth 2 credentials (like access tokens, refresh tokens, and client secrets) from being stored in plaintext inside the session state database, ADK supports encrypting them using Google Cloud KMS with **Envelope Encryption**.

## How It Works

1. **Envelope Encryption for Google OAuth Credentials**:
   * **Data Encryption Key (DEK)**: A local 256-bit symmetric key (Fernet) is generated locally to encrypt the sensitive fields (`access_token`, `refresh_token`, `client_secret`).
   * **Key Encryption Key (KEK)**: The Google Cloud KMS key acts as the KEK and is used to encrypt (wrap) the local DEK.
   * **Storage**: The session stores the locally encrypted credentials, the public reference of the KMS key (`kms_key_name`), and the encrypted DEK (`wrapped_dek`).
2. **Direct KMS Encryption for Generic Credentials (`SessionStateCredentialService`)**:
   * All non-OAuth credentials (API keys, HTTP Basic Auth, Bearer tokens, Service Account private keys) saved to session state via `SessionStateCredentialService` are automatically encrypted using Cloud KMS on save (`save_credential`) and decrypted on load (`load_credential`).
   * Encrypted values are stored in state with a `kms:` prefix.
3. **In-Memory Caching (Zero Latency)**:
   * To prevent performing a slow GCP KMS network request on every field encryption or decryption, the resolved plaintext DEK and its corresponding `wrapped_dek` are cached in-memory.
   * On deserialization, KMS is called **exactly once** per session load, and subsequent decryptions are processed locally in-memory (instantaneous). On serialization, we reuse the cached wrapped DEK (zero KMS calls).
4. **Re-Authentication Fallback (No-Crash)**:
   * If Cloud KMS decryption fails (e.g. key destroyed, IAM permission revoked, or key version unavailable), `SessionStateCredentialService` logs a warning and returns `None`, gracefully triggering user re-authentication instead of throwing validation errors.
5. **Backward Compatibility**: If no KMS key is configured or the stored credentials do not contain encrypted values, ADK automatically falls back to loading/saving them in plaintext without raising errors.

---

## Configuration

Set the environment variable `GOOGLE_CREDENTIAL_KMS_KEY` to point to your GCP KMS CryptoKey (optionally pinning a specific version):

```bash
export GOOGLE_CREDENTIAL_KMS_KEY="projects/{project_id}/locations/{location}/keyRings/{key_ring_name}/cryptoKeys/{key_name}/cryptoKeyVersions/{version_id}"
```

Alternatively, you can configure it programmatically on any `CredentialsConfig` (like `BigQueryCredentialsConfig`):

```python
oauth_credentials_config = BigQueryCredentialsConfig(
    client_id=client_id,
    client_secret=client_secret,
    scopes=scopes,
    kms_key_name="projects/{project_id}/locations/{location}/keyRings/{key_ring_name}/cryptoKeys/{key_name}/cryptoKeyVersions/{version_id}"
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
