# Fix for Issue #3622: Accept dict-shaped artifacts in InMemoryArtifactService

## Summary
Fixed the artifact services to accept dict-shaped (serialized) artifacts in addition to `types.Part` objects. This allows users to pass artifacts as dictionaries, which are automatically converted to `types.Part` objects internally.

## Changes Made

### 1. Base Artifact Service (`base_artifact_service.py`)
- Updated the `save_artifact()` method signature to accept `types.Part | dict[str, Any]`
- Updated the docstring to clarify that dict-shaped artifacts are now supported

### 2. InMemoryArtifactService (`in_memory_artifact_service.py`)
- Updated the `save_artifact()` method to:
  - Accept `types.Part | dict[str, Any]` parameter type
  - Added conversion logic: `if isinstance(artifact, dict): artifact = types.Part.model_validate(artifact)`
  - This deserialization happens before any artifact processing

### 3. GcsArtifactService (`gcs_artifact_service.py`)
- Updated the async `save_artifact()` method to:
  - Accept `types.Part | dict[str, Any]` parameter type
  - Added conversion logic before threading to sync method
- The internal `_save_artifact()` method processes the already-converted `types.Part` object

### 4. FileArtifactService (`file_artifact_service.py`)
- Updated the async `save_artifact()` method to:
  - Accept `types.Part | dict[str, Any]` parameter type
  - Added conversion logic before threading to sync method
- The internal `_save_artifact_sync()` method processes the already-converted `types.Part` object

### 5. ForwardingArtifactService (`_forwarding_artifact_service.py`)
- Updated the `save_artifact()` method to:
  - Accept `types.Part | dict[str, Any]` parameter type
  - Added conversion logic before forwarding to the parent tool context

### 6. Test Suite (`test_artifact_service.py`)
- Added `test_save_load_dict_shaped_artifact()` test to verify dict-shaped artifacts can be saved and loaded across all service types (IN_MEMORY, GCS, FILE)
- Added `test_save_text_dict_shaped_artifact()` test to verify text-based dict-shaped artifacts work correctly in InMemoryArtifactService

## How It Works

When a dictionary is passed to `save_artifact()`:
1. The method checks if the artifact is a dictionary using `isinstance(artifact, dict)`
2. If it is, it converts it to a `types.Part` object using `types.Part.model_validate(artifact)`
3. The rest of the method processes the converted `types.Part` object as usual

## Example Usage

```python
# Before (still supported)
artifact = types.Part(text="Hello, World!")
await service.save_artifact(..., artifact=artifact)

# After (now also supported)
artifact_dict = {"text": "Hello, World!"}
await service.save_artifact(..., artifact=artifact_dict)

# Also works with inline data
artifact_dict = {
    "inline_data": {
        "data": "dGVzdF9kYXRh",  # base64 encoded
        "mime_type": "text/plain",
    }
}
await service.save_artifact(..., artifact=artifact_dict)
```

## Backward Compatibility

✅ **Fully backward compatible** - All existing code using `types.Part` objects will continue to work exactly as before.
