#!/bin/bash

# Archive duplicate documentation from security_agent/docs to docs/legacy

echo "📦 Archiving duplicate documentation..."

# Create legacy archive directory
mkdir -p /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/docs/legacy

# Archive duplicate/obsolete files
LEGACY_DIR="/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/docs/legacy"
SOURCE_DIR="/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/docs/architecture"

# Files to archive (duplicates/obsolete)
FILES_TO_ARCHIVE=(
    "IMPROVED_ARCHITECTURE_DIAGRAMS.md"
    "CHAT_CENTRIC_ARCHITECTURE.md"
    "CHAT_CENTRIC_IMPLEMENTATION_SUMMARY.md"
    "GCP_SECURITY_CHAT_ARCHITECTURE.md"
    "PROPOSED_ARCHITECTURE.md"
    "Overall.md"
    "# fixes.md"
    "CLEANUP_SUMMARY.md"
    "MISSING_COMPONENTS.md"
)

# Archive each file if it exists
for file in "${FILES_TO_ARCHIVE[@]}"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        cp "$SOURCE_DIR/$file" "$LEGACY_DIR/"
        echo "  ✅ Archived: $file"
    fi
done

# Create a README in legacy directory
cat > "$LEGACY_DIR/README.md" << 'EOF'
# Legacy Documentation Archive

This directory contains archived documentation that has been superseded or consolidated into the main documentation structure.

## Why These Files Were Archived

- **Duplicates**: Content merged into primary documentation
- **Obsolete**: Outdated architecture proposals or implementation details
- **Temporary**: Work-in-progress files that are no longer needed

## Current Documentation

Please refer to the consolidated documentation in:
- `/docs/` - Primary documentation following BMad method
- `/docs/architecture/` - Architecture and technical documentation
- `/docs/guides/` - User and operational guides
- `/docs/reference/` - API and pattern references

## Archive Date
January 18, 2025
EOF

echo "✅ Archive complete. Legacy docs saved to: $LEGACY_DIR"
echo ""
echo "📝 Next steps:"
echo "  1. Review the consolidated documentation"
echo "  2. Verify all important content has been preserved"
echo "  3. Remove the old security_agent/docs directory when ready"