#!/bin/bash


# This script is to update sessions DB that is created in previous ADK version,
# to schema that current ADK version use. The sample usage is in the samples/migrate_session_db.
#
# Usage:
# ./db_migration.sh "sqlite:///%(here)s/sessions.db" "google.adk.sessions.database_session_service"
# ./db_migration.sh "postgresql://user:pass@localhost/mydb" "google.adk.sessions.database_session_service"
# First argument is the sessions DB url.
# Second argument is the model import path.

# --- Configuration ---
ALEMBIC_DIR="alembic"
INI_FILE="alembic.ini"
ENV_FILE="${ALEMBIC_DIR}/env.py"

# --- Functions ---
print_usage() {
    echo "Usage: $0 <sqlalchemy_url> <model_import_path>"
    echo "  <sqlalchemy_url>: The full SQLAlchemy connection string."
    echo "  <model_import_path>: The Python import path to your models (e.g., my_project.models)"
    echo ""
    echo "Example:"
    echo "  $0 \"sqlite:///%(here)s/sessions.db\" \"google.adk.sessions.database_session_service\""
}

# --- Argument Validation ---
if [ "$#" -ne 2 ]; then
    print_usage
    exit 1
fi

DB_URL=$1
MODEL_PATH=$2

echo "Setting up Alembic..."
echo "  Database URL: ${DB_URL}"
echo "  Model Path:   ${MODEL_PATH}"
echo ""

# --- Safety Check ---
if [ -f "$INI_FILE" ] || [ -d "$ALEMBIC_DIR" ]; then
    echo "Error: 'alembic.ini' or 'alembic/' directory already exists."
    echo "Please remove them before running this script."
    exit 1
fi

# --- 1. Run alembic init ---
echo "Running 'alembic init ${ALEMBIC_DIR}'..."
alembic init ${ALEMBIC_DIR}
if [ $? -ne 0 ]; then
    echo "Error: 'alembic init' failed. Is alembic installed?"
    exit 1
fi
echo "Initialization complete."
echo ""

# --- 2. Set sqlalchemy.url in alembic.ini ---
echo "Configuring ${INI_FILE}..."
# Use a different delimiter (#) for sed to avoid escaping slashes in the URL
sed -i.bak "s#sqlalchemy.url = driver://user:pass@localhost/dbname#sqlalchemy.url = ${DB_URL}#" "${INI_FILE}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to set sqlalchemy.url in ${INI_FILE}."
    exit 1
fi
echo "  Set sqlalchemy.url"

# --- 3. Set target_metadata in alembic/env.py ---
echo "Configuring ${ENV_FILE}..."

# Edit 1: Uncomment and replace the model import line
sed -i.bak "s/# from myapp import mymodel/from ${MODEL_PATH} import Base/" "${ENV_FILE}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to set model import in ${ENV_FILE}."
    exit 1
fi

# Edit 2: Set the target_metadata to use the imported Base
sed -i.bak "s/target_metadata = None/target_metadata = Base.metadata/" "${ENV_FILE}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to set target_metadata in ${ENV_FILE}."
    exit 1
fi

echo "  Set target_metadata"
echo ""

# --- 4. Clean up backup files ---
echo "Cleaning up backup files..."
rm "${INI_FILE}.bak"
rm "${ENV_FILE}.bak"

# --- 5. Run alembic stamp head ---
echo "Running 'alembic stamp head'..."
alembic stamp head
if [ $? -ne 0 ]; then
    echo "Error: 'alembic stamp head' failed."
    exit 1
fi
echo "stamping complete."
echo ""

# --- 6. Run alembic upgrade ---
echo "Running 'alembic revision --autogenerate'..."
alembic revision --autogenerate -m "ADK session DB upgrade"
if [ $? -ne 0 ]; then
    echo "Error: 'alembic revision' failed."
    exit 1
fi
echo "revision complete."
echo ""

# --- 7. Add import statement to version files ---
echo "Adding import statement to version files..."
for f in ${ALEMBIC_DIR}/versions/*.py; do
  if [ -f "$f" ]; then
    # Check if the first line is already the import statement
    FIRST_LINE=$(head -n 1 "$f")
    IMPORT_STATEMENT="import ${MODEL_PATH}"
    if [ "$FIRST_LINE" != "$IMPORT_STATEMENT" ]; then
      echo "Adding import to $f"
      sed -i.bak "1s|^|${IMPORT_STATEMENT}\n|" "$f"
      rm "${f}.bak"
    else
      echo "Import already exists in $f"
    fi
  fi
done
echo "Import statements added."
echo ""

# --- 8. Run alembic upgrade ---
echo "running 'alembic upgrade'..."
alembic upgrade head
if [ $? -ne 0 ]; then
    echo "Error: 'alembic upgrade' failed. "
    exit 1
fi
echo "upgrade complete."
echo ""

echo "---"
echo "✅ ADK session DB is Updated!"