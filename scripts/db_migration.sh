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
echo "Writing safe ${ENV_FILE} (only operate on provided metadata tables)..."
cat > "${ENV_FILE}" <<EOF
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from ${MODEL_PATH} import Base

TARGET_METADATAS = (Base.metadata,)
target_metadata = TARGET_METADATAS[0]
_ALLOWED_TABLE_NAMES = frozenset(
    table_name
    for metadata in TARGET_METADATAS
    for table_name in metadata.tables
)


def include_object(obj, name, type_, reflected, compare_to):
    if type_ == "table":
        return bool(_ALLOWED_TABLE_NAMES) and name in _ALLOWED_TABLE_NAMES
    if type_ == "index":
        try:
            return obj.table.name in _ALLOWED_TABLE_NAMES
        except AttributeError:
            return False
    return True


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        version_table="alembic_version_adk",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            version_table="alembic_version_adk",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
EOF
if [ $? -ne 0 ]; then
    echo "Error: Failed to write ${ENV_FILE}."
    exit 1
fi
echo "  Set target_metadata and include_object filter"
echo ""

# --- 4. Clean up backup files ---
echo "Cleaning up backup files..."
rm -f "${INI_FILE}.bak"
rm -f "${ENV_FILE}.bak"

# --- 5. Reset stale alembic_version (if any) ---
echo "Resetting any existing alembic_version entry..."
python - <<'PY'
import configparser
import pathlib

from sqlalchemy import create_engine, text

ini = pathlib.Path("alembic.ini")
parser = configparser.ConfigParser()
parser["DEFAULT"] = {"here": str(ini.parent)}
parser.read(ini)
db_url = parser.get("alembic", "sqlalchemy.url")

engine = create_engine(db_url)
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS alembic_version_adk"))
PY
if [ $? -ne 0 ]; then
    echo "Error: Failed to reset alembic_version table."
    exit 1
fi
echo "  alembic_version reset (if it existed)."
echo ""

# --- 6. Run alembic stamp head ---
echo "Running 'alembic stamp head'..."
alembic stamp head
if [ $? -ne 0 ]; then
    echo "Error: 'alembic stamp head' failed."
    exit 1
fi
echo "stamping complete."
echo ""

# --- 7. Run alembic revision ---
echo "Running 'alembic revision --autogenerate'..."
alembic revision --autogenerate -m "ADK session DB upgrade"
if [ $? -ne 0 ]; then
    echo "Error: 'alembic revision' failed."
    exit 1
fi
echo "revision complete."
echo ""

# --- 8. Add import statement to version files ---
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

# --- 9. Run alembic upgrade ---
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
