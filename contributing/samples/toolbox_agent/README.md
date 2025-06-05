# Toolbox Agent

This agent utilizes [MCP toolbox for database](https://googleapis.github.io/genai-toolbox/getting-started/introduction/) to assist end users based on information stored in a database.

Follow the steps below to run this agent.

## Prerequisites

Before starting, ensure you have Python installed on your system.

## Installation Steps

### 1. Install Toolbox

Run the following command to download and install the toolbox:

```bash
export OS="linux/amd64" # one of linux/amd64, darwin/arm64, darwin/amd64, or windows/amd64
curl -O https://storage.googleapis.com/genai-toolbox/v0.5.0/$OS/toolbox
chmod +x toolbox
```

### 2. Install SQLite

Install SQLite from [https://sqlite.org/](https://sqlite.org/)

### 3. Install Required Python Dependencies

**Important**: The ADK's `ToolboxToolset` class requires the `toolbox-core` package, which is not automatically installed with the ADK. Install it using:

```bash
pip install toolbox-core
```

### 4. Create Database (Optional)

*Note: A database instance is already included in the project folder. Skip this step if you want to use the existing database.*

To create a new database:

```bash
sqlite3 tool_box.db
```

Run the following SQL commands to set up the hotels table:

```sql
CREATE TABLE hotels(
  id            INTEGER NOT NULL PRIMARY KEY,
  name          VARCHAR NOT NULL,
  location      VARCHAR NOT NULL,
  price_tier    VARCHAR NOT NULL,
  checkin_date  DATE    NOT NULL,
  checkout_date DATE    NOT NULL,
  booked        BIT     NOT NULL
);

INSERT INTO hotels(id, name, location, price_tier, checkin_date, checkout_date, booked)
VALUES 
  (1, 'Hilton Basel', 'Basel', 'Luxury', '2024-04-22', '2024-04-20', 0),
  (2, 'Marriott Zurich', 'Zurich', 'Upscale', '2024-04-14', '2024-04-21', 0),
  (3, 'Hyatt Regency Basel', 'Basel', 'Upper Upscale', '2024-04-02', '2024-04-20', 0),
  (4, 'Radisson Blu Lucerne', 'Lucerne', 'Midscale', '2024-04-24', '2024-04-05', 0),
  (5, 'Best Western Bern', 'Bern', 'Upper Midscale', '2024-04-23', '2024-04-01', 0),
  (6, 'InterContinental Geneva', 'Geneva', 'Luxury', '2024-04-23', '2024-04-28', 0),
  (7, 'Sheraton Zurich', 'Zurich', 'Upper Upscale', '2024-04-27', '2024-04-02', 0),
  (8, 'Holiday Inn Basel', 'Basel', 'Upper Midscale', '2024-04-24', '2024-04-09', 0),
  (9, 'Courtyard Zurich', 'Zurich', 'Upscale', '2024-04-03', '2024-04-13', 0),
  (10, 'Comfort Inn Bern', 'Bern', 'Midscale', '2024-04-04', '2024-04-16', 0);
```

### 5. Create Tools Configuration

Create a YAML file named `tools.yaml`. See the contents in the agent folder for reference.

### 6. Start Toolbox Server

Run the following command in the agent folder:

```bash
toolbox --tools-file "tools.yaml"
```

The server will start at `http://127.0.0.1:5000` by default.

### 7. Start ADK Web UI

Follow the ADK documentation to start the web user interface.

## Testing the Agent

Once everything is set up, you can test the agent with these sample queries:

- **Query 1**: "What can you do for me?"
- **Query 2**: "Could you let me know the information about 'Hilton Basel' hotel?"
