# Toolbox Agent

This agent is utilizing [mcp toolbox for database](https://googleapis.github.io/genai-toolbox/getting-started/introduction/) to assist end user based on the informaton stored in database.
Follow below steps to run this agent

# Install toolbox

* Run below command:

```bash
export OS="linux/amd64" # one of linux/amd64, darwin/arm64, darwin/amd64, or windows/amd64
curl -O https://storage.googleapis.com/genai-toolbox/v0.5.0/$OS/toolbox
chmod +x toolbox
```

# install SQLite

* install sqlite from https://sqlite.org/


# Create DB (optional. The db instance is already attached in the folder)

* Run below command:

```bash
sqlite3 tool_box.db
```

* Run below SQL:

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

# create tools configurations

* Create a yaml file named "tools.yaml", see its contents in the agent folder.

# start toolbox server

* Run below commands in the agent folder

```bash
toolbox --tools-file "tools.yaml"
```

# start ADK web UI

# send user query

* query 1: what can you do for me ?
* query 2: could you let know the information about "Hilton Basel" hotel ? 
