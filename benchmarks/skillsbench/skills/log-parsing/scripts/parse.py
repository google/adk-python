"""Parse and analyze structured log files."""

import sys

SAMPLE_LOGS = """2024-01-15 10:00:01 INFO  Server started on port 8080
2024-01-15 10:00:05 DEBUG Database connection pool initialized
2024-01-15 10:01:12 INFO  User alice logged in
2024-01-15 10:02:33 WARNING High memory usage: 85%
2024-01-15 10:03:45 ERROR  Failed to process request: timeout
2024-01-15 10:04:01 INFO  User bob logged in
2024-01-15 10:05:22 ERROR  Database query failed: connection reset
2024-01-15 10:06:10 WARNING Disk space low: 10% remaining
2024-01-15 10:07:00 INFO  Scheduled backup started
2024-01-15 10:08:15 ERROR  Backup failed: permission denied
2024-01-15 10:09:30 INFO  User alice logged out
2024-01-15 10:10:00 DEBUG Cache cleared"""


def parse_entry(line):
  parts = line.split(None, 3)
  if len(parts) < 4:
    return None
  date, time, level, message = parts
  return {
      "timestamp": f"{date} {time}",
      "level": level.strip(),
      "message": message.strip(),
  }


def parse_args(args):
  params = {}
  for arg in args:
    if "=" in arg:
      key, value = arg.split("=", 1)
      params[key] = value
  return params


def main():
  params = parse_args(sys.argv[1:])
  level_filter = params.get("level", "ALL").upper()
  out_format = params.get("format", "summary")

  entries = []
  for line in SAMPLE_LOGS.strip().split("\n"):
    entry = parse_entry(line)
    if entry:
      entries.append(entry)

  if level_filter != "ALL":
    filtered = [e for e in entries if e["level"] == level_filter]
  else:
    filtered = entries

  if out_format == "summary":
    counts = {}
    for entry in entries:
      level = entry["level"]
      counts[level] = counts.get(level, 0) + 1

    print("Log Analysis Report")
    print("===================")
    print(f"Total entries: {len(entries)}")
    for level in ["ERROR", "WARNING", "INFO", "DEBUG"]:
      print(f"{level}: {counts.get(level, 0)}")

  elif out_format == "detail":
    for entry in filtered:
      print(f"[{entry['timestamp']}] {entry['level']} - {entry['message']}")

  elif out_format == "timeline":
    for entry in filtered:
      time_part = entry["timestamp"].split(" ")[1]
      print(f"{time_part} {entry['level']}: {entry['message']}")


if __name__ == "__main__":
  main()
