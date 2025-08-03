"""Log analysis API endpoints for Day Two SRE operations."""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import re
from collections import defaultdict, Counter

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])

class LogAnalysisRequest(BaseModel):
    log_path: str
    lines: Optional[int] = 100
    search_pattern: Optional[str] = None
    log_level: Optional[str] = None

class LogEntry(BaseModel):
    timestamp: Optional[str] = None
    level: Optional[str] = None
    message: str
    line_number: int

class LogAnalysisResponse(BaseModel):
    success: bool
    log_entries: List[LogEntry]
    summary: Dict
    error: Optional[str] = None

def parse_log_line(line: str, line_number: int) -> LogEntry:
    """Parse a log line to extract timestamp, level, and message."""
    # Common log patterns
    patterns = [
        # ISO timestamp with level: 2025-08-03 07:36:51,927 - module - INFO - message
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d+).*?-.*?- (\w+) - (.+)',
        # Simple level: INFO: message
        r'^(\w+):\s*(.+)',
        # HTTP logs: method path status
        r'(\w+)\s+("[^"]*"|\S+)\s+(\d{3})',
    ]
    
    timestamp = None
    level = None
    message = line.strip()
    
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            if len(match.groups()) == 3 and '-' in match.group(1):
                # Full timestamp pattern
                timestamp = match.group(1)
                level = match.group(2)
                message = match.group(3)
            elif len(match.groups()) == 2:
                # Simple level pattern
                level = match.group(1)
                message = match.group(2)
            break
    
    # Detect level from keywords if not found
    if not level:
        level_keywords = ['ERROR', 'WARN', 'INFO', 'DEBUG', 'CRITICAL', 'FATAL']
        for keyword in level_keywords:
            if keyword in line.upper():
                level = keyword
                break
        level = level or 'INFO'
    
    return LogEntry(
        timestamp=timestamp,
        level=level.upper(),
        message=message,
        line_number=line_number
    )

def analyze_log_entries(entries: List[LogEntry]) -> Dict:
    """Analyze log entries and provide summary statistics."""
    if not entries:
        return {"total_lines": 0}
    
    level_counts = Counter(entry.level for entry in entries)
    
    # Error analysis
    error_entries = [e for e in entries if e.level in ['ERROR', 'CRITICAL', 'FATAL']]
    warning_entries = [e for e in entries if e.level == 'WARN']
    
    # Pattern analysis
    common_errors = Counter()
    for entry in error_entries:
        # Extract common error patterns
        if 'timeout' in entry.message.lower():
            common_errors['timeout_errors'] += 1
        elif 'connection' in entry.message.lower():
            common_errors['connection_errors'] += 1
        elif 'authentication' in entry.message.lower():
            common_errors['auth_errors'] += 1
        elif 'permission' in entry.message.lower():
            common_errors['permission_errors'] += 1
        else:
            common_errors['other_errors'] += 1
    
    # Time analysis
    timestamps = [e.timestamp for e in entries if e.timestamp]
    time_range = None
    if timestamps:
        try:
            parsed_times = []
            for ts in timestamps:
                # Try to parse different timestamp formats
                for fmt in ["%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        parsed_times.append(datetime.strptime(ts[:19], fmt[:19]))
                        break
                    except ValueError:
                        continue
            
            if parsed_times:
                time_range = {
                    "start": min(parsed_times).isoformat(),
                    "end": max(parsed_times).isoformat(),
                    "duration_minutes": (max(parsed_times) - min(parsed_times)).total_seconds() / 60
                }
        except Exception:
            pass
    
    return {
        "total_lines": len(entries),
        "level_distribution": dict(level_counts),
        "error_count": len(error_entries),
        "warning_count": len(warning_entries),
        "error_patterns": dict(common_errors),
        "time_range": time_range,
        "health_score": max(0, 100 - (len(error_entries) * 10) - (len(warning_entries) * 2))
    }

@router.post("/analyze", response_model=LogAnalysisResponse)
async def analyze_logs(request: LogAnalysisRequest):
    """Analyze log file and return structured data with summary."""
    try:
        if not os.path.exists(request.log_path):
            raise HTTPException(status_code=404, detail=f"Log file not found: {request.log_path}")
        
        log_entries = []
        
        with open(request.log_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            
            # Get the last N lines if specified
            if request.lines and request.lines < len(lines):
                lines = lines[-request.lines:]
            
            for i, line in enumerate(lines, 1):
                if line.strip():  # Skip empty lines
                    # Apply search pattern filter if specified
                    if request.search_pattern and request.search_pattern.lower() not in line.lower():
                        continue
                    
                    entry = parse_log_line(line, i)
                    
                    # Apply log level filter if specified
                    if request.log_level and entry.level != request.log_level.upper():
                        continue
                    
                    log_entries.append(entry)
        
        summary = analyze_log_entries(log_entries)
        
        return LogAnalysisResponse(
            success=True,
            log_entries=log_entries,
            summary=summary
        )
        
    except Exception as e:
        return LogAnalysisResponse(
            success=False,
            log_entries=[],
            summary={},
            error=str(e)
        )

@router.get("/health")
async def logs_health():
    """Health check for log analysis service."""
    log_dir = "/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/logs"
    
    status = {
        "status": "healthy",
        "log_directory_exists": os.path.exists(log_dir),
        "available_logs": []
    }
    
    if os.path.exists(log_dir):
        try:
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            status["available_logs"] = log_files
        except Exception as e:
            status["error"] = str(e)
    
    return status

@router.get("/list")
async def list_log_files():
    """List available log files."""
    log_dir = "/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/logs"
    
    if not os.path.exists(log_dir):
        return {"success": False, "error": "Log directory not found"}
    
    try:
        log_files = []
        for filename in os.listdir(log_dir):
            if filename.endswith('.log'):
                file_path = os.path.join(log_dir, filename)
                stat = os.stat(file_path)
                log_files.append({
                    "name": filename,
                    "path": file_path,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {
            "success": True,
            "log_files": sorted(log_files, key=lambda x: x["modified"], reverse=True)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/tail/{filename}")
async def tail_log_file(filename: str, lines: int = Query(50, ge=1, le=1000)):
    """Get the last N lines from a log file."""
    log_dir = "/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/logs"
    file_path = os.path.join(log_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Log file not found: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            all_lines = file.readlines()
            tail_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            entries = []
            for i, line in enumerate(tail_lines, len(all_lines) - len(tail_lines) + 1):
                if line.strip():
                    entries.append(parse_log_line(line, i))
            
            summary = analyze_log_entries(entries)
            
            return {
                "success": True,
                "filename": filename,
                "total_lines_in_file": len(all_lines),
                "returned_lines": len(entries),
                "entries": [entry.dict() for entry in entries],
                "summary": summary
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{filename}")
async def search_log_file(
    filename: str, 
    pattern: str = Query(..., description="Search pattern"),
    case_sensitive: bool = Query(False),
    max_results: int = Query(100, ge=1, le=1000)
):
    """Search for patterns in a log file."""
    log_dir = "/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/logs"
    file_path = os.path.join(log_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Log file not found: {filename}")
    
    try:
        matches = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line_num, line in enumerate(file, 1):
                search_line = line if case_sensitive else line.lower()
                search_pattern = pattern if case_sensitive else pattern.lower()
                
                if search_pattern in search_line:
                    entry = parse_log_line(line, line_num)
                    matches.append(entry.dict())
                    
                    if len(matches) >= max_results:
                        break
        
        return {
            "success": True,
            "filename": filename,
            "pattern": pattern,
            "case_sensitive": case_sensitive,
            "matches_found": len(matches),
            "matches": matches
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))