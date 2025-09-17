"""
Knowledge Base API Endpoints
============================

FastAPI endpoints for managing enterprise security policies, coding standards,
compliance frameworks, and best practices.
"""

from fastapi import APIRouter, HTTPException, Query, Path, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import sqlite3
import csv
import io
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge_base"])


# Pydantic Models
class EnterprisePolicy(BaseModel):
    """Enterprise security policy model"""
    id: Optional[int] = None
    category: str
    policy_name: str
    description: str
    severity: str = Field(..., pattern="^(CRITICAL|HIGH|MEDIUM|LOW)$")
    implementation_guide: Optional[str] = None
    exceptions: Optional[str] = None
    tags: Optional[List[str]] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    version: Optional[int] = 1
    is_active: Optional[bool] = True


class CodingStandard(BaseModel):
    """Coding standard model"""
    id: Optional[int] = None
    language: str
    standard_name: str
    rule_description: str
    example_good: Optional[str] = None
    example_bad: Optional[str] = None
    auto_fixable: Optional[bool] = False
    linter_rule: Optional[str] = None
    severity: str = Field(..., pattern="^(ERROR|WARNING|INFO)$")
    tags: Optional[List[str]] = []
    is_active: Optional[bool] = True


class ComplianceFramework(BaseModel):
    """Compliance framework requirement model"""
    id: Optional[int] = None
    framework_name: str
    requirement_id: str
    requirement_text: str
    description: Optional[str] = None
    gcp_mapping: Optional[List[str]] = []
    evidence_required: Optional[str] = None
    audit_frequency: Optional[str] = None
    last_audit_date: Optional[str] = None
    compliance_status: Optional[str] = Field(None, pattern="^(COMPLIANT|NON_COMPLIANT|PARTIAL|NOT_ASSESSED)$")
    remediation_steps: Optional[str] = None
    tags: Optional[List[str]] = []


class BestPractice(BaseModel):
    """Best practice model"""
    id: Optional[int] = None
    service: str
    practice_name: str
    category: str
    rationale: str
    implementation_guide: str
    risk_if_not_followed: Optional[str] = None
    automation_possible: Optional[bool] = False
    terraform_snippet: Optional[str] = None
    gcloud_command: Optional[str] = None
    verification_steps: Optional[str] = None
    reference_links: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    is_active: Optional[bool] = True


class PolicyViolation(BaseModel):
    """Policy violation tracking model"""
    id: Optional[int] = None
    policy_type: str
    policy_id: int
    resource_type: Optional[str] = None
    resource_name: Optional[str] = None
    violation_details: str
    severity: str
    detected_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    false_positive: Optional[bool] = False


# Database connection
def get_db_connection():
    """Get database connection"""
    import os
    from pathlib import Path
    
    # Try different paths to find the database
    db_path = os.getenv("KNOWLEDGE_BASE_DB_PATH", "backend/cache/knowledge_base.db")
    db_path = Path(db_path)
    
    if not db_path:
        # Create the database if it doesn't exist
        db_path = Path(__file__).parent.parent / "cache" / "knowledge_base.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# Enterprise Policies Endpoints
@router.get("/policies", response_model=List[EnterprisePolicy])
async def get_policies(
    category: Optional[str] = None,
    severity: Optional[str] = None,
    is_active: Optional[bool] = True,
    search: Optional[str] = None
):
    """Get all enterprise policies with optional filters"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM enterprise_policies WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if is_active is not None:
            query += " AND is_active = ?"
            params.append(1 if is_active else 0)
        
        if search:
            query += " AND (policy_name LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        cursor.execute(query, params)
        policies = cursor.fetchall()
        
        result = []
        for policy in policies:
            policy_dict = dict(policy)
            if policy_dict.get("tags"):
                policy_dict["tags"] = json.loads(policy_dict["tags"])
            result.append(EnterprisePolicy(**policy_dict))
        
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error fetching policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policies", response_model=EnterprisePolicy)
async def create_policy(policy: EnterprisePolicy):
    """Create a new enterprise policy"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO enterprise_policies 
            (category, policy_name, description, severity, implementation_guide, 
             exceptions, tags, created_by, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            policy.category, policy.policy_name, policy.description,
            policy.severity, policy.implementation_guide, policy.exceptions,
            json.dumps(policy.tags) if policy.tags else None,
            policy.created_by, 1 if policy.is_active else 0
        ))
        
        policy.id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return policy
        
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=400, detail="Policy name already exists")
    except Exception as e:
        logger.error(f"Error creating policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/policies/{policy_id}", response_model=EnterprisePolicy)
async def update_policy(policy_id: int, policy: EnterprisePolicy):
    """Update an existing policy"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE enterprise_policies 
            SET category = ?, policy_name = ?, description = ?, severity = ?,
                implementation_guide = ?, exceptions = ?, tags = ?, version = version + 1
            WHERE id = ?
        """, (
            policy.category, policy.policy_name, policy.description,
            policy.severity, policy.implementation_guide, policy.exceptions,
            json.dumps(policy.tags) if policy.tags else None,
            policy_id
        ))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        conn.commit()
        conn.close()
        
        policy.id = policy_id
        return policy
        
    except Exception as e:
        logger.error(f"Error updating policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/policies/{policy_id}")
async def delete_policy(policy_id: int):
    """Soft delete a policy"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE enterprise_policies 
            SET is_active = 0
            WHERE id = ?
        """, (policy_id,))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        conn.commit()
        conn.close()
        
        return {"message": "Policy deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Coding Standards Endpoints
@router.get("/standards", response_model=List[CodingStandard])
async def get_standards(
    language: Optional[str] = None,
    severity: Optional[str] = None,
    search: Optional[str] = None
):
    """Get all coding standards with optional filters"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM coding_standards WHERE is_active = 1"
        params = []
        
        if language:
            query += " AND language = ?"
            params.append(language)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if search:
            query += " AND (standard_name LIKE ? OR rule_description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        cursor.execute(query, params)
        standards = cursor.fetchall()
        
        result = []
        for standard in standards:
            standard_dict = dict(standard)
            if standard_dict.get("tags"):
                standard_dict["tags"] = json.loads(standard_dict["tags"])
            result.append(CodingStandard(**standard_dict))
        
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error fetching standards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/standards", response_model=CodingStandard)
async def create_standard(standard: CodingStandard):
    """Create a new coding standard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO coding_standards 
            (language, standard_name, rule_description, example_good, example_bad,
             auto_fixable, linter_rule, severity, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            standard.language, standard.standard_name, standard.rule_description,
            standard.example_good, standard.example_bad, 
            1 if standard.auto_fixable else 0,
            standard.linter_rule, standard.severity,
            json.dumps(standard.tags) if standard.tags else None
        ))
        
        standard.id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return standard
        
    except Exception as e:
        logger.error(f"Error creating standard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Compliance Frameworks Endpoints
@router.get("/compliance", response_model=List[ComplianceFramework])
async def get_compliance_requirements(
    framework: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None
):
    """Get compliance framework requirements"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM compliance_frameworks WHERE 1=1"
        params = []
        
        if framework:
            query += " AND framework_name = ?"
            params.append(framework)
        
        if status:
            query += " AND compliance_status = ?"
            params.append(status)
        
        if search:
            query += " AND (requirement_text LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        cursor.execute(query, params)
        requirements = cursor.fetchall()
        
        result = []
        for req in requirements:
            req_dict = dict(req)
            if req_dict.get("gcp_mapping"):
                req_dict["gcp_mapping"] = json.loads(req_dict["gcp_mapping"])
            if req_dict.get("tags"):
                req_dict["tags"] = json.loads(req_dict["tags"])
            result.append(ComplianceFramework(**req_dict))
        
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error fetching compliance requirements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Best Practices Endpoints
@router.get("/practices", response_model=List[BestPractice])
async def get_best_practices(
    service: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None
):
    """Get best practices with optional filters"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM best_practices WHERE is_active = 1"
        params = []
        
        if service:
            query += " AND service = ?"
            params.append(service)
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if search:
            query += " AND (practice_name LIKE ? OR rationale LIKE ? OR implementation_guide LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        
        cursor.execute(query, params)
        practices = cursor.fetchall()
        
        result = []
        for practice in practices:
            practice_dict = dict(practice)
            if practice_dict.get("reference_links"):
                practice_dict["reference_links"] = json.loads(practice_dict["reference_links"])
            if practice_dict.get("tags"):
                practice_dict["tags"] = json.loads(practice_dict["tags"])
            result.append(BestPractice(**practice_dict))
        
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error fetching best practices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import/Export Endpoints
@router.post("/import/csv")
async def import_csv(
    file: UploadFile = File(...),
    table: str = Query(..., pattern="^(policies|standards|compliance|practices)$")
):
    """Import data from CSV file"""
    try:
        contents = await file.read()
        csv_data = csv.DictReader(io.StringIO(contents.decode('utf-8')))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        count = 0
        for row in csv_data:
            if table == "policies":
                cursor.execute("""
                    INSERT OR IGNORE INTO enterprise_policies 
                    (category, policy_name, description, severity, implementation_guide, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    row.get("category"), row.get("policy_name"), row.get("description"),
                    row.get("severity"), row.get("implementation_guide"), 
                    json.dumps(row.get("tags", "").split(","))
                ))
            # Add similar logic for other tables
            count += 1
        
        conn.commit()
        conn.close()
        
        return {"message": f"Successfully imported {count} records"}
        
    except Exception as e:
        logger.error(f"Error importing CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{table}")
async def export_data(
    table: str = Path(..., pattern="^(policies|standards|compliance|practices)$"),
    format: str = Query("json", pattern="^(json|csv)$")
):
    """Export knowledge base data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        table_map = {
            "policies": "enterprise_policies",
            "standards": "coding_standards",
            "compliance": "compliance_frameworks",
            "practices": "best_practices"
        }
        
        cursor.execute(f"SELECT * FROM {table_map[table]}")
        data = cursor.fetchall()
        
        result = []
        for row in data:
            row_dict = dict(row)
            # Parse JSON fields
            for field in ["tags", "gcp_mapping", "reference_links"]:
                if field in row_dict and row_dict[field]:
                    row_dict[field] = json.loads(row_dict[field])
            result.append(row_dict)
        
        conn.close()
        
        if format == "csv":
            # Convert to CSV format
            output = io.StringIO()
            if result:
                writer = csv.DictWriter(output, fieldnames=result[0].keys())
                writer.writeheader()
                writer.writerows(result)
            return output.getvalue()
        else:
            return result
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search Endpoint
@router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=100)
):
    """Full-text search across all knowledge base tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        results = {
            "policies": [],
            "standards": [],
            "practices": [],
            "compliance": []
        }
        
        # Search policies
        cursor.execute("""
            SELECT * FROM enterprise_policies 
            WHERE policy_name LIKE ? OR description LIKE ? OR implementation_guide LIKE ?
            AND is_active = 1
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        
        for row in cursor.fetchall():
            row_dict = dict(row)
            if row_dict.get("tags"):
                row_dict["tags"] = json.loads(row_dict["tags"])
            results["policies"].append(row_dict)
        
        # Search standards
        cursor.execute("""
            SELECT * FROM coding_standards 
            WHERE standard_name LIKE ? OR rule_description LIKE ?
            AND is_active = 1
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit))
        
        for row in cursor.fetchall():
            row_dict = dict(row)
            if row_dict.get("tags"):
                row_dict["tags"] = json.loads(row_dict["tags"])
            results["standards"].append(row_dict)
        
        # Search best practices
        cursor.execute("""
            SELECT * FROM best_practices 
            WHERE practice_name LIKE ? OR rationale LIKE ? OR implementation_guide LIKE ?
            AND is_active = 1
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        
        for row in cursor.fetchall():
            row_dict = dict(row)
            if row_dict.get("tags"):
                row_dict["tags"] = json.loads(row_dict["tags"])
            if row_dict.get("references"):
                row_dict["references"] = json.loads(row_dict["references"])
            results["practices"].append(row_dict)
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics Endpoint
@router.get("/stats")
async def get_knowledge_base_stats():
    """Get statistics about the knowledge base"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Count policies
        cursor.execute("SELECT COUNT(*) as total, COUNT(CASE WHEN is_active = 1 THEN 1 END) as active FROM enterprise_policies")
        row = cursor.fetchone()
        stats["policies"] = {"total": row["total"], "active": row["active"]}
        
        # Count standards
        cursor.execute("SELECT COUNT(*) as total, COUNT(CASE WHEN is_active = 1 THEN 1 END) as active FROM coding_standards")
        row = cursor.fetchone()
        stats["standards"] = {"total": row["total"], "active": row["active"]}
        
        # Count compliance requirements
        cursor.execute("SELECT COUNT(*) as total FROM compliance_frameworks")
        stats["compliance"] = {"total": cursor.fetchone()["total"]}
        
        # Count best practices
        cursor.execute("SELECT COUNT(*) as total, COUNT(CASE WHEN is_active = 1 THEN 1 END) as active FROM best_practices")
        row = cursor.fetchone()
        stats["practices"] = {"total": row["total"], "active": row["active"]}
        
        # Count violations
        cursor.execute("""
            SELECT COUNT(*) as total, 
                   COUNT(CASE WHEN resolved_at IS NULL THEN 1 END) as unresolved
            FROM policy_violations
        """)
        row = cursor.fetchone()
        stats["violations"] = {"total": row["total"], "unresolved": row["unresolved"]}
        
        conn.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))