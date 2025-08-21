"""
Service for analyzing new Google Cloud services and persisting evaluations.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import sqlite3
import json
from pathlib import Path
import logging
import time
import google.auth
from google.cloud import service_usage, iam

logger = logging.getLogger(__name__)

# Enhanced Pydantic Models
class RiskProfile(BaseModel):
    data_exposure: int = Field(..., description="Risk of data exposure (0-10).")
    misconfiguration: int = Field(..., description="Risk of misconfiguration (0-10).")
    attack_surface: int = Field(..., description="Size of the potential attack surface (0-10).")
    compliance_violation: int = Field(..., description="Risk of compliance violation (0-10).")

class SecurityAssessment(BaseModel):
    iam_permissions: List[str]
    network_exposure: str
    data_encryption: str
    compliance_certifications: List[str]
    risk_score: int
    risk_profile: RiskProfile
    threat_model_summary: str = Field(..., description="Summary of the threat model.")
    data_residency: str = Field(..., description="Information on data residency.")

class ServiceProfile(BaseModel):
    service_name: str
    description: str
    use_cases: List[str]
    security_assessment: SecurityAssessment
    release_stage: str = Field(..., description="The release stage of the service (e.g., 'GA', 'Beta').")
    is_enabled: Optional[bool] = Field(None, description="Whether the service is enabled in the project.")

# Service Implementation
class GoogleServiceAnalyzer:
    """
    Analyzes new Google Cloud services and stores the evaluation in a SQLite DB.
    """
    def __init__(self, db_path: str = "backend/data/service_evaluations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()
        self.credentials, self.project_id = self._get_credentials()

    def _get_credentials(self):
        """Loads default GCP credentials."""
        try:
            credentials, project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            logger.info(f"Successfully loaded GCP credentials for project: {project_id}")
            return credentials, project_id
        except google.auth.exceptions.DefaultCredentialsError:
            logger.error("GCP credentials not found. Please configure your environment.")
            return None, None

    def _create_schema(self):
        # ... (schema creation logic remains the same)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS service_evaluations (
                    service_name TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def _save_evaluation(self, profile: ServiceProfile):
        # ... (save logic remains the same)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO service_evaluations (service_name, profile_json) VALUES (?, ?)",
                (profile.service_name, profile.model_dump_json())
            )
            conn.commit()

    def _get_evaluation_by_name(self, service_name: str) -> Optional[ServiceProfile]:
        # ... (get logic remains the same)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT profile_json FROM service_evaluations WHERE service_name = ?", (service_name,))
            row = cursor.fetchone()
            if row:
                return ServiceProfile.model_validate_json(row[0])
        return None

    def list_all_evaluations(self) -> List[ServiceProfile]:
        # ... (list logic remains the same)
        profiles = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT profile_json FROM service_evaluations")
            rows = cursor.fetchall()
            for row in rows:
                profiles.append(ServiceProfile.model_validate_json(row[0]))
        return profiles

    def _fetch_real_service_data(self, service_name: str, project_id: str) -> Dict[str, Any]:
        """Fetches real data about a service from GCP APIs."""
        if not self.credentials:
            raise ConnectionError("GCP credentials are not configured.")

        service_usage_client = service_usage.ServiceUsageClient(credentials=self.credentials)
        iam_client = iam.IAMClient(credentials=self.credentials)

        # 1. Check if the service is enabled
        service_path = f"projects/{project_id}/services/{service_name}"
        try:
            logger.info(f"Fetching service details for {service_name}...")
            start_time = time.time()
            service_details = service_usage_client.get_service(name=service_path)
            is_enabled = service_details.state == service_usage.State.ENABLED
            logger.info(f"Fetched service details in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.warning(f"Could not get service details for {service_name}: {e}")
            is_enabled = False

        # 2. Get testable IAM permissions for the service
        try:
            logger.info(f"Fetching IAM permissions for {service_name}...")
            start_time = time.time()
            response = iam_client.query_testable_permissions(
                full_resource_name=f"//serviceusage.googleapis.com/{service_path}"
            )
            permissions = [p.name for p in response.permissions]
            logger.info(f"Fetched IAM permissions in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.warning(f"Could not query testable permissions for {service_name}: {e}")
            permissions = []

        return {"is_enabled": is_enabled, "iam_permissions": permissions}

    def analyze_new_service(self, service_name: str, project_id: str = None) -> ServiceProfile:
        """
        Retrieves an existing evaluation or generates a new one with real data.
        """
        project_id = project_id or self.project_id
        if not project_id:
            raise ValueError("A project ID must be provided.")

        existing_profile = self._get_evaluation_by_name(service_name)
        if existing_profile and existing_profile.is_enabled is not None:
             return existing_profile

        # Fetch real data from GCP
        real_data = self._fetch_real_service_data(service_name, project_id)

        # Generate a simulated profile
        if "run" in service_name.lower():
            profile = self._get_cloud_run_profile(service_name)
        else:
            profile = self._get_generic_profile(service_name)
        
        # Merge real data into the profile
        profile.is_enabled = real_data["is_enabled"]
        if real_data["iam_permissions"]:
            profile.security_assessment.iam_permissions = real_data["iam_permissions"]

        self._save_evaluation(profile)
        return profile

    # ... (profile generation methods remain the same)
    def _get_cloud_run_profile(self, service_name: str) -> ServiceProfile:
        """Returns a simulated profile for a Cloud Run-like service."""
        return ServiceProfile(
            service_name=service_name,
            description="A managed compute platform that enables you to run stateless containers.",
            use_cases=["Web services", "APIs", "Data processing pipelines"],
            release_stage="GA",
            security_assessment=SecurityAssessment(
                iam_permissions=["run.services.create", "run.services.get", "iam.serviceAccountUser"],
                network_exposure="Public endpoint by default, can be restricted to internal or VPC.",
                data_encryption="Google-managed, CMEK support in beta.",
                compliance_certifications=["SOC2", "HIPAA", "PCI-DSS"],
                risk_score=7,
                risk_profile=RiskProfile(data_exposure=8, misconfiguration=6, attack_surface=7, compliance_violation=5),
                threat_model_summary="Primary threats include unauthorized access to public endpoints and container image vulnerabilities.",
                data_residency="Data can be restricted to a specific region."
            )
        )

    def _get_generic_profile(self, service_name: str) -> ServiceProfile:
        """Returns a generic, simulated profile for an unknown service."""
        return ServiceProfile(
            service_name=service_name,
            description="A newly detected Google Cloud service.",
            use_cases=["General purpose"],
            release_stage="Beta",
            security_assessment=SecurityAssessment(
                iam_permissions=["servicemanagement.services.bind"],
                network_exposure="VPC-native",
                data_encryption="Google-managed",
                compliance_certifications=["SOC2"],
                risk_score=3,
                risk_profile=RiskProfile(data_exposure=2, misconfiguration=4, attack_surface=3, compliance_violation=3),
                threat_model_summary="Primary threats include misconfiguration of IAM permissions and potential data exfiltration.",
                data_residency="Data residency is not guaranteed."
            )
        )
