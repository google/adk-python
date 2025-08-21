# STORY-301: New GCP Service Evaluation Framework
**Epic:** Service Discovery & Integration  
**Priority:** P1 - High  
**Estimated Effort:** 8-13 story points  
**Status:** Draft  

## 📋 Story Description

**As a** Security Operations Engineer  
**I want** to automatically evaluate and integrate new GCP services (like Vertex AI Memory Store) into our security monitoring system  
**So that** I can maintain comprehensive security coverage as Google Cloud evolves and ensure no new attack surfaces go unmonitored  

### 🎯 Business Value
- **Risk Mitigation**: Automatically discover security implications of new GCP services
- **Compliance**: Ensure new services meet organizational security standards
- **Operational Efficiency**: Standardized evaluation process reduces manual effort
- **Security Coverage**: No gaps in monitoring as GCP service portfolio expands

## 🔍 Example: Vertex AI Memory Store Integration

### Service Overview
**Vertex AI Memory Store** - A new managed vector database service for AI applications requiring semantic search and recommendation capabilities.

**Key Components:**
- Vector database instances with configurable dimensions
- Real-time vector indexing and similarity search
- Integration with Vertex AI Workbench and Pipelines
- Multi-region replication and backup capabilities
- IAM-based access controls with custom roles

## 📋 Acceptance Criteria

### ✅ **AC1: Service Discovery & Analysis**
- [ ] **Automated Detection**: System detects when new GCP services become available in target projects
- [ ] **API Exploration**: Automatically discovers service APIs, methods, and resource types
- [ ] **Security Model Analysis**: Maps IAM permissions, roles, and access patterns
- [ ] **Data Classification**: Identifies what types of data the service stores/processes
- [ ] **Network Analysis**: Understands network exposure (private/public endpoints, VPC integration)

### ✅ **AC2: Security Assessment Framework**
- [ ] **Vulnerability Scanning**: Checks for common misconfigurations and security risks
- [ ] **Compliance Mapping**: Maps service features to compliance frameworks (SOC2, ISO27001, etc.)
- [ ] **Threat Modeling**: Identifies potential attack vectors and security concerns
- [ ] **Best Practices Generation**: Creates service-specific security recommendations
- [ ] **Risk Scoring**: Assigns risk levels based on data sensitivity and exposure

### ✅ **AC3: Database Schema Integration**
- [ ] **New Tables Creation**: Automatically creates database tables for new service resources
- [ ] **Schema Versioning**: Manages schema evolution as service APIs change
- [ ] **Data Mapping**: Maps API responses to normalized database schema
- [ ] **Relationship Modeling**: Links new service resources to existing assets
- [ ] **Indexing Strategy**: Optimizes queries for new service data

### ✅ **AC4: Agent Intelligence Enhancement**
- [ ] **Query Type Addition**: Adds new query types for service-specific analysis
- [ ] **Remediation Knowledge**: Integrates security best practices into agent instructions
- [ ] **Risk Assessment Logic**: Teaches agent to identify service-specific security issues
- [ ] **Integration Guidance**: Provides recommendations for secure service adoption
- [ ] **Monitoring Recommendations**: Suggests appropriate alerts and monitoring

### ✅ **AC5: Data Collection Pipeline**
- [ ] **API Client Integration**: Implements service API clients with proper authentication
- [ ] **Resource Discovery**: Fetches all service instances, configurations, and metadata
- [ ] **Incremental Updates**: Efficiently syncs changes without full rebuilds
- [ ] **Error Handling**: Robust error handling for API rate limits and failures
- [ ] **Data Validation**: Ensures data quality and completeness

### ✅ **AC6: Frontend Integration**
- [ ] **Service-Specific Views**: Creates dedicated UI sections for new service analysis
- [ ] **Security Dashboards**: Integrates service metrics into executive dashboard
- [ ] **Quick Query Buttons**: Adds service-specific quick queries to sidebar
- [ ] **Visualization Components**: Creates charts and graphs for service data
- [ ] **Deep-Dive Analysis**: Provides detailed drill-down capabilities

## 🛠️ Technical Implementation Plan

### **Phase 1: Service Discovery & Analysis Engine**

#### 1.1 Service Registry System
```python
# backend/services/service_registry.py
class ServiceRegistry:
    def discover_new_services(self) -> List[GCPService]:
        """Automatically detect new GCP services in target projects"""
        
    def analyze_service_apis(self, service: GCPService) -> ServiceAPIAnalysis:
        """Deep dive into service API structure and capabilities"""
        
    def assess_security_implications(self, service: GCPService) -> SecurityAssessment:
        """Evaluate security risks and requirements"""
```

#### 1.2 New Service Detection
- **API Scanning**: Use Google Cloud Discovery API to find new service endpoints
- **Service Catalog Integration**: Monitor Google Cloud Service Catalog for additions
- **Resource Type Analysis**: Identify new resource types in Asset Inventory
- **Permission Mapping**: Scan IAM for new permissions and predefined roles

#### 1.3 Security Assessment Framework
```python
# backend/services/security_assessor.py
class SecurityAssessor:
    def evaluate_service(self, service: GCPService) -> SecurityEvaluation:
        return SecurityEvaluation(
            data_classification=self._classify_data_types(service),
            network_exposure=self._analyze_network_security(service),
            iam_complexity=self._evaluate_permissions(service),
            compliance_gaps=self._check_compliance_requirements(service),
            threat_vectors=self._identify_threats(service),
            risk_score=self._calculate_risk_score(service)
        )
```

### **Phase 2: Database Schema Evolution**

#### 2.1 Dynamic Schema Management
```sql
-- Example for Vertex AI Memory Store
CREATE TABLE IF NOT EXISTS vertex_ai_memory_stores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    project_id TEXT NOT NULL,
    location TEXT NOT NULL,
    display_name TEXT,
    description TEXT,
    
    -- Service-specific fields
    vector_dimensions INTEGER,
    index_config TEXT, -- JSON
    deployed_indexes TEXT, -- JSON array
    endpoint_config TEXT, -- JSON
    
    -- Security fields
    encryption_config TEXT, -- JSON
    network_config TEXT, -- JSON (VPC, private endpoints)
    access_config TEXT, -- JSON (IAM bindings)
    
    -- Metadata
    state TEXT,
    create_time TIMESTAMP,
    update_time TIMESTAMP,
    labels TEXT, -- JSON
    data TEXT, -- Full API response JSON
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(name, project_id, location)
);

-- Service-specific security findings table
CREATE TABLE IF NOT EXISTS vertex_ai_memory_store_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_name TEXT NOT NULL,
    finding_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT,
    recommendation TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    
    FOREIGN KEY (resource_name) REFERENCES vertex_ai_memory_stores(name)
);
```

#### 2.2 Schema Migration System
```python
# backend/services/schema_manager.py
class SchemaManager:
    def create_service_tables(self, service: GCPService) -> None:
        """Dynamically create tables for new service"""
        
    def migrate_schema_version(self, service: GCPService, version: str) -> None:
        """Handle API version changes"""
        
    def validate_data_integrity(self, service: GCPService) -> ValidationResult:
        """Ensure data consistency after schema changes"""
```

### **Phase 3: Data Collection Pipeline**

#### 3.1 Service API Client
```python
# backend/api/vertex_ai_memory_store.py
from fastapi import APIRouter, HTTPException
from google.cloud import aiplatform_v1
from typing import List, Dict, Any

router = APIRouter(prefix="/api/v1/vertex-ai-memory-store", tags=["Vertex AI Memory Store"])

class VertexAIMemoryStoreAnalyzer:
    def __init__(self, project_id: str):
        self.client = aiplatform_v1.IndexServiceClient()
        self.project_id = project_id
    
    def discover_instances(self) -> List[Dict[str, Any]]:
        """Discover all Memory Store instances"""
        
    def analyze_security_config(self, instance: Dict) -> SecurityAnalysis:
        """Analyze security configuration of instance"""
        
    def check_compliance(self, instance: Dict) -> ComplianceReport:
        """Check compliance with security standards"""

@router.get("/discover")
async def discover_memory_stores(project_id: str = None):
    """Discover Vertex AI Memory Store instances"""
    
@router.get("/security-analysis/{instance_name}")
async def analyze_security(instance_name: str):
    """Get detailed security analysis for specific instance"""
    
@router.get("/compliance-report")
async def compliance_report():
    """Generate compliance report for all Memory Store instances"""
```

#### 3.2 Data Fetcher Integration
```python
# backend/services/data_fetcher.py (enhanced)
class DataFetcher:
    def fetch_vertex_ai_memory_stores(self) -> Dict[str, Any]:
        """Fetch Vertex AI Memory Store data"""
        try:
            instances = []
            security_findings = []
            
            # Discover instances
            for location in self.locations:
                instances.extend(self._fetch_memory_store_instances(location))
            
            # Analyze security for each instance
            for instance in instances:
                findings = self._analyze_instance_security(instance)
                security_findings.extend(findings)
            
            # Store in database
            self._store_memory_store_data(instances, security_findings)
            
            return {
                "count": len(instances),
                "security_findings": len(security_findings),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Failed to fetch Vertex AI Memory Store data: {e}")
            return {"count": 0, "error": str(e)}
```

### **Phase 4: Agent Intelligence Enhancement**

#### 4.1 New Query Types
```python
# agents/gcp_security/sqlite_tool.py (enhanced)
def query_security_data(query_type: str, parameters: Optional[str] = None) -> str:
    # Add new query types
    if query_type == 'vertex_ai_memory_stores':
        return _query_vertex_ai_memory_stores(cursor, params)
    elif query_type == 'memory_store_security':
        return _query_memory_store_security(cursor, params)
    elif query_type == 'ai_service_compliance':
        return _query_ai_service_compliance(cursor, params)

def _query_vertex_ai_memory_stores(cursor, params: Dict) -> str:
    """Query Vertex AI Memory Store instances"""
    instance_name = params.get('instance_name', '')
    
    if instance_name:
        cursor.execute("""
            SELECT * FROM vertex_ai_memory_stores
            WHERE name = ?
        """, (instance_name,))
        result = cursor.fetchone()
        
        if not result:
            return f"Memory Store instance {instance_name} not found."
        
        output = f"🧠 Vertex AI Memory Store: {instance_name}\n\n"
        output += f"Project: {result['project_id']}\n"
        output += f"Location: {result['location']}\n"
        output += f"Vector Dimensions: {result['vector_dimensions']}\n"
        output += f"State: {result['state']}\n"
        
        # Security analysis
        if result['encryption_config']:
            encryption = json.loads(result['encryption_config'])
            output += f"Encryption: {encryption.get('type', 'Unknown')}\n"
        
        if result['network_config']:
            network = json.loads(result['network_config'])
            output += f"Network: {network.get('network_type', 'Unknown')}\n"
        
        return output
    else:
        # List all instances
        cursor.execute("""
            SELECT name, project_id, location, state, vector_dimensions
            FROM vertex_ai_memory_stores
            ORDER BY name
        """)
        results = cursor.fetchall()
        
        if not results:
            return "No Vertex AI Memory Store instances found."
        
        output = f"🧠 Vertex AI Memory Store Instances ({len(results)} found):\n\n"
        for instance in results:
            output += f"• {instance['name']} ({instance['location']})\n"
            output += f"  Project: {instance['project_id']}\n"
            output += f"  Dimensions: {instance['vector_dimensions']}\n"
            output += f"  State: {instance['state']}\n\n"
        
        return output
```

#### 4.2 Agent Instructions Enhancement
```python
# agents/gcp_security/vertex_sqlite_agent.py (enhanced)
instruction = """
# Updated query types (add to existing list):

23. **vertex_ai_memory_stores** - List and analyze Vertex AI Memory Store instances
    - Optional parameters: {"instance_name": "my-memory-store"}

24. **memory_store_security** - Security analysis for Memory Store instances
    - Optional parameters: {"instance_name": "my-memory-store"}, {"check_type": "encryption"}

25. **ai_service_compliance** - Compliance analysis for AI/ML services
    - Optional parameters: {"service": "vertex_ai"}, {"framework": "sox"}

## Remediation Guide for AI/ML Services:

**For Vertex AI Memory Store (vertex_ai_memory_stores table):**
- **Unencrypted Instances**: Enable customer-managed encryption keys (CMEK) for sensitive vector data
- **Public Endpoints**: Use private service connect or VPC peering for internal access
- **Weak Access Controls**: Implement fine-grained IAM roles (aiplatform.admin vs aiplatform.user)
- **No Monitoring**: Enable audit logging for vector operations and access patterns
- **Data Residency**: Ensure vector storage complies with data locality requirements
- **Vector Security**: Implement access patterns that don't leak sensitive embedding information
- **Integration Security**: Secure connections to Vertex AI Workbench and Pipelines
- **Backup Strategy**: Configure automated backups for business-critical vector indexes
- **Network Isolation**: Use VPC Service Controls to isolate AI workloads
- **Embedding Privacy**: Ensure training data privacy in vector representations

**AI/ML Service Security Framework:**
1. **Data Classification**: Classify training data and model outputs by sensitivity
2. **Model Security**: Protect against model theft, adversarial attacks, and data poisoning  
3. **Access Governance**: Implement least-privilege access for AI resources
4. **Audit & Monitoring**: Log all AI operations for compliance and security analysis
5. **Privacy Controls**: Implement differential privacy and federated learning where applicable
"""
```

### **Phase 5: Frontend Integration**

#### 5.1 Service-Specific UI Components
```python
# frontend/unified_streaming_client.py (enhanced)
def display_ai_services_dashboard():
    """Display AI/ML services security dashboard"""
    st.header("🧠 AI/ML Services Security")
    st.caption("Vertex AI, Memory Store, and other AI service security analysis")
    
    # AI Services Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        vertex_instances = get_ai_service_count("vertex_ai_memory_stores")
        st.metric(
            "Memory Store Instances",
            vertex_instances,
            help="Total Vertex AI Memory Store instances"
        )
    
    with col2:
        workbench_instances = get_ai_service_count("vertex_ai_workbench")
        st.metric(
            "Workbench Instances", 
            workbench_instances,
            help="Vertex AI Workbench instances"
        )
    
    with col3:
        ai_findings = get_ai_security_findings()
        st.metric(
            "AI Security Issues",
            ai_findings,
            delta="Critical items need attention" if ai_findings > 0 else "All secure"
        )
    
    with col4:
        compliance_score = calculate_ai_compliance_score()
        st.metric(
            "AI Compliance",
            f"{compliance_score}%",
            delta="Meeting standards" if compliance_score > 80 else "Needs improvement"
        )
    
    # Memory Store Analysis
    st.subheader("🔍 Memory Store Security Analysis")
    
    memory_stores = query_memory_stores()
    if memory_stores:
        for store in memory_stores:
            with st.expander(f"📊 {store['name']} ({store['location']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Configuration:**")
                    st.write(f"• Dimensions: {store['vector_dimensions']}")
                    st.write(f"• State: {store['state']}")
                    st.write(f"• Project: {store['project_id']}")
                
                with col2:
                    st.write("**Security Status:**")
                    encryption_status = analyze_encryption(store)
                    network_status = analyze_network_security(store)
                    
                    if encryption_status == "CMEK":
                        st.success("✅ Customer-managed encryption enabled")
                    else:
                        st.warning("⚠️ Using Google-managed keys only")
                    
                    if network_status == "PRIVATE":
                        st.success("✅ Private network access")
                    else:
                        st.error("🔴 Public network exposure detected")

# Add to main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Security Chat", 
    "📧 MSA Analyzer", 
    "🧠 AI Services",  # New tab
    "📊 Deep Analytics"
])

with tab3:
    display_ai_services_dashboard()
```

#### 5.2 Quick Query Integration
```python
# Add to quick_queries list
quick_queries.extend([
    "List all Vertex AI Memory Store instances",
    "Check Memory Store security configurations",
    "Show AI service compliance status",
    "What AI services need encryption improvements?",
    "Analyze Vertex AI Workbench security"
])
```

### **Phase 6: Automated Security Rules**

#### 6.1 Security Rule Engine
```python
# backend/services/security_rules.py
class AIServiceSecurityRules:
    def evaluate_memory_store(self, instance: Dict) -> List[SecurityFinding]:
        findings = []
        
        # Rule: Memory Store instances should use CMEK
        if not self._has_cmek_encryption(instance):
            findings.append(SecurityFinding(
                resource_name=instance['name'],
                finding_type="ENCRYPTION_WEAKNESS",
                severity="HIGH", 
                description="Memory Store instance not using customer-managed encryption",
                recommendation="Enable CMEK encryption for sensitive vector data"
            ))
        
        # Rule: Memory Store should not have public endpoints
        if self._has_public_endpoint(instance):
            findings.append(SecurityFinding(
                resource_name=instance['name'],
                finding_type="NETWORK_EXPOSURE",
                severity="CRITICAL",
                description="Memory Store instance accessible from public internet",
                recommendation="Configure private service connect or VPC peering"
            ))
        
        # Rule: Vector dimensions should be reasonable
        dimensions = instance.get('vector_dimensions', 0)
        if dimensions > 2048:
            findings.append(SecurityFinding(
                resource_name=instance['name'],
                finding_type="CONFIGURATION_RISK",
                severity="MEDIUM",
                description=f"Very high vector dimensions ({dimensions}) may impact performance",
                recommendation="Consider dimensionality reduction or verify requirements"
            ))
        
        return findings
```

## 🧪 Testing Strategy

### **Unit Tests**
- Service discovery functionality
- Database schema creation and migration
- API client data fetching
- Security rule evaluation
- Agent query functions

### **Integration Tests**
- End-to-end service evaluation workflow
- Database integration with real service data
- Frontend UI component rendering
- Agent intelligence with new service queries

### **Manual Testing Scenarios**
1. **New Service Detection**: Verify system detects when Vertex AI Memory Store is enabled
2. **Security Assessment**: Confirm security rules identify common misconfigurations
3. **Data Collection**: Validate complete data fetching from Memory Store API
4. **Agent Queries**: Test all new query types return accurate, helpful information
5. **Frontend Integration**: Ensure UI displays new service data correctly

## 📈 Success Metrics

### **Quantitative Metrics**
- **Detection Speed**: New services detected within 24 hours of enablement
- **Coverage Completeness**: 100% of enabled GCP services monitored
- **Security Finding Accuracy**: <5% false positive rate on security assessments
- **Query Response Time**: Agent responds to service queries in <3 seconds
- **Data Freshness**: Service data updated at least every 30 minutes

### **Qualitative Metrics**
- **User Satisfaction**: Security teams can easily evaluate new services
- **Risk Reduction**: Faster identification of security gaps in new services
- **Compliance**: Automated compliance checking for new service types
- **Knowledge Transfer**: Agent provides expert-level guidance on new services

## 🚀 Deployment Plan

### **Phase 1: Core Framework (Week 1-2)**
- Implement service discovery engine
- Create dynamic schema management
- Build security assessment framework

### **Phase 2: Vertex AI Memory Store Integration (Week 3)**
- Implement Memory Store API client
- Create database schema and data pipeline
- Add agent query capabilities

### **Phase 3: Frontend & Visualization (Week 4)**
- Build AI services dashboard
- Add Memory Store security analysis views
- Integrate quick query shortcuts

### **Phase 4: Testing & Validation (Week 5)**
- Comprehensive testing of all components
- Security rule validation
- Performance optimization

### **Phase 5: Documentation & Training (Week 6)**
- Update user documentation
- Create developer integration guides
- Conduct team training sessions

## 🔄 Future Enhancements

### **Short-term (Next Quarter)**
- **Additional AI Services**: Vertex AI Pipelines, AutoML, AI Platform
- **Advanced Security Rules**: ML-based anomaly detection for AI workloads
- **Compliance Automation**: Automatic compliance reporting for AI services

### **Long-term (Next 6 Months)**
- **Service Dependency Mapping**: Understand relationships between AI services
- **Cost Security Analysis**: Identify cost anomalies that may indicate security issues
- **Predictive Risk Assessment**: ML models to predict security risks in new services

## 🏷️ Labels
`enhancement` `security` `ai-ml` `vertex-ai` `automation` `P1` `epic:service-discovery`

---

**Stakeholders:**
- **Product Owner:** Security Architecture Team
- **Technical Lead:** Platform Engineering
- **SME:** Cloud Security Engineer
- **QA Lead:** Security Testing Team

**Dependencies:**
- Google Cloud Discovery API access
- Vertex AI API enablement in target projects
- Database migration framework
- Agent instruction framework