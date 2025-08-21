# MSA Impact Analyzer Guide

## Overview

The MSA (Monthly Service Announcement) Impact Analyzer is a powerful feature that helps security teams understand how Google Cloud service changes affect their specific environment. It uses Gemini to intelligently parse MSA emails and provides personalized impact assessments.

## Key Features

### 1. Intelligent Email Parsing
- **Gemini-Powered**: Uses Gemini 1.5 Pro to extract structured information
- **Automatic Categorization**: Identifies service, change type, and impact level
- **Date Extraction**: Captures effective dates for planning
- **Action Items**: Extracts required customer actions

### 2. Environment-Specific Impact Analysis
- **Resource Matching**: Queries your GCP inventory to find affected resources
- **Quantified Impact**: Shows exact count of affected resources
- **Service Mapping**: Maps changes to your actual infrastructure
- **Severity Assessment**: Prioritizes changes by potential disruption

### 3. Visual Impact Reporting
- **Impact Distribution Chart**: Pie chart showing critical/high/medium/low breakdown
- **Service Impact Bar Chart**: Shows which services have the most changes
- **Color-Coded Alerts**: Visual severity indicators (🔴🟠🟡🟢)
- **Expandable Details**: Full information for each detected change

## How to Use

### Step 1: Access the MSA Analyzer

1. Start the frontend application:
   ```bash
   python run_frontend.py
   ```

2. Navigate to the **MSA Analyzer** tab in the dashboard

### Step 2: Input MSA Email

You have two options:

#### Option A: Paste Your MSA Email
1. Copy the entire MSA email from your inbox
2. Paste it into the text area
3. Enter your GCP Project ID for impact analysis

#### Option B: Use Sample MSA
1. Click **"Load Sample MSA"** button
2. A pre-populated example will appear
3. Perfect for testing and demonstrations

### Step 3: Analyze Impact

1. Click **"Analyze MSA Impact"** button
2. Wait ~5-10 seconds for Gemini analysis
3. Review the comprehensive results

### Step 4: Interpret Results

#### Summary Metrics
- **Total Changes**: Number of service changes detected
- **Critical Changes**: Changes requiring immediate attention
- **Resources Affected**: Total count of your resources impacted

#### Visual Analytics
- **Impact Level Distribution**: See severity breakdown at a glance
- **Changes by Service**: Identify which services are most affected

#### Detailed Changes
Each change shows:
- Service name and change type
- Detailed description
- Effective date (when applicable)
- Required actions
- Impact level with color coding
- Affected resource types

#### Project-Specific Impact
If you provided a project ID:
- Exact resources affected per service
- Resource counts and types
- Personalized recommendations
- Sample of affected resource names

## Example MSA Analysis

### Input: BigQuery Permission Change
```
We are updating BigQuery service account permissions. 
The bigquery.tables.getData permission will be split into:
- bigquery.tables.getData.read (for reading data)
- bigquery.tables.getData.export (for exporting data)

Action Required: Review and update your IAM policies by January 15, 2025.
```

### Output:
- **Service**: BigQuery
- **Change Type**: Permission Change
- **Impact Level**: HIGH
- **Affected Resources**: 15 BigQuery datasets in your project
- **Recommendations**:
  - Review IAM policies for all service accounts
  - Update automation scripts using old permission
  - Test in development environment first
  - Create backup before effective date

## API Integration

### Backend Endpoint
```python
POST /api/v1/msa/analyze
{
    "email_content": "Full MSA email text",
    "project_id": "your-gcp-project-id"  # Optional
}
```

### Response Structure
```python
{
    "success": true,
    "extracted_changes": [
        {
            "service": "BigQuery",
            "change_type": "permission_change",
            "description": "...",
            "effective_date": "2025-01-15",
            "required_action": "...",
            "impact_level": "high",
            "affected_resources": ["datasets", "tables"]
        }
    ],
    "impact_assessments": [
        {
            "project_id": "my-project",
            "resource_type": "bigquery_datasets",
            "resource_count": 15,
            "impact_level": "high",
            "recommended_actions": [...],
            "affected_resources": [...]
        }
    ],
    "summary": {
        "total_changes": 4,
        "critical_changes": 1,
        "high_impact_changes": 2,
        "total_resources_affected": 47
    },
    "recommendations": [
        "🚨 Critical changes detected - immediate action required",
        "📊 Large number of resources affected - consider phased rollout"
    ]
}
```

## Best Practices

### 1. Regular Analysis
- Analyze MSAs as soon as they arrive (monthly)
- Keep your GCP inventory cache updated for accurate impact assessment
- Review critical and high-impact changes immediately

### 2. Team Collaboration
- Share analysis results with relevant teams
- Create tickets for required actions
- Schedule changes before effective dates

### 3. Documentation
- Document decisions made based on MSA analysis
- Track which changes were applied to your environment
- Maintain a history of MSA impacts

## Troubleshooting

### Issue: "Analysis failed"
**Solution**: Check that:
- Backend is running (`python run_backend.py`)
- Vertex AI/Gemini is properly configured
- Valid GCP credentials are set

### Issue: "No resources affected shown"
**Solution**: 
- Ensure SQLite database is populated (`python populate_sqlite.py`)
- Verify project ID matches your GCP project
- Check that data refresh has completed recently

### Issue: "Gemini parsing error"
**Solution**:
- Ensure MSA email is complete (not truncated)
- Check for special characters that might break parsing
- Try with the sample MSA first to verify setup

## Technical Architecture

```
MSA Email → Gemini 1.5 Pro → Structured Changes
                ↓
         Impact Analysis ← SQLite Database (GCP Inventory)
                ↓
         Visual Reports → Streamlit Dashboard
```

### Components
1. **Frontend** (`unified_streaming_client.py`): UI for input and visualization
2. **Backend API** (`msa_analyzer.py`): Gemini integration and analysis logic
3. **Database** (`gcp_data.db`): Cached GCP resource inventory
4. **Gemini Model**: Natural language processing for email parsing

## Future Enhancements

- [ ] Email integration for automatic MSA ingestion
- [ ] Historical MSA tracking and trends
- [ ] Automated ticket creation for required actions
- [ ] Multi-project impact analysis
- [ ] Integration with change management systems
- [ ] Support for other advisory types (CVEs, security bulletins)

## Support

For issues or questions:
- Check the main `CLAUDE.md` for architectural details
- Review `STORY-012-ADVISORY-NOTIFICATIONS.md` for requirements
- Run `python test_msa_analyzer.py` for diagnostics

---

**Built with Gemini 1.5 Pro and Google Cloud Platform**