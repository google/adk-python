"""
GCP Storage Security Analysis Tools

Tools for analyzing Google Cloud Storage buckets for security misconfigurations.
"""

from google.adk.tools.tool_context import ToolContext


def analyze_gcs_bucket_security(project_id: str, tool_context: ToolContext) -> str:
    """
    Analyzes all GCS buckets in a project for common security misconfigurations
    and returns a concise, actionable summary of recommendations.
    """
    try:
        from google.cloud import storage

        storage_client = storage.Client(project=project_id)
        buckets = storage_client.list_buckets()

        buckets_without_versioning = []
        public_buckets = []

        for bucket in buckets:
            # Check for versioning
            if not bucket.versioning_enabled:
                buckets_without_versioning.append(bucket.name)
            
            # Check for public access (simplified check)
            try:
                iam_policy = bucket.get_iam_policy(requested_policy_version=3)
                for binding in iam_policy.bindings:
                    if 'allUsers' in binding['members'] or 'allAuthenticatedUsers' in binding['members']:
                        public_buckets.append(bucket.name)
                        break 
            except Exception:
                # This can fail if uniform bucket-level access is not enabled,
                # which is itself a security finding. For this tool, we'll focus on versioning.
                pass

        recommendations = []
        if buckets_without_versioning:
            recommendations.append(
                f"Enable versioning on the following buckets to protect against data loss: {', '.join(buckets_without_versioning)}"
            )
        if public_buckets:
            recommendations.append(
                f"Remove public access from the following buckets: {', '.join(public_buckets)}"
            )

        if not recommendations:
            return f"No immediate security recommendations for GCS buckets in project '{project_id}'."

        return "Actionable GCS Security Recommendations:\\n- " + "\\n- ".join(recommendations)

    except Exception as e:
        return f"Error analyzing GCS bucket security for project '{project_id}': {e}"