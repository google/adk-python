# Database Migrations on Kubernetes

In Kubernetes deployments, run database migrations before application pods start
using Helm `pre-install` and `pre-upgrade` hooks. This ensures all pods see the
same schema and avoids race conditions from concurrent migration attempts.

Disable `ADK_AUTO_MIGRATE_DB` in your application pods when using this approach.

## Migration Job

Add a Job template to your Helm chart that runs `adk migrate upgrade` as a
pre-install and pre-upgrade hook:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ .Release.Name }}-db-migration
  annotations:
    "helm.sh/hook": pre-install,pre-upgrade
    "helm.sh/hook-weight": "-5"
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  ttlSecondsAfterFinished: 86400          # Auto-cleanup after 24h
  backoffLimit: 3
  activeDeadlineSeconds: 300              # Overall job timeout (5 min)
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: {{ .Values.serviceAccountName }}
      containers:
      - name: migration
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        command: ["adk", "migrate", "upgrade", "--db_url", "$(DATABASE_URL)"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {{ .Values.database.secretName }}
              key: url
        securityContext:
          runAsNonRoot: true
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          capabilities:
            drop:
              - ALL
```

Key fields:

- **`activeDeadlineSeconds`**: Overall timeout for the Job. If the migration
  hasn't finished within this window the Job is terminated. Adjust based on
  your expected migration duration.
- **`ttlSecondsAfterFinished`**: Automatically deletes completed Job resources
  after 24 hours to avoid clutter.
- **`backoffLimit`**: Number of retries before the Job is marked as failed.
- **`securityContext`**: Follows least-privilege: non-root user, no privilege
  escalation, all Linux capabilities dropped.

The `pre-install,pre-upgrade` annotations ensure the Job runs before any
application pods are created or updated. `helm.sh/hook-delete-policy:
before-hook-creation` cleans up the previous Job before creating a new one on
subsequent upgrades.

The `adk migrate upgrade` command auto-bootstraps databases that predate Alembic
support, so this Job handles both fresh deployments and upgrades from earlier ADK
versions.

## Application Configuration

In your application Deployment, disable auto-migration since the Helm hook
handles it. `false` is the default, so this is optional, but you may want to set it for explicitness:

```yaml
env:
- name: ADK_AUTO_MIGRATE_DB
  value: "false"  # default
```

## Cloud SQL Proxy

If your database is a Cloud SQL instance on GKE, add the
[Cloud SQL Auth Proxy](https://cloud.google.com/sql/docs/postgres/connect-kubernetes-engine)
as a
[native sidecar container](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/)
by defining it as an `initContainer` with `restartPolicy: Always`. Kubernetes starts it before the
migration container, keeps it running alongside, and terminates it automatically
when the migration exits:

```yaml
initContainers:
- name: cloud-sql-proxy
  image: gcr.io/cloud-sql-connectors/cloud-sql-proxy:latest
  restartPolicy: Always           # Native sidecar (K8s 1.28+)
  args:
    - "--structured-logs"
    - "--port=5432"
    - "{{ .Values.database.instanceConnectionName }}"
  securityContext:
    runAsNonRoot: true
containers:
- name: migration
  image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
  command: ["adk", "migrate", "upgrade", "--db_url",
    "postgresql://$(DB_USER):$(DB_PASS)@127.0.0.1:5432/$(DB_NAME)"]
  env:
    - name: DB_USER
      valueFrom:
        secretKeyRef:
          name: {{ .Values.database.secretName }}
          key: username
    - name: DB_PASS
      valueFrom:
        secretKeyRef:
          name: {{ .Values.database.secretName }}
          key: password
    - name: DB_NAME
      valueFrom:
        secretKeyRef:
          name: {{ .Values.database.secretName }}
          key: database
```

Refer to the
[GKE Cloud SQL connectivity documentation](https://cloud.google.com/sql/docs/postgres/connect-kubernetes-engine)
for Workload Identity and IAM setup.