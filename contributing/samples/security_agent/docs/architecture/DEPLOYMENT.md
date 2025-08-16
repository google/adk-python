# Deployment Process

This document outlines the process for deploying the ADK Security Agent to a production environment.

## Prerequisites

Before you begin, ensure that you have the following prerequisites in place:

* A Google Cloud Platform (GCP) project with the following APIs enabled:
    * Cloud Run API
    * Cloud Build API
    * Artifact Registry API
* A service account with the following roles:
    * Cloud Run Admin
    * Cloud Build Editor
    * Artifact Registry Writer
* A Dockerfile that defines the container image for the application.
* A `cloudbuild.yaml` file that defines the steps for building and deploying the application.

## Steps

1. **Configure the environment.**

    * Set the `gcloud` project to the correct project ID.
    * Configure the Docker credential helper to use the `gcloud` command-line tool.

2. **Build the container image.**

    * Use the `gcloud builds submit` command to build the container image and store it in the Artifact Registry.

3. **Deploy the application.**

    * Use the `gcloud run deploy` command to deploy the container image to Cloud Run.

## Example

The following is an example of a `cloudbuild.yaml` file that can be used to build and deploy the application:

```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: [ 'build', '-t', 'gcr.io/$PROJECT_ID/adk-security-agent', '.' ]
- name: 'gcr.io/cloud-builders/docker'
  args: [ 'push', 'gcr.io/$PROJECT_ID/adk-security-agent' ]
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: gcloud
  args:
  - 'run'
  - 'deploy'
  - 'adk-security-agent'
  - '--image'
  - 'gcr.io/$PROJECT_ID/adk-security-agent'
  - '--region'
  - 'us-central1'
  - '--platform'
  - 'managed'
  - '--allow-unauthenticated'