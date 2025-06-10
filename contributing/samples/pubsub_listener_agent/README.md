# PubSub Listener Agent Sample

## Introduction
This sample demonstrates how to implement an agent that listens to Google Cloud Pub/Sub messages and emits events within the ADK framework. The PubSubListenerAgent connects to a specified Pub/Sub subscription, processes incoming messages asynchronously, acknowledges them, and generates ADK events to be handled by other agents or components.

Note: Graceful shutdown only works on Mac and Linux

I am also experimenting with integrating this agent into a larger multi-agent workflow. While not polished enough to include here, the current approach involves a root LlmAgent that starts a ParallelAgent as one of its subagents. This parallel agent runs “listener” agents like this Pub/Sub listener. 

## Requirements

* 1. Google Cloud SDK configured with access to your Pub/Sub subscription

* 2. google-cloud-pubsub Python package installed (pip install google-cloud-pubsub)

* 3 python-dotenv package installed for environment variable loading (pip install python-dotenv) (Or replace with your environment vars management solution of choice)

## How to Use

* 1. Create a .env file in the folder with the following required variables: 

APP_NAME=your_app_name
PROJECT_ID=your_gcp_project_id
SUBSCRIPTION_NAME=your_pubsub_subscription_name

* 2. Run the agent
python pubsub_listener_agent.py

* 3. Trigger a Pub/Sub message from another terminal. Ex: 
gcloud pubsub topics publish <your-pubsub-topic> --project="<your-project-id>" --message='{ "<key1>": "<value1>",  "<key2>": "<value2>",  "timestamp": "'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'"}'

* 4. Trigger graceful shutdown with CTRL+C 


