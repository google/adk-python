This agent connects to the Jira Cloud using Google Application Integration workflow and Integrations Connector

**Instructions to connect to an agent:**

**Use Integration Connectors**

Connect your agent to enterprise applications using [Integration Connectors](https://cloud.google.com/integration-connectors/docs/overview).

**Steps:**

1. To use a connector from Integration Connectors, you need to [provision](https://console.cloud.google.com/) Application Integration in the same region as your connection by clicking on "QUICK SETUP" button.
Google Cloud Tools
![image_alt](https://github.com/karthidec/adk-python/blob/adk-samples-jira-agent/contributing/samples/jira_agent/image-application-integration.png?raw=true)

2. Go to [Connection Tool]((https://console.cloud.google.com/)) template from the template library and click on "USE TEMPLATE" button.
![image_alt](https://github.com/karthidec/adk-python/blob/adk-samples-jira-agent/contributing/samples/jira_agent/image-connection-tool.png?raw=true)

3. Fill the Integration Name as **ExecuteConnection** (It is mandatory to use this integration name only) and select the region same as the connection region. Click on "CREATE".

4. Publish the integration by using the "PUBLISH" button on the Application Integration Editor.
![image_alt](https://github.com/karthidec/adk-python/blob/adk-samples-jira-agent/contributing/samples/jira_agent/image-app-intg-editor.png?raw=true)

**References:**

https://google.github.io/adk-docs/tools/google-cloud-tools/#application-integration-tools
