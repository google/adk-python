# GitHub MCP Agent

This is an agent that uses the GitHub MCP tool to call GitHub’s API. It demonstrates how to pass in your Personal Access Token and interact with repositories, issues, and pull requests.

Follow the instructions below to use it:

* Follow the installation instructions in the GitHub MCP Server README to set up and run the server via Docker:  
  https://github.com/github/github-mcp-server

* Set the environment variable `GITHUB_PERSONAL_ACCESS_TOKEN` to the token you obtained in the previous step:
  ```bash
  export GITHUB_PERSONAL_ACCESS_TOKEN=<your_personal_access_token>
  ```

* Run the agent in the ADK Web UI or via CLI:
  ```bash
  python agent.py
  ```


* Sample queries using GitHub MCP capabilities:

  - `List all my repositories for the authenticated user`  
  - `Search for issues in "octocat/Hello-World" with state "open"`  
  - `Get the contents of "src/" in repo "your-org/your-repo"`  
  - `Create a new issue in "your-org/your-repo" titled "MCP integration" with body "Please review the setup."`  
  - `Add a comment to issue #42 in "your-org/your-repo": "Looks good to me!"`  
  - `List pull requests in "your-org/your-repo" sorted by "created"`  
  - `Get code scanning alerts for "your-org/your-repo"`  
  - `Search code in "your-org/your-repo" for query "def initialize"`  
