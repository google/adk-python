# Page snapshot

```yaml
- generic [ref=e4]:
  - banner [ref=e5]:
    - generic [ref=e8]:
      - generic [ref=e10]:
        - img "Running..." [ref=e11]:
          - img [ref=e12]
        - button "Stop" [ref=e16] [cursor=pointer]
      - button "Deploy" [ref=e18] [cursor=pointer]:
        - generic [ref=e20] [cursor=pointer]: Deploy
      - button [ref=e22] [cursor=pointer]:
        - img [ref=e23] [cursor=pointer]
  - alert [ref=e31]:
    - generic [ref=e34]:
      - generic [ref=e35]:
        - generic [ref=e36]: FileNotFoundError
        - text: ": Agent directory not found at: /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/agents/agents/gcp_security"
      - generic [ref=e37]:
        - generic [ref=e38]: "Traceback:"
        - code [ref=e41]:
          - generic [ref=e42]: "File \"/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/agents/gcp_security/unified_streaming_client.py\", line 46, in <module> raise FileNotFoundError(f\"Agent directory not found at: {agent_dir}\")"
      - generic [ref=e43]:
        - button "Copy" [ref=e44] [cursor=pointer]
        - link "Ask Google" [ref=e45] [cursor=pointer]:
          - /url: https://www.google.com/search?q=FileNotFoundError%3A%20Agent%20directory%20not%20found%20at%3A%20%2FUsers%2Fstuartgano%2FDesktop%2FMicron%2FADK%2Fcontributing%2Fsamples%2Fsecurity_agent%2Fagents%2Fagents%2Fgcp_security
        - link "Ask ChatGPT" [ref=e46] [cursor=pointer]:
          - /url: https://chatgpt.com/?q=FileNotFoundError%3A%20Agent%20directory%20not%20found%20at%3A%20%2FUsers%2Fstuartgano%2FDesktop%2FMicron%2FADK%2Fcontributing%2Fsamples%2Fsecurity_agent%2Fagents%2Fagents%2Fgcp_security%0A%0AFile%20%22%2FUsers%2Fstuartgano%2FDesktop%2FMicron%2FADK%2Fcontributing%2Fsamples%2Fsecurity_agent%2Fagents%2Fgcp_security%2Funified_streaming_client.py%22%2C%20line%2046%2C%20in%20%3Cmodule%3E%0A%20%20%20%20raise%20FileNotFoundError(f%22Agent%20directory%20not%20found%20at%3A%20%7Bagent_dir%7D%22)
```