# Instructions

## Run Evaluation

1. Set environment variables in your terminal:

  ```shell
  export GOOGLE_GENAI_USE_VERTEXAI=FALSE
  export GOOGLE_API_KEY=<your_api_key>
  export GOOGLE_CLOUD_PROJECT=<your_bigquery_project>
  ```
1. Change to the current directory:

  ```shell
  cd third_party/py/google/adk/tests/integration/fixture/bigquery_agent/
  ```
1. Customize the evaluation dataset to the environment `GOOGLE_CLOUD_PROJECT`
   by replacing the placeholder to the real project set in your environment:

  ```shell
  sed -e "s:\${GOOGLE_CLOUD_PROJECT}:${GOOGLE_CLOUD_PROJECT}:g" simple.test.json -i
  ```
1. Run the following command as per https://google.github.io/adk-docs/evaluate/#3-adk-eval-run-evaluations-via-the-cli:

  ```shell
  adk eval . simple.test.json --config_file_path=test_config.json
  ```

  If it fails, re-run with `--print_detailed_results` flag to see more details
  on turn-by-turn evaluation.

## Generate Evaluation dataset

1. Set environment variables in your terminal:

  ```shell
  export GOOGLE_GENAI_USE_VERTEXAI=FALSE
  export GOOGLE_API_KEY=<your_api_key>
  export GOOGLE_CLOUD_PROJECT=<your_bigquery_project>
  ```
1. Set up google [application default credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc)
   on your machine.

  ```shell
  gcloud auth application-default login
  ```
1. Change to the directory containing agent folder:

  ```shell
  cd third_party/py/google/adk/tests/integration/fixture/
  ```
1. Run the following command to start the ADK web app:

  ```shell
  adk web
  ```
1. Open the ADK web UI in your browser http://127.0.0.1:8000/dev-ui/?app=bigquery_agent.
1. Create an evaluation dataset by following [these steps](https://google.github.io/adk-docs/evaluate/#1-adk-web-run-evaluations-via-the-web-ui).
   This would generate file `bigquery_agent/simple.evalset.json`.
1. Note that this evaluation data would be tied to the agent interaction in the
   `GOOGLE_CLOUD_PROJECT` set in your environment. To normalize it by replacing
   the real project set in your environment to a placeholder, let's run the
   following command:

  ```shell
  sed -e "s:${GOOGLE_CLOUD_PROJECT}:\${GOOGLE_CLOUD_PROJECT}:g"  bigquery_agent/simple.evalset.json > bigquery_agent/simple.test.json
  ```