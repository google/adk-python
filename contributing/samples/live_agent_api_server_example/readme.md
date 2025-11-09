# What's this?

This is a sample that shows how to start the ADK api server, and how to connect
your agents in a live(bidi-streaming) way. It works text and audio input, and
the response is always audio.

## Prerequisite

- Make sure you go through https://google.github.io/adk-docs/streaming/

## Instruction for this sample

- The audio libraries we used here doesn't have noise cancellation. So the noise
 may feed back to the model. You can use headset to avoid this or tune down
  voice volume, or implement your own noise cancellation logic.
- Please ensure you grant the right mic/sound device permission to the terminal
 that runs the script. Sometimes, terminal inside VSCode etc doesn't really work
  well. So try native terminals if you have permission issue.
- start api server first for your agent folder. For example, my agents are
 located in contributing/samples. So I will run
  `adk api_server contributing/samples/`. Keep this running.
- then in a separate window, run `python3 live_agent_example.py`

## Misc

- Provide a few pre-recorded audio files for testing.