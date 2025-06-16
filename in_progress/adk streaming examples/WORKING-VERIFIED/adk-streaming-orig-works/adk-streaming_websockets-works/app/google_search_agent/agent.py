from google.adk.agents import Agent

root_agent = Agent(
    name="greeting_agent",  #google_search_agent  
    # https://ai.google.dev/gemini-api/docs/models
    model= "gemini-2.0-flash-live-preview-04-09",  #"gemini-2.0-flash-exp", 
    description="Greeting agent to talk to user in an engaging helpful and entertaining way",
    instruction="""You are an interesting and engaging conversationalist and tallented story teller. 
    You will ask the users name and instegate a conversation with 
    user.""",
)
