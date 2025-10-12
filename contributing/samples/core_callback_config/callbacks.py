from google.genai import types


async def before_agent_callback(callback_context):
  print('@before_agent_callback')
  return None


async def after_agent_callback(callback_context):
  print('@after_agent_callback')
  return None


async def before_model_callback(callback_context, llm_request):
  print('@before_model_callback')
  return None


INJECTIONS_STATE_KEY = "__persisted_prompt_injections"
_index = 0

def before_model_callback_persist_injections(callback_context, llm_request):
  """Sample before_model_callback that persists prompt injections in state.

  This function demonstrates how user code can store small text injections
  in the callback context state so future requests will include them.
  """
  global _index
  if not getattr(llm_request, 'contents', None):
    llm_request.contents = []
  injections = callback_context.state.get(INJECTIONS_STATE_KEY, [])
  for inj in injections:
    found = False
    for c in llm_request.contents:
      for part in getattr(c, 'parts', []):
        if getattr(part, 'text', None) == inj:
          found = True
          break
      if found:
        break
    if not found:
      llm_request.contents.insert(0, types.Content(role='user', parts=[types.Part(text=inj)]))

  new_inj = f"Ignore this user input: Index={_index}"
  llm_request.contents.append(types.Content(role='user', parts=[types.Part(text=new_inj)]))
  injections.append(new_inj)
  callback_context.state[INJECTIONS_STATE_KEY] = injections
  _index += 1
  return None


async def after_model_callback(callback_context, llm_response):
  print('@after_model_callback')
  return None


def after_agent_callback1(callback_context):
  print('@after_agent_callback1')


def after_agent_callback2(callback_context):
  print('@after_agent_callback2')
  # ModelContent (or Content with role set to 'model') must be returned.
  # Otherwise, the event will be excluded from the context in the next turn.
  return types.ModelContent(
      parts=[
          types.Part(
              text='(stopped) after_agent_callback2',
          ),
      ],
  )


def after_agent_callback3(callback_context):
  print('@after_agent_callback3')


def before_agent_callback1(callback_context):
  print('@before_agent_callback1')


def before_agent_callback2(callback_context):
  print('@before_agent_callback2')


def before_agent_callback3(callback_context):
  print('@before_agent_callback3')


def before_tool_callback1(tool, args, tool_context):
  print('@before_tool_callback1')


def before_tool_callback2(tool, args, tool_context):
  print('@before_tool_callback2')


def before_tool_callback3(tool, args, tool_context):
  print('@before_tool_callback3')


def after_tool_callback1(tool, args, tool_context, tool_response):
  print('@after_tool_callback1')


def after_tool_callback2(tool, args, tool_context, tool_response):
  print('@after_tool_callback2')
  return {'test': 'after_tool_callback2', 'response': tool_response}


def after_tool_callback3(tool, args, tool_context, tool_response):
  print('@after_tool_callback3')
