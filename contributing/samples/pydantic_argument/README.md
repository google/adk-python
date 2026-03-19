# Pydantic Argument Sample Agent

This sample demonstrates the automatic Pydantic model conversion feature in ADK FunctionTool.

## What This Demonstrates

This sample shows two key features of the Pydantic argument conversion:

### 1. Optional Type Handling

The `create_full_user_account` function demonstrates `Optional[PydanticModel]` conversion:

Before the fix, Optional parameters required manual conversion:

```python
def create_full_user_account(
    profile: UserProfile, 
    preferences: Optional[UserPreferences] = None
) -> dict:
    # Manual conversion needed:
    if not isinstance(profile, UserProfile):
        profile = UserProfile.model_validate(profile)
    
    if preferences is not None and not isinstance(preferences, UserPreferences):
        preferences = UserPreferences.model_validate(preferences)
    
    # Your function logic here...
```

**After the fix**, Union/Optional Pydantic models are handled automatically:

```python
def create_full_user_account(
    profile: UserProfile, 
    preferences: Optional[UserPreferences] = None
) -> dict:
    # Both profile and preferences are guaranteed to be proper instances!
    # profile: UserProfile instance (converted from JSON)
    # preferences: UserPreferences instance OR None (converted from JSON or kept as None)
    return {"profile": profile.name, "theme": preferences.theme if preferences else "default"}
```

### 2. Union Type Handling

The `create_entity_profile` function demonstrates `Union[PydanticModel1, PydanticModel2]` conversion:

**Before the fix**, Union types required complex manual type checking:

```python
def create_entity_profile(entity: Union[UserProfile, CompanyProfile]) -> dict:
    # Manual conversion needed:
    if isinstance(entity, dict):
        # Try to determine which model to use and convert manually
        if 'company_name' in entity:
            entity = CompanyProfile.model_validate(entity)
        elif 'name' in entity:
            entity = UserProfile.model_validate(entity)
        else:
            raise ValueError("Cannot determine entity type")
    # Your function logic here...
```

**After the fix**, Union Pydantic models are handled automatically:

```python
def create_entity_profile(entity: Union[UserProfile, CompanyProfile]) -> dict:
    # entity is guaranteed to be either UserProfile or CompanyProfile instance!
    # The LLM sends appropriate JSON structure, and it gets converted
    # to the correct Pydantic model based on JSON schema matching
    if isinstance(entity, UserProfile):
        return {"type": "user", "name": entity.name}
    else:  # CompanyProfile
        return {"type": "company", "name": entity.company_name}
```

## How to Run

1. **Set up API credentials** (choose one):

   **Option A: Google AI API**
   ```bash
   export GOOGLE_GENAI_API_KEY="your-api-key"
   ```

   **Option B: Vertex AI (requires Google Cloud project)**
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export GOOGLE_CLOUD_LOCATION="us-central1"
   ```

2. **Run the sample**:
   ```bash
   cd contributing/samples
   python -m pydantic_argument.main
   ```

## Expected Output

The agent will be prompted to create user profiles and accounts, demonstrating automatic Pydantic model conversion.

### Test Scenarios:

1. **Full Account with Preferences (Optional Type)**:
   - **Input**: "Create an account for Alice, 25 years old, with dark theme and Spanish language preferences"
   - **Tool Called**: `create_full_user_account(profile=UserProfile(...), preferences=UserPreferences(...))`
   - **Conversion**: Two JSON dicts → `UserProfile` + `UserPreferences` instances

2. **Account with Different Preferences (Optional Type)**:
   - **Input**: "Create a user account for Bob, age 30, with light theme, French language, and notifications disabled"
   - **Tool Called**: `create_full_user_account(profile=UserProfile(...), preferences=UserPreferences(...))`
   - **Conversion**: Two JSON dicts → `UserProfile` + `UserPreferences` instances

3. **Account with Default Preferences (Optional Type)**:
   - **Input**: "Make an account for Charlie, 28 years old, but use default preferences"
   - **Tool Called**: `create_full_user_account(profile=UserProfile(...), preferences=None)`
   - **Conversion**: JSON dict → `UserProfile`, None → None (Optional handling)

4. **Company Profile Creation (Union Type)**:
   - **Input**: "Create a profile for Tech Corp company, software industry, with 150 employees"
   - **Tool Called**: `create_entity_profile(entity=CompanyProfile(...))`
   - **Conversion**: JSON dict → `CompanyProfile` instance (Union type resolution)

5. **User Profile Creation (Union Type)**:
   - **Input**: "Create an entity profile for Diana, 32 years old"
   - **Tool Called**: `create_entity_profile(entity=UserProfile(...))`
   - **Conversion**: JSON dict → `UserProfile` instance (Union type resolution)
