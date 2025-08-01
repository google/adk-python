"""OpenAPI converter for ADK compatibility."""

def create_adk_compatible_openapi(app):
    """Convert OpenAPI 3.1 to 3.0 for ADK compatibility."""
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Convert from 3.1 to 3.0.3
    openapi_schema["openapi"] = "3.0.3"
    
    # Handle nullable types (3.1 uses 'type: [string, null]', 3.0 uses 'nullable: true')
    def convert_nullable(schema_dict):
        if isinstance(schema_dict, dict):
            if "type" in schema_dict and isinstance(schema_dict["type"], list):
                types = schema_dict["type"]
                if "null" in types:
                    types.remove("null")
                    schema_dict["nullable"] = True
                    if len(types) == 1:
                        schema_dict["type"] = types[0]
                    else:
                        schema_dict["type"] = types
            
            for key, value in schema_dict.items():
                if isinstance(value, dict):
                    convert_nullable(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            convert_nullable(item)
    
    convert_nullable(openapi_schema)
    
    return openapi_schema