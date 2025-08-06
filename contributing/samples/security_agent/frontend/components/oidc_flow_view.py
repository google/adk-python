"""OIDC authentication flow demonstration component."""

import streamlit as st
import uuid
from urllib.parse import urlencode
from typing import Dict, Any


# OIDC Demo Configuration
OIDC_DEMO_CONFIG = {
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "client_id": "demo-client-id",
    "redirect_uri": "http://localhost:8501/callback",
    "scope": "openid email profile",
    "response_type": "code"
}


def render_oidc_flow_view():
    """Render the OIDC authentication flow demonstration."""
    st.header("🔐 OIDC Authentication Flow Demo")
    st.write("Interactive demonstration of OpenID Connect authentication flows.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Authorization Code Flow",
        "📊 Flow Diagrams", 
        "⚙️ Configuration",
        "🧪 Test Scenarios"
    ])
    
    with tab1:
        render_authorization_code_flow()
    
    with tab2:
        render_flow_diagrams()
    
    with tab3:
        render_oidc_configuration()
    
    with tab4:
        render_test_scenarios()


def render_authorization_code_flow():
    """Render the authorization code flow demonstration."""
    st.subheader("🔄 OAuth 2.0 Authorization Code Flow with PKCE")
    
    # Initialize OIDC session state
    if 'oidc_state' not in st.session_state:
        st.session_state.oidc_state = None
    if 'oidc_code' not in st.session_state:
        st.session_state.oidc_code = None
    if 'oidc_tokens' not in st.session_state:
        st.session_state.oidc_tokens = None
    if 'pkce_verifier' not in st.session_state:
        st.session_state.pkce_verifier = None
    
    # Step-by-step flow
    st.markdown("### Step 1: Initialize Authentication")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Status:**")
        if st.session_state.oidc_tokens:
            st.success("✅ Authenticated")
        elif st.session_state.oidc_code:
            st.info("🔄 Authorization code received")
        elif st.session_state.oidc_state:
            st.warning("⏳ Waiting for authorization")
        else:
            st.error("❌ Not authenticated")
    
    with col2:
        st.markdown("**Actions:**")
        if not st.session_state.oidc_state:
            if st.button("🚀 Start OIDC Flow", type="primary"):
                start_oidc_flow()
        elif st.session_state.oidc_state and not st.session_state.oidc_code:
            st.info("👆 Click the authorization URL above to continue")
            if st.button("🔄 Reset Flow"):
                reset_oidc_flow()
        elif st.session_state.oidc_code and not st.session_state.oidc_tokens:
            if st.button("🔑 Exchange Code for Tokens"):
                exchange_code_for_tokens()
        else:
            if st.button("🔄 Reset Flow"):
                reset_oidc_flow()
    
    # Display current flow state
    if st.session_state.oidc_state:
        st.markdown("### Step 2: Authorization Request")
        
        # Generate authorization URL
        auth_params = {
            "client_id": OIDC_DEMO_CONFIG["client_id"],
            "redirect_uri": OIDC_DEMO_CONFIG["redirect_uri"],
            "response_type": OIDC_DEMO_CONFIG["response_type"],
            "scope": OIDC_DEMO_CONFIG["scope"],
            "state": st.session_state.oidc_state,
            "code_challenge": generate_code_challenge(),
            "code_challenge_method": "S256"
        }
        
        auth_url = f"{OIDC_DEMO_CONFIG['authorization_endpoint']}?{urlencode(auth_params)}"
        
        st.code(auth_url, language="text")
        st.markdown(f"[🔗 Click here to authorize (Demo)]({auth_url})")
        
        # Show PKCE details
        with st.expander("🔒 PKCE Details"):
            st.markdown("**Code Verifier:**")
            st.code(st.session_state.pkce_verifier or "Not generated")
            st.markdown("**Code Challenge:**")
            st.code(generate_code_challenge())
            st.markdown("**Challenge Method:** S256 (SHA256)")
    
    if st.session_state.oidc_code:
        st.markdown("### Step 3: Authorization Code Received")
        
        st.success(f"✅ Authorization code: `{st.session_state.oidc_code[:20]}...`")
        
        # Token exchange request
        with st.expander("🔧 Token Exchange Request"):
            token_request = {
                "grant_type": "authorization_code",
                "client_id": OIDC_DEMO_CONFIG["client_id"],
                "code": st.session_state.oidc_code,
                "redirect_uri": OIDC_DEMO_CONFIG["redirect_uri"],
                "code_verifier": st.session_state.pkce_verifier
            }
            
            st.json(token_request)
    
    if st.session_state.oidc_tokens:
        st.markdown("### Step 4: Tokens Received")
        
        # Display tokens (mock)
        tokens = st.session_state.oidc_tokens
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Access Token:**")
            st.code(f"{tokens.get('access_token', '')[:50]}...")
            
            st.markdown("**Token Type:**")
            st.code(tokens.get('token_type', 'Bearer'))
        
        with col2:
            st.markdown("**ID Token:**")
            st.code(f"{tokens.get('id_token', '')[:50]}...")
            
            st.markdown("**Expires In:**")
            st.code(f"{tokens.get('expires_in', 3600)} seconds")
        
        # Decode ID token (mock)
        if tokens.get('id_token'):
            st.markdown("**Decoded ID Token Claims:**")
            id_token_claims = {
                "iss": "https://accounts.google.com",
                "sub": "1234567890",
                "aud": OIDC_DEMO_CONFIG["client_id"],
                "exp": 1234567890,
                "iat": 1234567890,
                "email": "user@example.com",
                "email_verified": True,
                "name": "Demo User"
            }
            st.json(id_token_claims)


def render_flow_diagrams():
    """Render OIDC flow diagrams."""
    st.subheader("📊 OIDC Authentication Flow Diagrams")
    
    # Create tabs for different flows
    diagram_tab1, diagram_tab2, diagram_tab3 = st.tabs([
        "🔄 Authorization Code Flow",
        "🏢 Client Credentials", 
        "🔧 Token Exchange"
    ])
    
    with diagram_tab1:
        st.markdown("**OAuth 2.0 Authorization Code Flow with PKCE**")
        
        mermaid_auth_code = """
        ```mermaid
        sequenceDiagram
            participant User as 👤 User
            participant App as 🖥️ Security Agent
            participant AuthServer as 🔐 Auth Server<br/>(Google/Entra ID)
            participant API as 🛡️ GCP APIs
            
            Note over User,API: 1. Authorization Request
            User->>App: Access Security Agent
            App->>App: Generate PKCE verifier & challenge
            App->>User: Redirect to Auth Server
            Note right of App: /oauth2/authorize?<br/>client_id=xxx&<br/>redirect_uri=xxx&<br/>code_challenge=xxx
            
            Note over User,API: 2. User Authentication
            User->>AuthServer: Login & Consent
            AuthServer->>AuthServer: Validate credentials
            AuthServer->>User: Redirect with auth code
            Note left of AuthServer: /callback?code=abc123&state=xyz
            
            Note over User,API: 3. Token Exchange
            User->>App: Authorization code
            App->>AuthServer: Exchange code for tokens
            Note right of App: POST /oauth2/token<br/>code=abc123&<br/>code_verifier=xxx
            AuthServer->>App: Access & ID tokens
            
            Note over User,API: 4. API Access
            App->>API: API call with access token
            Note right of App: Authorization: Bearer token
            API->>API: Validate token
            API->>App: Protected resource
            App->>User: Security evaluation results
        ```
        """
        st.markdown(mermaid_auth_code)
    
    with diagram_tab2:
        st.markdown("**OAuth 2.0 Client Credentials Flow (Service-to-Service)**")
        
        mermaid_client_creds = """
        ```mermaid
        sequenceDiagram
            participant Agent as 🤖 Security Agent
            participant AuthServer as 🔐 Auth Server<br/>(Google/Entra ID)
            participant API as 🛡️ GCP APIs
            
            Note over Agent,API: Service-to-Service Authentication
            Agent->>AuthServer: POST /oauth2/token<br/>grant_type=client_credentials<br/>client_id=xxx<br/>client_secret=xxx<br/>scope=api.read
            
            AuthServer->>AuthServer: Validate client credentials
            AuthServer->>Agent: Access token
            
            Note over Agent,API: API Access
            Agent->>API: GET /api/resource<br/>Authorization: Bearer token
            API->>API: Validate token & scopes
            API->>Agent: Protected resource data
        ```
        """
        st.markdown(mermaid_client_creds)
    
    with diagram_tab3:
        st.markdown("**OAuth 2.0 Token Exchange Flow**")
        
        mermaid_token_exchange = """
        ```mermaid
        sequenceDiagram
            participant Client as 📱 Client App
            participant TokenService as 🔄 Token Service
            participant AuthServer as 🔐 Auth Server
            participant TargetAPI as 🎯 Target API
            
            Note over Client,TargetAPI: Token Exchange for Different Audience
            Client->>TokenService: Request token exchange<br/>subject_token=original_token<br/>audience=target_api
            
            TokenService->>AuthServer: Validate original token
            AuthServer->>TokenService: Token validation response
            
            TokenService->>AuthServer: Request new token<br/>for target audience
            AuthServer->>TokenService: New access token
            
            TokenService->>Client: Exchange response<br/>access_token=new_token<br/>audience=target_api
            
            Client->>TargetAPI: API call with new token
            TargetAPI->>Client: Protected resource
        ```
        """
        st.markdown(mermaid_token_exchange)


def render_oidc_configuration():
    """Render OIDC configuration interface."""
    st.subheader("⚙️ OIDC Configuration")
    
    # Current configuration
    st.markdown("**Current Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Authorization Endpoint:", 
                     value=OIDC_DEMO_CONFIG["authorization_endpoint"],
                     disabled=True)
        st.text_input("Token Endpoint:",
                     value=OIDC_DEMO_CONFIG["token_endpoint"], 
                     disabled=True)
        st.text_input("Client ID:",
                     value=OIDC_DEMO_CONFIG["client_id"],
                     disabled=True)
    
    with col2:
        st.text_input("Redirect URI:",
                     value=OIDC_DEMO_CONFIG["redirect_uri"],
                     disabled=True)
        st.text_input("Scope:",
                     value=OIDC_DEMO_CONFIG["scope"],
                     disabled=True)
        st.text_input("Response Type:",
                     value=OIDC_DEMO_CONFIG["response_type"],
                     disabled=True)
    
    # Provider configuration
    st.markdown("**Provider Configuration:**")
    
    provider_options = {
        "Google": {
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
            "issuer": "https://accounts.google.com"
        },
        "Microsoft": {
            "authorization_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            "userinfo_endpoint": "https://graph.microsoft.com/oidc/userinfo",
            "issuer": "https://login.microsoftonline.com/common/v2.0"
        },
        "Auth0": {
            "authorization_endpoint": "https://YOUR_DOMAIN.auth0.com/authorize",
            "token_endpoint": "https://YOUR_DOMAIN.auth0.com/oauth/token",
            "userinfo_endpoint": "https://YOUR_DOMAIN.auth0.com/userinfo",
            "issuer": "https://YOUR_DOMAIN.auth0.com/"
        }
    }
    
    selected_provider = st.selectbox("Identity Provider:", list(provider_options.keys()))
    
    if selected_provider:
        provider_config = provider_options[selected_provider]
        
        with st.expander(f"📋 {selected_provider} Configuration"):
            st.json(provider_config)
    
    # Security settings
    st.markdown("**Security Settings:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_pkce = st.checkbox("Enable PKCE", value=True)
        enable_state = st.checkbox("Enable State Parameter", value=True)
        enable_nonce = st.checkbox("Enable Nonce", value=True)
    
    with col2:
        token_validation = st.checkbox("Validate ID Token", value=True)
        audience_validation = st.checkbox("Validate Audience", value=True)
        issuer_validation = st.checkbox("Validate Issuer", value=True)
    
    # Token configuration
    st.markdown("**Token Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        access_token_lifetime = st.number_input("Access Token Lifetime (minutes):", 1, 1440, 60)
        refresh_token_lifetime = st.number_input("Refresh Token Lifetime (days):", 1, 365, 30)
    
    with col2:
        id_token_lifetime = st.number_input("ID Token Lifetime (minutes):", 1, 60, 15)
        token_refresh_threshold = st.number_input("Refresh Threshold (minutes):", 1, 30, 5)


def render_test_scenarios():
    """Render OIDC test scenarios."""
    st.subheader("🧪 Test Scenarios")
    
    # Test scenario selection
    scenarios = [
        {
            "name": "Happy Path Authentication",
            "description": "Complete successful OIDC flow with valid credentials",
            "steps": [
                "Start authorization flow",
                "User provides valid credentials", 
                "Authorization code is returned",
                "Tokens are exchanged successfully",
                "API calls succeed with access token"
            ]
        },
        {
            "name": "Invalid Client Credentials",
            "description": "Test behavior with invalid client ID or secret",
            "steps": [
                "Start authorization flow with invalid client_id",
                "Authorization server returns error",
                "Error is handled gracefully"
            ]
        },
        {
            "name": "Token Expiration",
            "description": "Test token refresh when access token expires",
            "steps": [
                "Complete initial authentication",
                "Wait for token expiration",
                "Use refresh token to get new access token",
                "Continue API operations"
            ]
        },
        {
            "name": "PKCE Validation",
            "description": "Test PKCE code challenge/verifier validation",
            "steps": [
                "Generate PKCE code verifier and challenge",
                "Send challenge in authorization request",
                "Send verifier in token exchange",
                "Server validates PKCE parameters"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        with st.expander(f"🧪 {scenario['name']}"):
            st.write(scenario['description'])
            
            st.markdown("**Test Steps:**")
            for j, step in enumerate(scenario['steps']):
                st.markdown(f"{j+1}. {step}")
            
            if st.button(f"🚀 Run Test", key=f"test_{i}"):
                run_test_scenario(scenario['name'])


def start_oidc_flow():
    """Start the OIDC authentication flow."""
    # Generate state and PKCE verifier
    st.session_state.oidc_state = str(uuid.uuid4())
    st.session_state.pkce_verifier = str(uuid.uuid4()).replace('-', '') + str(uuid.uuid4()).replace('-', '')
    
    st.success("✅ OIDC flow started! State and PKCE verifier generated.")
    st.rerun()


def reset_oidc_flow():
    """Reset the OIDC flow state."""
    st.session_state.oidc_state = None
    st.session_state.oidc_code = None
    st.session_state.oidc_tokens = None
    st.session_state.pkce_verifier = None
    
    st.info("🔄 OIDC flow reset.")
    st.rerun()


def exchange_code_for_tokens():
    """Simulate token exchange (demo purposes)."""
    # Mock token response
    mock_tokens = {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "id_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "1//04abc123def456...",
        "scope": "openid email profile"
    }
    
    st.session_state.oidc_tokens = mock_tokens
    st.success("✅ Tokens exchanged successfully!")
    st.rerun()


def generate_code_challenge():
    """Generate PKCE code challenge."""
    import hashlib
    import base64
    
    if not st.session_state.pkce_verifier:
        return ""
    
    # In a real implementation, this would be proper SHA256
    # For demo purposes, we'll use a simplified version
    challenge = hashlib.sha256(st.session_state.pkce_verifier.encode()).digest()
    return base64.urlsafe_b64encode(challenge).decode().rstrip('=')


def run_test_scenario(scenario_name):
    """Run a test scenario."""
    with st.spinner(f"Running test scenario: {scenario_name}..."):
        # Simulate test execution
        import time
        time.sleep(2)
        
        # Mock test results
        test_results = {
            "Happy Path Authentication": {"status": "PASSED", "details": "All steps completed successfully"},
            "Invalid Client Credentials": {"status": "PASSED", "details": "Error handled correctly"},
            "Token Expiration": {"status": "PASSED", "details": "Token refresh successful"},
            "PKCE Validation": {"status": "PASSED", "details": "PKCE validation successful"}
        }
        
        result = test_results.get(scenario_name, {"status": "UNKNOWN", "details": "Test not implemented"})
        
        if result["status"] == "PASSED":
            st.success(f"✅ Test PASSED: {result['details']}")
        else:
            st.error(f"❌ Test FAILED: {result['details']}")


def render_oidc_summary_card():
    """Render a compact OIDC summary card for the dashboard."""
    with st.container():
        st.subheader("🔐 OIDC Demo")
        
        # Check authentication status
        if st.session_state.get('oidc_tokens'):
            st.success("✅ Authenticated")
            st.text("Demo tokens available")
        else:
            st.info("🔒 Not authenticated")
            st.text("Demo flow ready")
        
        if st.button("Demo OIDC Flow", key="demo_oidc"):
            st.session_state.page = "oidc"
            st.rerun()