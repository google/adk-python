### **TODO List: Application Encapsulation and Reorganization Plan**

**Goal:** Refactor the application to be more modular, consistent, and easier for new developers to understand and extend.

#### **Phase 1: Backend Refactoring (Domain-Driven Structure)**

The current structure separates code by *type* (e.g., all services in `services/`, all APIs in `api/`). We will refactor this to group code by *feature* or *domain*.

*   [x] **1.1: Create Feature-Based Directories**
    *   For each major feature (e.g., `compliance`, `iam`, `recommendations`), create a new directory inside `contributing/samples/security_agent/backend/`.
    *   Move the relevant API endpoints from `api/` and services from `services/` into these new directories.
    *   **Example:**
        *   `api/compliance.py` -> `backend/compliance/api.py`
        *   `services/compliance_service.py` -> `backend/compliance/service.py`
        *   `services/iam_policy_analyzer.py` -> `backend/iam/service.py`

*   [ ] **1.2: Relocate Data Models**
    *   Move Pydantic models from the central `models/api_models.py` into `models.py` files within their corresponding feature directories.
    *   **Example:** `compliance`-related models go into `backend/compliance/models.py`.

*   [ ] **1.3: Update `main.py` for New Structure**
    *   Modify the router `include` statements in `main.py` to point to the new API endpoint locations.
    *   Update service instantiations to import from their new, feature-specific locations.

*   [ ] **1.4: Standardize Naming Conventions**
    *   Ensure all files within a feature directory follow a consistent naming scheme (e.g., `api.py`, `service.py`, `models.py`).

#### **Phase 2: Frontend Refactoring (Component-Based Architecture)**

The frontend is currently a single, large Streamlit application. We will break it down into smaller, more manageable components.

*   [ ] **2.1: Create a Component Library**
    *   Create a `frontend/components/` directory.
    *   Break down the monolithic `enhanced_security_agent_app.py` into smaller, reusable components. Each component will be a Python file containing a function that renders a specific part of the UI.
    *   **Example:**
        *   `frontend/components/recommendations_view.py`
        *   `frontend/components/iam_analyzer_view.py`

*   [ ] **2.2: Implement an API Client**
    *   Create a `frontend/api_client.py` module.
    *   This module will contain functions that make all the necessary API calls to the backend (e.g., `get_recommendations()`, `get_iam_analysis()`).
    *   Refactor the frontend components to use this API client instead of making direct `requests` calls. This decouples the frontend from the backend's implementation details.

*   [ ] **2.3: Refactor the Main Frontend App**
    *   The `enhanced_security_agent_app.py` file will become a lightweight "container" that imports and assembles the different UI components from the `components/` directory.

#### **Phase 3: Documentation and Finalization**

*   [ ] **3.1: Update `README.md`**
    *   Update the project structure documentation in the `README.md` to reflect the new, modular organization.
    *   Add a new "How to Extend the Application" section with a step-by-step guide for adding a new feature.

*   [ ] **3.2: Add and Improve Docstrings**
    *   Ensure all new modules and functions have clear, concise docstrings explaining their purpose and usage.
