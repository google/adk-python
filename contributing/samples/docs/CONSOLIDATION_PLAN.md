# Documentation Consolidation Plan

## Current State
- **Root docs** (`/contributing/samples/docs/`): BMad method, up-to-date, 7 files
- **Security agent docs** (`/contributing/samples/security_agent/docs/`): Comprehensive, 45 files

## Consolidation Strategy

### 1. Primary Documentation Structure (Root Level)
```
/contributing/samples/docs/
├── README.md                           # Main project overview
├── DOCUMENTATION_INDEX.md              # Documentation map
├── prd.md                             # Product requirements
├── architecture.md                    # System architecture
├── architecture/
│   ├── coding-standards.md           # Development guidelines
│   ├── source-tree.md                # Project structure
│   ├── tech-stack.md                 # Technology choices
│   ├── adk-integration.md           # ADK-specific details (from security_agent)
│   └── api-specification.md         # API specs (from security_agent)
├── guides/
│   ├── quick-start.md               # Getting started (from security_agent)
│   ├── env-setup.md                 # Environment setup (from security_agent)
│   ├── deployment.md                # Deployment guide (from security_agent)
│   └── testing.md                   # Testing guide (from security_agent)
├── reference/
│   ├── api-reference.md            # API documentation (from security_agent)
│   ├── agent-patterns.md           # Agent delegation patterns
│   └── radar-methodology.md        # RADAR framework details
└── legacy/                          # Archive of old/superseded docs
```

### 2. Files to Migrate from security_agent/docs

#### High Priority (Unique valuable content)
- `ADK_INTEGRATION.md` → `/docs/architecture/adk-integration.md`
- `ADK_DELEGATION_PATTERN_IMPLEMENTATION.md` → `/docs/reference/agent-patterns.md`
- `API_SPECIFICATION.md` → `/docs/architecture/api-specification.md`
- `API_REFERENCE.md` → `/docs/reference/api-reference.md`
- `DEPLOYMENT_GUIDE.md` → `/docs/guides/deployment.md`
- `TESTING_GUIDE.md` → `/docs/guides/testing.md`
- `guides/QUICK_START.md` → `/docs/guides/quick-start.md`
- `guides/ENV_SETUP.md` → `/docs/guides/env-setup.md`

#### Medium Priority (Reference material)
- `RADAR_FRONTEND_ARCHITECTURE.md` → `/docs/reference/radar-methodology.md`
- `RECOMMENDER_API_INTEGRATION.md` → Merge into api-specification.md
- `ASSET_INVENTORY_IMPLEMENTATION.md` → Merge into api-specification.md
- `OPERATIONS_MANUAL.md` → `/docs/guides/operations.md`

#### Low Priority (Archive/Delete)
- Duplicate architecture files (CHAT_CENTRIC_*, PROPOSED_*, etc.)
- Temporary files (`# fixes.md`, `Overall.md`)
- Implementation-specific files that are now outdated

### 3. Actions Required

1. **Create new directory structure** in root docs
2. **Copy and update** unique valuable content
3. **Merge overlapping content** into consolidated files
4. **Update all references** in code and README files
5. **Archive old documentation** for reference
6. **Delete duplicates** after verification

### 4. Benefits
- Single source of truth for documentation
- Clear BMad method organization
- Preserves valuable content from both locations
- Easier to maintain and update
- Better discoverability