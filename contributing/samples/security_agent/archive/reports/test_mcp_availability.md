# MCP Server Availability Test

Copy and run these commands in Claude to test availability:

```javascript
// Test exa server
mcp__exa__search({test: true})
mcp__exa__find_similar({test: true})

// Test reference server
mcp__reference__search({test: true})
mcp__reference__get_doc({test: true})

// Test filesystem server
mcp__filesystem__read_file({test: true})
mcp__filesystem__write_file({test: true})

// Test git server
mcp__git__status({test: true})
mcp__git__diff({test: true})

// Test fetch server
mcp__fetch__get({test: true})
mcp__fetch__post({test: true})

// Test database server
mcp__database__query({test: true})
mcp__database__insert({test: true})

// Test slack server
mcp__slack__send_message({test: true})
mcp__slack__list_channels({test: true})

// Test github server
mcp__github__create_issue({test: true})
mcp__github__create_pr({test: true})

// Test anthropic server
mcp__anthropic__get_context({test: true})
mcp__anthropic__set_context({test: true})

// Test memory server
mcp__memory__store({test: true})
mcp__memory__retrieve({test: true})

// Test browser server
mcp__browser__navigate({test: true})
mcp__browser__click({test: true})

// Test puppeteer server
mcp__puppeteer__launch({test: true})
mcp__puppeteer__goto({test: true})

// Test search server
mcp__search__web({test: true})
mcp__search__semantic({test: true})

// Test docs server
mcp__docs__search({test: true})
mcp__docs__get({test: true})

```

## Expected Results

- ✅ Available: Tool executes or returns parameter errors
- ❌ Not Available: Returns 'tool not found' or similar
