import { useState, useMemo } from "react";
const CLASSES = [
  { id: "exec",  label: "EXEC",  title: "Code execution — subprocess / exec() / eval() with user-controlled input" },
  { id: "file",  label: "FILE",  title: "File read/write — open() with user-controlled paths" },
  { id: "ssrf",  label: "SSRF",  title: "SSRF / unfiltered URL fetch (urllib, requests, httpx)" },
  { id: "sql",   label: "SQL",   title: "SQL injection / NL2SQL no validation" },
  { id: "ipi",   label: "IPI",   title: "IPI attack surface — external untrusted data flows into agent context" },
  { id: "deser", label: "DESR",  title: "Unsafe deserialization — yaml.load (not safe_load), pickle.loads" },
  { id: "tmpl",  label: "TMPL",  title: "Template / prompt injection — user-controlled data interpolated into system prompts" },
  { id: "cloud", label: "CLOUD", title: "Cloud API abuse — user/LLM-controlled params to GCS, BQ, Dataform, GCR..." },
  { id: "creds", label: "CRED",  title: "Hardcoded secrets / credential leakage in source" },
];
const F="FIND", N="NONE", L="LOOK", T="TODO";
const RAW_AGENTS = [
  { name:"youtube-analyst",                    findings:"AS-01",    cells:{exec:F,file:F,ssrf:N,sql:N,ipi:F,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-01 VERIFIED: visualization_tools.py:48 exec(code, global_vars, local_vars) — __builtins__ not removed, full builtins accessible. Path traversal: line 36 os.path.join(output_dir, filename), filename LLM-controlled, no validation. IPI chain confirmed: get_video_details()/get_video_comments() → youtube_agent → visualization_agent (sub_agent line 28) → execute_visualization_code → exec()." },
  { name:"policy-as-code",                     findings:"AS-02",    cells:{exec:F,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-02 VERIFIED: simulation.py:122 exec(policy_code, safe_globals). safe_globals explicitly includes '__import__': __import__ at line 111. AST validation (lines 38-49) blocks ast.Import/ast.ImportFrom nodes but NOT __import__() call nodes (ast.Call) — bypass confirmed. Limited blast radius in simulation context." },
  { name:"machine-learning-engineering",       findings:"AS-03",    cells:{exec:F,file:F,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-03 VERIFIED: code_util.py:29 subprocess.run(['python', py_filepath], cwd=run_cwd). LLM-generated code_text written to disk at line 26 (open(output_filepath,'w')), then executed. No sandboxing. FILE confirmed: write to run_cwd path controlled via callback_context.state — LLM-influenced path." },
  { name:"plumber-data-engineering-assistant", findings:"AS-04",    cells:{exec:F,file:N,ssrf:N,sql:N,ipi:F,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-04 VERIFIED: dataproc.py:261 subprocess.run(command_str, shell=True). job_id injected via \" \".join(command) at line 256, no escaping. Dev comment at line 242: 'This function uses shell=True for simplicity, which is less secure.' IPI path: Dataproc job output → job_id reused in subsequent commands (lines 286-375)." },
  { name:"data-science",                       findings:"AS-05",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-05 VERIFIED: bigquery/agent.py:70-84 BigQueryToolset + system prompt (prompts.py:36-42) instructs LLM to 'generate initial SQL' via nl2sql_tool then 'execute' via execute_sql_tool. No parameterization at application layer." },
  { name:"blog-writer",                        findings:"AS-06,07", cells:{exec:N,file:F,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-06 VERIFIED: tools.py:21 open(filename,'w') — zero path validation, filename fully LLM-controlled. AS-07 VERIFIED: tools.py:28 glob.glob(os.path.join(directory,'**'), recursive=True) on LLM-supplied directory, reads + returns all file contents in codebase_context." },
  { name:"fomc-research",                      findings:"AS-08",    cells:{exec:N,file:N,ssrf:F,sql:N,ipi:F,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-08 VERIFIED: fetch_page.py:40 urllib.request.urlopen(url) — zero scheme filter, file:// and http://169.254.169.254 confirmed. IPI chain: LLM reads Fed page → extracts URLs from content → passes to fetch_page, attacker poisons via Fed page injection. ESCALATED: file_utils.py:48 requests.get(url) second SSRF sink, same poisoning chain — no scheme validation." },
  { name:"supply-chain",                       findings:"AS-09",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-09 VERIFIED: execute_sql.py:57 bigquery_client.query_and_wait(sql_query) — raw SQL string, zero parameterization. User NL → LLM SQL → raw BQ exec." },
  { name:"swe-benchmark-agent",                findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"Intentional Docker-isolated exec for SWE/Terminal-Bench. By design. Not in scope." },
  { name:"camel",                              findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"eval() only on Python AST field type annotations via pydantic — not user data." },
  { name:"brand-search-optimization",          findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"Selenium execute_script() with hardcoded scroll values only. BQ uses parameterized queries." },
  { name:"retail-ai-location-strategy",        findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"BuiltInCodeExecutor (sandboxed Vertex AI execution). Not exploitable." },
  { name:"ai-security-agent",                  findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"LLM red-team orchestrator — all simulation via sub-agents, no exec sinks." },
  { name:"software-bug-assistant",             findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"CLOSED CLEAN: SQL tool is ToolboxSyncClient (MCP Toolbox for Databases) at localhost:5000. Toolbox uses parameterized @param syntax in tools.yaml — SQL never touches agent code. No injection surface." },
  { name:"data-engineering",                   findings:"AS-14",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-14 VERIFIED (4 instances): (1) BigQueryToolset NL2SQL confirmed. (2) bigquery_tools.py:87 get_udf_sp_tool — WHERE routine_type = '{routine_type}' f-string, LLM-controlled. (3) bigquery_tools.py:146,150,159 validate_table_data — WHERE {column} IS NULL / GROUP BY {column} / WHERE {column} != {value}, all LLM-controlled. (4) bigquery_tools.py:223,225 sample_table_data_tool — FROM ...{dataset_id}.{table_id} LIMIT {sample_size}, LLM-controlled. CLOUD confirmed clean: bigquery_job_details_tool (client.get_job read-only), GCS tools (bucket.exists/blob.exists read-only only)." },
  { name:"google-trends-agent",                findings:"AS-10",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-10 VERIFIED: tools.py:17-36 clean_sql_query() strips only backticks/newlines/code-fences (cosmetic). client.query(cleaned_sql) at line 31 — raw SQL execution. Two-agent pipeline confirmed: TrendsQueryGeneratorAgent (output_key='generated_sql') → TrendsQueryExecutorAgent instruction 'do not modify it, simply pass it to the tool'." },
  { name:"product-catalog-ad-generation",      findings:"AS-11",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-11 VERIFIED: select_product.py:46-50 query = f\"SELECT * FROM ...WHERE '{normalized_product_name}' IN UNNEST(search_tags)\". .lower().strip() cosmetic only. LLM-controlled product_name f-string interpolated directly into WHERE clause." },
  { name:"agent-observability-bq",             findings:"AS-12",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-12 VERIFIED: agent.py:79-80 system prompt instructs LLM to 'write a valid BigQuery Standard SQL query' then execute via BigQueryToolset. No parameterization guardrail at application layer." },
  { name:"hierarchical-workflow-automation",   findings:"AS-13",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-13 VERIFIED: agent.py:658,744 BigQueryToolset NL2SQL confirmed. ESCALATED: agent.py:490-491 Gmail send — to/subject/body LLM-controlled, subject+body f-string interpolated into MIME headers (gmail_manager.py:176-182) with no sanitization → email header injection risk." },
  { name:"brand-aligner",                      findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"CLOSED CLEAN: GCS blob path f'{app_name}/{user_id}/{user_id}/{filename}'. GCS blob names are opaque strings — ../ sequences stored literally, not normalized by GCS. No cross-user path traversal possible." },
  { name:"medical-pre-authorization",          findings:"AS-15",    cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:F,cloud:N,creds:N}, notes:"AS-15 VERIFIED: tools.py:47-50 extract_treatment_name — si_text = f'...User Query: \"{user_query}\"...' as system_instruction (line 63) to nested Gemini call. user_query unsanitized. Two-stage chain confirmed: (1) poisoned output → treatment_name; (2) treatment_name f-string into prompt_text of extract_policy_information() (lines 105,108-109) and extract_medical_details() (lines 155-157) accessing insurance/medical documents. Note: stage-2 injection is into user prompt_text (not system_instruction) — still a valid prompt injection vector, slightly lower severity than originally noted." },
  ...["RAG","academic-research","agent-skills-tutorial",
      "antom-payment","auto-insurance-agent","bidi-demo",
      "currency-agent","customer-service","deep-search","financial-advisor",
      "gemini-fullstack","image-scoring","incident-management","llm-auditor",
      "marketing-agency","order-processing","parallel_task_decomposition_execution",
      "personalized-shopping","podcast_transcript_agent",
      "realtime-conversational-agent","safety-plugins","short-movie-agents",
      "story_teller","tau2-benchmark-agent","travel-concierge",
      "workflow-concurrent_research_writer","workflow-morning_email_debrief",
      "workflows-HITL_concierge","workflows-sequential"
  ].map(name=>({name, findings:"—", cells:Object.fromEntries(CLASSES.map(c=>[c.id,N])), notes:"Reviewed clean — no exploitable sinks across all 9 attack classes."}))
];
const STATUS_COLORS = {
  FIND: { bg:"#fde8e8", text:"#991f1f", border:"#f09595" },
  NONE: { bg:"#eaf3de", text:"#3b6d11", border:"#97c459" },
  LOOK: { bg:"#faeeda", text:"#854f0b", border:"#ef9f27" },
  TODO: { bg:"transparent", text:"#888780", border:"#d3d1c7" },
};
function Cell({ val }) {
  const s = STATUS_COLORS[val];
  const label = val === "TODO" ? "·" : val === "NONE" ? "✓" : val === "LOOK" ? "?" : "!";
  return (
    <td style={{ textAlign:"center", padding:"4px 2px" }}>
      <span style={{
        display:"inline-block", width:32, height:22, lineHeight:"22px",
        borderRadius:4, fontSize:11, fontWeight:500,
        background:s.bg, color:s.text,
        border:`0.5px solid ${s.border}`,
        userSelect:"none"
      }}>{label}</span>
    </td>
  );
}
function agentStatus(cells) {
  const vals = Object.values(cells);
  if (vals.every(v=>v===T)) return "todo";
  if (vals.some(v=>v===F)) return "finding";
  if (vals.some(v=>v===L)) return "look";
  return "clean";
}
export default function App() {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState(null);
  const agents = useMemo(()=>{
    return RAW_AGENTS.filter(a=>{
      const st = agentStatus(a.cells);
      if (filter==="finding" && st!=="finding") return false;
      if (filter==="look" && st!=="look" && st!=="finding") return false;
      if (filter==="todo" && st!=="todo") return false;
      if (filter==="clean" && st!=="clean") return false;
      if (search && !a.name.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [filter, search]);
  const stats = useMemo(()=>{
    const total = RAW_AGENTS.length;
    const reviewed = RAW_AGENTS.filter(a=>agentStatus(a.cells)!=="todo").length;
    const findings = RAW_AGENTS.filter(a=>agentStatus(a.cells)==="finding").length;
    const look = RAW_AGENTS.filter(a=>agentStatus(a.cells)==="look" || (agentStatus(a.cells)==="clean" && Object.values(a.cells).some(v=>v===L))).length;
    const findingCount = ["AS-01","AS-02","AS-03","AS-04","AS-05","AS-06","AS-07","AS-08","AS-09","AS-10","AS-11","AS-12","AS-13","AS-14","AS-15"].length;
    return { total, reviewed, findings, look, findingCount };
  }, []);
  const pill = (label, val, count) => {
    const active = filter===val;
    const colors = {
      all:     active ? "#534AB7" : "#EEEDFE",
      finding: active ? "#991f1f" : "#fde8e8",
      look:    active ? "#854f0b" : "#faeeda",
      clean:   active ? "#3b6d11" : "#eaf3de",
      todo:    active ? "#5F5E5A" : "#F1EFE8",
    };
    const textColors = {
      all:     active ? "#fff" : "#534AB7",
      finding: active ? "#fff" : "#991f1f",
      look:    active ? "#fff" : "#854f0b",
      clean:   active ? "#fff" : "#3b6d11",
      todo:    active ? "#fff" : "#5F5E5A",
    };
    return (
      <button key={val} onClick={()=>setFilter(val)} style={{
        padding:"4px 10px", borderRadius:20, fontSize:12, fontWeight:500,
        border:`0.5px solid ${colors[val]}`,
        background:colors[val], color:textColors[val],
        cursor:"pointer", transition:"all .15s"
      }}>{label}{count!=null ? ` (${count})` : ""}</button>
    );
  };
  return (
    <div style={{ fontFamily:"var(--font-sans)", padding:"1rem 0" }}>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4, 1fr)", gap:10, marginBottom:"1.25rem" }}>
        {[
          { label:"Total agents", val:stats.total, sub:"in repo" },
          { label:"Reviewed", val:`${stats.reviewed}/${stats.total}`, sub:`${Math.round(stats.reviewed/stats.total*100)}% coverage` },
          { label:"With findings", val:stats.findings, sub:"agents" },
          { label:"AS-XX findings", val:stats.findingCount, sub:"logged so far" },
        ].map(({label,val,sub})=>(
          <div key={label} style={{ background:"var(--color-background-secondary)", borderRadius:8, padding:"10px 14px" }}>
            <div style={{ fontSize:12, color:"var(--color-text-secondary)", marginBottom:2 }}>{label}</div>
            <div style={{ fontSize:22, fontWeight:500, color:"var(--color-text-primary)", lineHeight:1.2 }}>{val}</div>
            <div style={{ fontSize:11, color:"var(--color-text-tertiary)", marginTop:2 }}>{sub}</div>
          </div>
        ))}
      </div>
      <div style={{ display:"flex", gap:6, alignItems:"center", marginBottom:"1rem", flexWrap:"wrap" }}>
        {pill("All", "all")}
        {pill("Findings", "finding", stats.findings)}
        {pill("Needs look", "look")}
        {pill("Clean", "clean")}
        {pill("Not started", "todo", stats.total - stats.reviewed)}
        <input
          placeholder="Filter by name..."
          value={search} onChange={e=>setSearch(e.target.value)}
          style={{ marginLeft:"auto", fontSize:12, padding:"4px 10px", borderRadius:6,
            border:"0.5px solid var(--color-border-secondary)", background:"var(--color-background-primary)",
            color:"var(--color-text-primary)", width:160 }}
        />
      </div>
      <div style={{ overflowX:"auto" }}>
        <table style={{ width:"100%", borderCollapse:"collapse", fontSize:12, tableLayout:"fixed" }}>
          <colgroup>
            <col style={{ width:220 }} />
            {CLASSES.map(c=><col key={c.id} style={{ width:52 }} />)}
            <col style={{ width:80 }} />
          </colgroup>
          <thead>
            <tr style={{ borderBottom:"1px solid var(--color-border-secondary)" }}>
              <th style={{ textAlign:"left", padding:"6px 8px", fontWeight:500, color:"var(--color-text-secondary)", fontSize:11 }}>Agent</th>
              {CLASSES.map(c=>(
                <th key={c.id} title={c.title} style={{ textAlign:"center", padding:"6px 2px", fontWeight:500,
                  color:"var(--color-text-secondary)", fontSize:10, cursor:"help", letterSpacing:"0.03em" }}>
                  {c.label}
                </th>
              ))}
              <th style={{ textAlign:"center", padding:"6px 4px", fontWeight:500, color:"var(--color-text-secondary)", fontSize:11 }}>Findings</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((a, i)=>{
              const st = agentStatus(a.cells);
              const isExpanded = expanded === a.name;
              const rowBg = st==="finding" ? "rgba(253,232,232,0.35)"
                          : st==="look" ? "rgba(250,238,218,0.25)"
                          : i%2===0 ? "transparent" : "var(--color-background-secondary)";
              return [
                <tr key={a.name}
                  onClick={()=>a.notes ? setExpanded(isExpanded?null:a.name) : null}
                  style={{ background:rowBg, cursor:a.notes?"pointer":"default",
                    borderBottom:"0.5px solid var(--color-border-tertiary)" }}>
                  <td style={{ padding:"5px 8px", color:"var(--color-text-primary)", fontWeight:st==="finding"?500:400,
                    whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>
                    {a.notes ? <span style={{ marginRight:4, fontSize:10, color:"var(--color-text-tertiary)" }}>{isExpanded?"▾":"▸"}</span> : null}
                    {a.name}
                  </td>
                  {CLASSES.map(c=><Cell key={c.id} val={a.cells[c.id]} />)}
                  <td style={{ textAlign:"center", padding:"5px 4px",
                    fontWeight:500, fontSize:11,
                    color: a.findings!=="—" ? "#991f1f" : "var(--color-text-tertiary)" }}>
                    {a.findings}
                  </td>
                </tr>,
                isExpanded && a.notes ? (
                  <tr key={a.name+"_notes"} style={{ background:"var(--color-background-secondary)" }}>
                    <td colSpan={CLASSES.length+2} style={{ padding:"8px 12px 10px 24px",
                      fontSize:12, color:"var(--color-text-secondary)", lineHeight:1.6,
                      borderBottom:"0.5px solid var(--color-border-secondary)" }}>
                      {a.notes}
                    </td>
                  </tr>
                ) : null
              ];
            })}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop:"1.25rem", display:"flex", gap:16, flexWrap:"wrap" }}>
        {[
          { label:"! = finding confirmed", ...STATUS_COLORS.FIND },
          { label:"? = needs deeper look", ...STATUS_COLORS.LOOK },
          { label:"✓ = reviewed clean", ...STATUS_COLORS.NONE },
          { label:"· = not started", ...STATUS_COLORS.TODO },
        ].map(({label,bg,text,border})=>(
          <span key={label} style={{ display:"flex", alignItems:"center", gap:6, fontSize:11, color:"var(--color-text-secondary)" }}>
            <span style={{ display:"inline-block", width:22, height:16, borderRadius:3,
              background:bg, border:`0.5px solid ${border}` }} />
            {label}
          </span>
        ))}
        <span style={{ fontSize:11, color:"var(--color-text-tertiary)", marginLeft:"auto" }}>
          Hover column headers for descriptions · Click rows with findings to expand notes
        </span>
      </div>
    </div>
  );
}
