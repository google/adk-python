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
  { name:"youtube-analyst",                    findings:"AS-01",    cells:{exec:F,file:F,ssrf:N,sql:N,ipi:F,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-01: exec() unrestricted builtins; os.path.join() path traversal in filename arg. IPI chain: YouTube video/comment → visualization agent → exec()." },
  { name:"policy-as-code",                     findings:"AS-02",    cells:{exec:F,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-02: exec(policy_code) sandbox escape via __import__. Boss: simulation context, limited blast radius — deprioritise." },
  { name:"machine-learning-engineering",       findings:"AS-03",    cells:{exec:F,file:L,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-03: LLM generates code → subprocess.run() in code_util.py. File ops on run_cwd paths — needs deeper path-control check." },
  { name:"plumber-data-engineering-assistant", findings:"AS-04",    cells:{exec:F,file:N,ssrf:N,sql:N,ipi:F,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-04: monitoring_agent shell=True + job_id injection. Path A: direct user input. Path B: IPI via Dataproc job output. Dev explicitly noted insecurity inline." },
  { name:"data-science",                       findings:"AS-05",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-05: NL2SQL no validation, confused deputy pattern. BigQueryToolset." },
  { name:"blog-writer",                        findings:"AS-06,07", cells:{exec:N,file:F,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-06: open(filename,'w') LLM-controlled filename → arbitrary write (path traversal to ~/.ssh, cron.d etc). AS-07: analyze_codebase(directory) LLM-controlled dir → recursive full read + LLM exfil." },
  { name:"fomc-research",                      findings:"AS-08",    cells:{exec:N,file:N,ssrf:F,sql:N,ipi:F,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-08: urllib.request.urlopen(url) zero scheme filter → file:// and http://169.254.169.254. IPI: LLM extracts URLs from fetched Fed page content, passes to fetch_page — attacker controls via Fed page injection or confused deputy." },
  { name:"supply-chain",                       findings:"AS-09",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-09: execute_sql_query(sql_query) → client.query_and_wait(sql_query), zero parameterization. User NL → LLM SQL → raw BQ exec." },
  { name:"swe-benchmark-agent",                findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"Intentional Docker-isolated exec for SWE/Terminal-Bench. By design. Not in scope." },
  { name:"camel",                              findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"eval() only on Python AST field type annotations via pydantic — not user data." },
  { name:"brand-search-optimization",          findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"Selenium execute_script() with hardcoded scroll values only. BQ uses parameterized queries." },
  { name:"retail-ai-location-strategy",        findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"BuiltInCodeExecutor (sandboxed Vertex AI execution). Not exploitable." },
  { name:"ai-security-agent",                  findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"LLM red-team orchestrator — all simulation via sub-agents, no exec sinks." },
  { name:"software-bug-assistant",             findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"CLOSED CLEAN: SQL tool is ToolboxSyncClient (MCP Toolbox for Databases) at localhost:5000. Toolbox uses parameterized @param syntax in tools.yaml — SQL never touches agent code. No injection surface." },
  { name:"data-engineering",                   findings:"AS-14",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-14 (extended): (1) BigQueryToolset NL2SQL — query_and_wait(raw_sql), unparameterized (confirmed via PY-04). (2) get_udf_sp_tool(dataset_id, routine_type): f-string into client.query() — LLM-controlled. (3) validate_table_data(dataset_id, table_id, column, value): f-string WHERE {column} IS NULL — all params LLM-controlled. (4) sample_table_data_tool(dataset_id, table_id): f-string FROM ...{dataset_id}.{table_id} — LLM-controlled. CLOUD LOOK closed clean: bigquery_job_details_tool(job_id) and validate_bucket_exists_tool(bucket_name) are bounded read-only GCP API calls, no injection surface." },
  { name:"google-trends-agent",                findings:"AS-10",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-10: tools.py execute_bigquery_sql(sql): cleaned_sql = clean_sql_query(sql); client.query(cleaned_sql). clean_sql_query() removes backticks/newlines/code fences — cosmetic only, not a security control. Two-stage: TrendsQueryGeneratorAgent writes SQL as string, TrendsQueryExecutorAgent executes verbatim. Third instance of NL2SQL no-parameterization pattern." },
  { name:"product-catalog-ad-generation",      findings:"AS-11",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-11: select_product.py: normalized_product_name = product_name.lower().strip(); query = f\"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` WHERE '{normalized_product_name}' IN UNNEST(search_tags)\"; client.query(query). .lower().strip() is cosmetic, not SQL sanitization. LLM-controlled product_name interpolated directly into query. Fourth NL2SQL instance." },
  { name:"agent-observability-bq",             findings:"AS-12",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-12: ADK BigQueryToolset NL2SQL — system prompt 'write a valid BigQuery Standard SQL query and execute it.' Toolset confirmed unparameterized via PY-04 (query_and_wait(raw_sql)). Closed via BigQueryToolset library source read." },
  { name:"hierarchical-workflow-automation",   findings:"AS-13",    cells:{exec:N,file:N,ssrf:N,sql:F,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"AS-13: ADK BigQueryToolset NL2SQL — same root cause as AS-12, confirmed via PY-04. Note: Gmail send with LLM-controlled to/subject/body — bounded but potential agent-as-spam-relay if injected." },
  { name:"brand-aligner",                      findings:"—",        cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:N,cloud:N,creds:N}, notes:"CLOSED CLEAN: GCS blob path f'{app_name}/{user_id}/{user_id}/{filename}'. GCS blob names are opaque strings — ../ sequences stored literally, not normalized by GCS. No cross-user path traversal possible." },
  { name:"medical-pre-authorization",          findings:"AS-15",    cells:{exec:N,file:N,ssrf:N,sql:N,ipi:N,deser:N,tmpl:F,cloud:N,creds:N}, notes:"AS-15: extract_treatment_name(user_query) in information_extractor/tools/tools.py: si_text = f'...User Query: \"{user_query}\"...' passed as system_instruction to nested Gemini API call. user_query is raw user input, no sanitization. Two-stage chain: (1) poisoned output becomes treatment_name, (2) treatment_name f-string interpolated into system_instruction of extract_policy_information() and extract_medical_details() which access patient records and insurance documents. Medical context elevates impact. Design-level." },
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
