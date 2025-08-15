# GCP Security Agent - Algorithm Documentation (Pseudocode)

## 1. Overview

This document contains the core algorithms used in the GCP Security Agent system, presented in pseudocode format. These algorithms form the backbone of asset discovery, security analysis, recommendation generation, and agent routing functionality.

## 2. Asset Discovery Algorithm

### 2.1 Natural Language Query Processing Algorithm

```pseudocode
ALGORITHM: ProcessNaturalLanguageQuery
INPUT: query (string), project_id (string), user_context (object)
OUTPUT: structured_response (object)

BEGIN
    // Step 1: Parse and analyze the query
    parsed_query = ParseQuery(query)
    
    // Step 2: Determine query intent
    intent = DetermineQueryIntent(parsed_query)
    
    // Step 3: Extract entities and resource types
    entities = ExtractEntities(parsed_query)
    resource_types = MapEntitiesToResourceTypes(entities)
    
    // Step 4: Route to appropriate discovery function
    SWITCH intent:
        CASE "ASSET_DISCOVERY":
            result = DiscoverAssets(resource_types, project_id)
        CASE "SECURITY_ANALYSIS":
            result = AnalyzeSecurityPosture(resource_types, project_id)
        CASE "RECOMMENDATION_REQUEST":
            result = GenerateRecommendations(resource_types, project_id)
        DEFAULT:
            result = GeneralAssetOverview(project_id)
    END SWITCH
    
    // Step 5: Format response with natural language
    structured_response = FormatNaturalLanguageResponse(result, query, intent)
    
    // Step 6: Log API calls made
    LogAPICallsMetadata(result.api_calls)
    
    RETURN structured_response
END

FUNCTION DetermineQueryIntent(parsed_query):
    keywords_asset = ["show", "list", "get", "what", "how many", "tell me about"]
    keywords_security = ["analyze", "security", "vulnerabilities", "risks", "compliance"]
    keywords_recommendations = ["recommend", "suggest", "improve", "optimize", "best practices"]
    
    IF ContainsAny(parsed_query, keywords_recommendations):
        RETURN "RECOMMENDATION_REQUEST"
    ELSE IF ContainsAny(parsed_query, keywords_security):
        RETURN "SECURITY_ANALYSIS"
    ELSE IF ContainsAny(parsed_query, keywords_asset):
        RETURN "ASSET_DISCOVERY"
    ELSE:
        RETURN "GENERAL_INQUIRY"
    END IF
END

FUNCTION ExtractEntities(parsed_query):
    entity_mappings = {
        "buckets": "storage.googleapis.com/Bucket",
        "instances": "compute.googleapis.com/Instance",
        "functions": "cloudfunctions.googleapis.com/CloudFunction",
        "databases": ["sqladmin.googleapis.com/Instance", "spanner.googleapis.com/Instance"],
        "clusters": "container.googleapis.com/Cluster"
    }
    
    entities = []
    FOR each keyword IN entity_mappings:
        IF keyword IN parsed_query:
            entities.append(entity_mappings[keyword])
        END IF
    END FOR
    
    RETURN entities
END
```

### 2.2 Asset Discovery Core Algorithm

```pseudocode
ALGORITHM: DiscoverGCPResources
INPUT: resource_types (array), project_id (string), filters (object)
OUTPUT: discovery_result (object)

BEGIN
    discovery_result = {
        "assets": [],
        "total_count": 0,
        "api_calls_made": [],
        "processing_time": 0,
        "security_summary": {}
    }
    
    start_time = GetCurrentTime()
    
    // Step 1: Build Asset Inventory query
    query_filter = BuildAssetInventoryQuery(resource_types, project_id, filters)
    
    // Step 2: Call GCP Asset Inventory API
    TRY:
        api_call_start = GetCurrentTime()
        raw_assets = CallAssetInventoryAPI(query_filter)
        api_call_duration = GetCurrentTime() - api_call_start
        
        LogAPICall("cloudasset.googleapis.com", "searchAllResources", api_call_duration)
        discovery_result.api_calls_made.append({
            "service": "cloudasset.googleapis.com",
            "method": "searchAllResources",
            "duration_ms": api_call_duration,
            "timestamp": GetCurrentTime()
        })
        
    CATCH api_error:
        RETURN HandleAPIError(api_error, "Asset Inventory")
    END TRY
    
    // Step 3: Process and enrich asset data
    processed_assets = []
    FOR each asset IN raw_assets:
        enriched_asset = EnrichAssetData(asset)
        security_analysis = AnalyzeAssetSecurity(enriched_asset)
        enriched_asset.security_findings = security_analysis.findings
        enriched_asset.risk_level = security_analysis.risk_level
        processed_assets.append(enriched_asset)
    END FOR
    
    // Step 4: Generate security summary
    security_summary = GenerateSecuritySummary(processed_assets)
    
    discovery_result.assets = processed_assets
    discovery_result.total_count = LENGTH(processed_assets)
    discovery_result.security_summary = security_summary
    discovery_result.processing_time = GetCurrentTime() - start_time
    
    RETURN discovery_result
END

FUNCTION BuildAssetInventoryQuery(resource_types, project_id, filters):
    base_query = "projects/" + project_id
    
    IF resource_types IS NOT EMPTY:
        type_filter = "assetTypes:(" + JOIN(resource_types, " OR ") + ")"
        query = base_query + " AND " + type_filter
    ELSE:
        query = base_query
    END IF
    
    // Add additional filters
    IF filters.location IS NOT NULL:
        query += " AND location:" + filters.location
    END IF
    
    IF filters.state IS NOT NULL:
        query += " AND state:" + filters.state
    END IF
    
    RETURN query
END

FUNCTION EnrichAssetData(asset):
    enriched_asset = asset
    
    // Add computed fields
    enriched_asset.resource_hierarchy = ParseResourceHierarchy(asset.name)
    enriched_asset.last_seen = GetCurrentTime()
    enriched_asset.tags = ExtractResourceTags(asset)
    
    // Fetch additional metadata based on asset type
    SWITCH asset.assetType:
        CASE "compute.googleapis.com/Instance":
            enriched_asset.metadata = GetComputeInstanceMetadata(asset)
        CASE "storage.googleapis.com/Bucket":
            enriched_asset.metadata = GetStorageBucketMetadata(asset)
        CASE "container.googleapis.com/Cluster":
            enriched_asset.metadata = GetGKEClusterMetadata(asset)
    END SWITCH
    
    RETURN enriched_asset
END
```

## 3. Security Analysis Algorithm

### 3.1 Asset Security Scoring Algorithm

```pseudocode
ALGORITHM: AnalyzeAssetSecurity
INPUT: asset (object)
OUTPUT: security_analysis (object)

BEGIN
    security_analysis = {
        "findings": [],
        "risk_level": "UNKNOWN",
        "security_score": 0,
        "recommendations": [],
        "compliance_status": {}
    }
    
    base_score = 100  // Start with perfect score
    
    // Step 1: Run asset-specific security checks
    security_checks = GetSecurityChecksForAssetType(asset.assetType)
    
    FOR each check IN security_checks:
        check_result = ExecuteSecurityCheck(check, asset)
        
        IF check_result.passed == FALSE:
            // Create security finding
            finding = {
                "check_name": check.name,
                "severity": check.severity,
                "description": check_result.description,
                "remediation": check.remediation_guidance,
                "compliance_frameworks": check.compliance_frameworks
            }
            security_analysis.findings.append(finding)
            
            // Deduct points based on severity
            score_deduction = CalculateScoreDeduction(check.severity)
            base_score -= score_deduction
        END IF
    END FOR
    
    // Step 2: Calculate final risk level
    security_analysis.security_score = MAX(0, base_score)
    security_analysis.risk_level = DetermineRiskLevel(security_analysis.security_score)
    
    // Step 3: Generate recommendations
    security_analysis.recommendations = GenerateSecurityRecommendations(
        security_analysis.findings, 
        asset
    )
    
    // Step 4: Check compliance status
    security_analysis.compliance_status = CheckComplianceFrameworks(
        security_analysis.findings,
        asset
    )
    
    RETURN security_analysis
END

FUNCTION GetSecurityChecksForAssetType(asset_type):
    SWITCH asset_type:
        CASE "storage.googleapis.com/Bucket":
            RETURN [
                {
                    "name": "public_access_check",
                    "severity": "HIGH",
                    "description": "Check for public read/write access",
                    "remediation_guidance": "Remove public access and use IAM policies"
                },
                {
                    "name": "uniform_bucket_access_check",
                    "severity": "MEDIUM",
                    "description": "Check for uniform bucket-level access",
                    "remediation_guidance": "Enable uniform bucket-level access"
                },
                {
                    "name": "versioning_check",
                    "severity": "LOW",
                    "description": "Check if versioning is enabled",
                    "remediation_guidance": "Enable versioning for data protection"
                }
            ]
        
        CASE "compute.googleapis.com/Instance":
            RETURN [
                {
                    "name": "public_ip_check",
                    "severity": "MEDIUM",
                    "description": "Check for public IP exposure",
                    "remediation_guidance": "Use Cloud NAT or private IPs"
                },
                {
                    "name": "os_login_check",
                    "severity": "HIGH",
                    "description": "Check if OS Login is enabled",
                    "remediation_guidance": "Enable OS Login for centralized access"
                },
                {
                    "name": "shielded_vm_check",
                    "severity": "MEDIUM",
                    "description": "Check if Shielded VM is enabled",
                    "remediation_guidance": "Enable Shielded VM features"
                }
            ]
        
        DEFAULT:
            RETURN GetGenericSecurityChecks()
    END SWITCH
END

FUNCTION ExecuteSecurityCheck(check, asset):
    SWITCH check.name:
        CASE "public_access_check":
            RETURN CheckBucketPublicAccess(asset)
        CASE "uniform_bucket_access_check":
            RETURN CheckUniformBucketAccess(asset)
        CASE "public_ip_check":
            RETURN CheckInstancePublicIP(asset)
        CASE "os_login_check":
            RETURN CheckOSLogin(asset)
        DEFAULT:
            RETURN {"passed": TRUE, "description": "Check not implemented"}
    END SWITCH
END

FUNCTION CalculateScoreDeduction(severity):
    SWITCH severity:
        CASE "CRITICAL":
            RETURN 30
        CASE "HIGH":
            RETURN 20
        CASE "MEDIUM":
            RETURN 10
        CASE "LOW":
            RETURN 5
        DEFAULT:
            RETURN 0
    END SWITCH
END

FUNCTION DetermineRiskLevel(security_score):
    IF security_score >= 90:
        RETURN "LOW"
    ELSE IF security_score >= 70:
        RETURN "MEDIUM"
    ELSE IF security_score >= 50:
        RETURN "HIGH"
    ELSE:
        RETURN "CRITICAL"
    END IF
END
```

### 3.2 Security Finding Aggregation Algorithm

```pseudocode
ALGORITHM: GenerateSecuritySummary
INPUT: assets (array)
OUTPUT: security_summary (object)

BEGIN
    security_summary = {
        "total_assets": 0,
        "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
        "finding_categories": {},
        "top_risks": [],
        "compliance_overview": {},
        "average_security_score": 0
    }
    
    total_score = 0
    all_findings = []
    
    // Step 1: Aggregate data from all assets
    FOR each asset IN assets:
        security_summary.total_assets += 1
        security_summary.risk_distribution[asset.risk_level] += 1
        total_score += asset.security_score
        
        // Collect all findings
        FOR each finding IN asset.security_findings:
            all_findings.append(finding)
            
            category = finding.check_name
            IF category NOT IN security_summary.finding_categories:
                security_summary.finding_categories[category] = 0
            END IF
            security_summary.finding_categories[category] += 1
        END FOR
    END FOR
    
    // Step 2: Calculate average security score
    IF security_summary.total_assets > 0:
        security_summary.average_security_score = total_score / security_summary.total_assets
    END IF
    
    // Step 3: Identify top risks
    security_summary.top_risks = IdentifyTopRisks(all_findings)
    
    // Step 4: Generate compliance overview
    security_summary.compliance_overview = GenerateComplianceOverview(all_findings)
    
    RETURN security_summary
END

FUNCTION IdentifyTopRisks(findings):
    risk_priority = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }
    
    // Group findings by type and severity
    risk_groups = {}
    FOR each finding IN findings:
        key = finding.check_name + "_" + finding.severity
        IF key NOT IN risk_groups:
            risk_groups[key] = {
                "type": finding.check_name,
                "severity": finding.severity,
                "count": 0,
                "priority_score": risk_priority[finding.severity]
            }
        END IF
        risk_groups[key].count += 1
        risk_groups[key].priority_score *= risk_groups[key].count
    END FOR
    
    // Sort by priority score and return top 5
    sorted_risks = SortByPriorityScore(risk_groups)
    RETURN Take(sorted_risks, 5)
END
```

## 4. Recommendation Prioritization Algorithm

### 4.1 Recommendation Generation Algorithm

```pseudocode
ALGORITHM: GenerateRecommendations
INPUT: security_findings (array), assets (array), context (object)
OUTPUT: prioritized_recommendations (array)

BEGIN
    all_recommendations = []
    
    // Step 1: Generate recommendations from security findings
    FOR each finding IN security_findings:
        recommendation = CreateRecommendationFromFinding(finding)
        all_recommendations.append(recommendation)
    END FOR
    
    // Step 2: Generate proactive recommendations
    proactive_recs = GenerateProactiveRecommendations(assets, context)
    all_recommendations.extend(proactive_recs)
    
    // Step 3: Deduplicate similar recommendations
    deduplicated_recs = DeduplicateRecommendations(all_recommendations)
    
    // Step 4: Prioritize recommendations
    prioritized_recommendations = PrioritizeRecommendations(deduplicated_recs)
    
    RETURN prioritized_recommendations
END

FUNCTION PrioritizeRecommendations(recommendations):
    // Scoring factors
    weights = {
        "security_impact": 0.4,
        "compliance_requirement": 0.3,
        "implementation_effort": 0.2,
        "cost_impact": 0.1
    }
    
    FOR each recommendation IN recommendations:
        score = 0
        
        // Security Impact Score (0-10)
        security_score = CalculateSecurityImpactScore(recommendation)
        score += security_score * weights.security_impact
        
        // Compliance Requirement Score (0-10)
        compliance_score = CalculateComplianceScore(recommendation)
        score += compliance_score * weights.compliance_requirement
        
        // Implementation Effort Score (inverted, 10-0)
        effort_score = 10 - CalculateImplementationEffortScore(recommendation)
        score += effort_score * weights.implementation_effort
        
        // Cost Impact Score (inverted, 10-0)
        cost_score = 10 - CalculateCostImpactScore(recommendation)
        score += cost_score * weights.cost_impact
        
        recommendation.priority_score = score
    END FOR
    
    // Sort by priority score (highest first)
    sorted_recommendations = SortByPriorityScore(recommendations, "DESC")
    
    // Assign priority levels
    total_count = LENGTH(sorted_recommendations)
    FOR i = 0 TO total_count - 1:
        IF i < total_count * 0.2:
            sorted_recommendations[i].priority = "CRITICAL"
        ELSE IF i < total_count * 0.5:
            sorted_recommendations[i].priority = "HIGH"
        ELSE IF i < total_count * 0.8:
            sorted_recommendations[i].priority = "MEDIUM"
        ELSE:
            sorted_recommendations[i].priority = "LOW"
        END IF
    END FOR
    
    RETURN sorted_recommendations
END

FUNCTION CalculateSecurityImpactScore(recommendation):
    impact_mapping = {
        "data_exposure": 10,
        "unauthorized_access": 9,
        "privilege_escalation": 8,
        "compliance_violation": 7,
        "configuration_weakness": 6,
        "monitoring_gap": 5,
        "best_practice": 4
    }
    
    RETURN impact_mapping.get(recommendation.impact_category, 5)
END

FUNCTION CalculateComplianceScore(recommendation):
    compliance_frameworks = recommendation.compliance_frameworks
    
    IF "SOC2" IN compliance_frameworks:
        RETURN 10
    ELSE IF "ISO27001" IN compliance_frameworks:
        RETURN 9
    ELSE IF "NIST" IN compliance_frameworks:
        RETURN 8
    ELSE:
        RETURN 5
    END IF
END
```

## 5. Agent Routing Algorithm

### 5.1 Intelligent Agent Selection Algorithm

```pseudocode
ALGORITHM: RouteToOptimalAgent
INPUT: query (string), context (object), available_agents (array)
OUTPUT: selected_agent (string), routing_confidence (float)

BEGIN
    // Step 1: Analyze query characteristics
    query_features = ExtractQueryFeatures(query)
    
    // Step 2: Calculate agent suitability scores
    agent_scores = {}
    FOR each agent IN available_agents:
        suitability_score = CalculateAgentSuitability(agent, query_features, context)
        agent_scores[agent.name] = suitability_score
    END FOR
    
    // Step 3: Apply context-based routing adjustments
    adjusted_scores = ApplyContextAdjustments(agent_scores, context)
    
    // Step 4: Select the best agent
    selected_agent = GetAgentWithHighestScore(adjusted_scores)
    routing_confidence = adjusted_scores[selected_agent] / 100.0
    
    // Step 5: Log routing decision
    LogRoutingDecision(query, selected_agent, routing_confidence, adjusted_scores)
    
    RETURN selected_agent, routing_confidence
END

FUNCTION ExtractQueryFeatures(query):
    features = {
        "asset_focus": 0,
        "security_focus": 0,
        "analysis_complexity": 0,
        "recommendation_request": 0,
        "specific_resource_type": 0
    }
    
    query_lower = ToLowerCase(query)
    
    // Asset focus indicators
    asset_keywords = ["show", "list", "get", "what", "how many", "tell me about"]
    features.asset_focus = CountKeywordMatches(query_lower, asset_keywords) * 20
    
    // Security focus indicators
    security_keywords = ["security", "vulnerability", "risk", "compliance", "analyze"]
    features.security_focus = CountKeywordMatches(query_lower, security_keywords) * 25
    
    // Analysis complexity indicators
    complex_keywords = ["analyze", "evaluate", "assess", "comprehensive", "detailed"]
    features.analysis_complexity = CountKeywordMatches(query_lower, complex_keywords) * 15
    
    // Recommendation request indicators
    rec_keywords = ["recommend", "suggest", "improve", "optimize", "best practice"]
    features.recommendation_request = CountKeywordMatches(query_lower, rec_keywords) * 30
    
    // Specific resource type indicators
    resource_keywords = ["bucket", "instance", "function", "database", "cluster"]
    features.specific_resource_type = CountKeywordMatches(query_lower, resource_keywords) * 10
    
    RETURN features
END

FUNCTION CalculateAgentSuitability(agent, query_features, context):
    base_score = 50  // Neutral starting point
    
    SWITCH agent.type:
        CASE "SecurityAgent":
            base_score += query_features.security_focus
            base_score += query_features.analysis_complexity
            IF context.previous_agent == "SecurityAgent":
                base_score += 10  // Continuity bonus
            END IF
        
        CASE "AssetDiscoveryAgent":
            base_score += query_features.asset_focus
            base_score += query_features.specific_resource_type
            IF context.topic == "asset_discovery":
                base_score += 15  // Context match bonus
            END IF
        
        CASE "CoordinatorAgent":
            base_score += query_features.analysis_complexity
            IF query_features.asset_focus > 50 AND query_features.security_focus > 50:
                base_score += 20  // Multi-domain query bonus
            END IF
        
        CASE "SearchEnabledAgent":
            search_indicators = CountSearchKeywords(query)
            base_score += search_indicators * 15
    END SWITCH
    
    // Apply agent-specific capabilities bonus
    FOR each capability IN agent.capabilities:
        IF capability.matches_query(query_features):
            base_score += capability.relevance_score
        END IF
    END FOR
    
    // Apply workload balancing
    workload_factor = CalculateWorkloadFactor(agent)
    base_score *= workload_factor
    
    RETURN MIN(100, MAX(0, base_score))
END

FUNCTION ApplyContextAdjustments(agent_scores, context):
    adjusted_scores = agent_scores.copy()
    
    // Conversation continuity adjustment
    IF context.previous_agent IS NOT NULL:
        IF context.conversation_depth < 3:
            // Favor the same agent for short conversations
            adjusted_scores[context.previous_agent] += 15
        END IF
    END IF
    
    // Topic consistency adjustment
    IF context.topic IS NOT NULL:
        topic_agent_mapping = {
            "storage_analysis": "AssetDiscoveryAgent",
            "security_evaluation": "SecurityAgent",
            "multi_resource_analysis": "CoordinatorAgent"
        }
        
        preferred_agent = topic_agent_mapping.get(context.topic)
        IF preferred_agent IS NOT NULL:
            adjusted_scores[preferred_agent] += 10
        END IF
    END IF
    
    // Expertise escalation adjustment
    IF context.query_complexity == "HIGH":
        adjusted_scores["CoordinatorAgent"] += 20
        adjusted_scores["SecurityAgent"] += 15
    END IF
    
    RETURN adjusted_scores
END
```

### 5.2 Multi-Agent Coordination Algorithm

```pseudocode
ALGORITHM: CoordinateMultiAgentWorkflow
INPUT: complex_query (string), available_agents (array), context (object)
OUTPUT: workflow_result (object)

BEGIN
    workflow_result = {
        "primary_result": "",
        "contributing_agents": [],
        "coordination_time": 0,
        "partial_results": {},
        "synthesis_quality": 0
    }
    
    start_time = GetCurrentTime()
    
    // Step 1: Decompose complex query into sub-tasks
    sub_tasks = DecomposeQuery(complex_query)
    
    // Step 2: Assign agents to sub-tasks
    agent_assignments = AssignAgentsToTasks(sub_tasks, available_agents)
    
    // Step 3: Execute sub-tasks in parallel
    partial_results = {}
    FOR each assignment IN agent_assignments:
        ExecuteInParallel(
            task_id = assignment.task_id,
            agent = assignment.agent,
            sub_query = assignment.sub_query,
            callback = StorePartialResult(partial_results)
        )
    END FOR
    
    // Step 4: Wait for all tasks to complete
    WaitForAllTasks(agent_assignments)
    
    // Step 5: Synthesize results
    synthesized_result = SynthesizePartialResults(
        partial_results, 
        complex_query, 
        context
    )
    
    workflow_result.primary_result = synthesized_result.text
    workflow_result.contributing_agents = GetContributingAgents(agent_assignments)
    workflow_result.coordination_time = GetCurrentTime() - start_time
    workflow_result.partial_results = partial_results
    workflow_result.synthesis_quality = synthesized_result.quality_score
    
    RETURN workflow_result
END

FUNCTION DecomposeQuery(complex_query):
    // Use NLP techniques to identify multiple intents
    intents = IdentifyMultipleIntents(complex_query)
    
    sub_tasks = []
    FOR each intent IN intents:
        sub_task = {
            "task_id": GenerateTaskID(),
            "intent": intent.type,
            "sub_query": ExtractSubQuery(complex_query, intent),
            "priority": intent.priority,
            "dependencies": intent.dependencies
        }
        sub_tasks.append(sub_task)
    END FOR
    
    // Sort by priority and dependencies
    ordered_tasks = TopologicalSort(sub_tasks)
    
    RETURN ordered_tasks
END

FUNCTION SynthesizePartialResults(partial_results, original_query, context):
    synthesis = {
        "text": "",
        "quality_score": 0,
        "coherence": 0,
        "completeness": 0
    }
    
    // Step 1: Organize results by relevance
    ordered_results = OrderResultsByRelevance(partial_results, original_query)
    
    // Step 2: Create coherent narrative
    narrative_sections = []
    FOR each result IN ordered_results:
        section = CreateNarrativeSection(result, context)
        narrative_sections.append(section)
    END FOR
    
    // Step 3: Combine sections with transitions
    synthesis.text = CombineWithTransitions(narrative_sections)
    
    // Step 4: Calculate quality metrics
    synthesis.coherence = CalculateCoherence(narrative_sections)
    synthesis.completeness = CalculateCompleteness(partial_results, original_query)
    synthesis.quality_score = (synthesis.coherence + synthesis.completeness) / 2
    
    RETURN synthesis
END
```

## 6. Performance Optimization Algorithms

### 6.1 Caching Strategy Algorithm

```pseudocode
ALGORITHM: OptimizedCacheStrategy
INPUT: request (object), cache_store (object)
OUTPUT: response (object), cache_hit (boolean)

BEGIN
    // Step 1: Generate cache key
    cache_key = GenerateCacheKey(request)
    
    // Step 2: Check cache
    cached_response = cache_store.get(cache_key)
    IF cached_response IS NOT NULL:
        IF IsValidCacheEntry(cached_response, request):
            RETURN cached_response.data, TRUE
        ELSE:
            cache_store.delete(cache_key)
        END IF
    END IF
    
    // Step 3: Execute request
    response = ExecuteRequest(request)
    
    // Step 4: Determine cache worthiness
    IF IsCacheWorthy(request, response):
        ttl = CalculateOptimalTTL(request.type, response.complexity)
        cache_store.set(cache_key, response, ttl)
    END IF
    
    RETURN response, FALSE
END

FUNCTION CalculateOptimalTTL(request_type, complexity):
    base_ttl = {
        "asset_discovery": 300,    // 5 minutes
        "security_analysis": 900,  // 15 minutes
        "recommendations": 1800,   // 30 minutes
        "session_data": 3600      // 1 hour
    }
    
    ttl = base_ttl.get(request_type, 300)
    
    // Adjust based on complexity
    IF complexity == "HIGH":
        ttl *= 2  // Cache longer for expensive operations
    ELSE IF complexity == "LOW":
        ttl /= 2  // Shorter cache for simple operations
    END IF
    
    RETURN ttl
END
```

This algorithm documentation provides the foundation for understanding how the GCP Security Agent processes queries, discovers assets, analyzes security, generates recommendations, routes requests to agents, and optimizes performance through intelligent caching and coordination strategies.