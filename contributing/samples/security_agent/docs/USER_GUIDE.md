# GCP Security Agent - User Guide

## 1. Getting Started

### 1.1 Welcome to GCP Security Agent
The GCP Security Agent is an intelligent security analysis platform that helps you understand, monitor, and improve the security posture of your Google Cloud Platform infrastructure. Using conversational AI and comprehensive asset discovery, it provides actionable security insights and recommendations.

### 1.2 What You Can Do
- **Discover Assets**: Get a complete inventory of your GCP resources
- **Analyze Security**: Comprehensive security analysis with risk assessment
- **Get Recommendations**: AI-powered security improvement suggestions
- **Monitor Compliance**: Track compliance with security frameworks (SOC2, ISO27001)
- **Chat Interface**: Natural language queries about your security posture

### 1.3 Quick Start
1. **Access the Application**: Navigate to your deployed instance or run locally
2. **Connect to GCP**: Ensure your project credentials are configured
3. **Start Chatting**: Ask questions about your GCP security in natural language
4. **Review Results**: Get instant insights and actionable recommendations

## 2. System Requirements

### 2.1 For Users
- **Web Browser**: Modern browser (Chrome, Firefox, Safari, Edge)
- **Internet Connection**: Stable connection for real-time analysis
- **GCP Access**: Read access to your Google Cloud Project

### 2.2 GCP Permissions Required
Your account needs these IAM roles:
- `roles/cloudasset.viewer` - View cloud assets
- `roles/compute.viewer` - View compute resources
- `roles/storage.objectViewer` - View storage resources
- `roles/iam.securityReviewer` - Review security configurations

### 2.3 Supported GCP Services
The system can analyze security for:
- **Compute Engine**: Virtual machines and instances
- **Cloud Storage**: Buckets and objects
- **Cloud Functions**: Serverless functions
- **Cloud SQL**: Managed databases
- **Google Kubernetes Engine**: Container clusters
- **IAM**: Identity and access management
- **Networking**: VPCs, firewalls, and load balancers

## 3. User Interface Overview

### 3.1 Main Dashboard
When you first access the application, you'll see:

#### 3.1.1 Navigation Sidebar
- **💬 AI Assistant**: Main chat interface
- **📊 Security Dashboard**: Overview of security metrics
- **🔍 Asset Explorer**: Detailed asset inventory
- **📋 Recommendations**: Security improvement suggestions
- **⚙️ Settings**: Configuration options

#### 3.1.2 Asset Inventory Overview
- **Total Assets**: Count of discovered resources
- **Security Findings**: Number of security issues detected
- **High Risk Assets**: Critical security concerns
- **Active Recommendations**: Suggested improvements

### 3.2 Chat Interface
The main interaction method is through natural language chat:

#### 3.2.1 Chat Features
- **Natural Language**: Ask questions in plain English
- **Context Awareness**: Maintains conversation context
- **Agent Delegation**: Intelligent routing to specialized agents
- **Follow-up Suggestions**: Smart suggestions for next questions
- **Session Management**: Persistent conversation history

#### 3.2.2 Chat Interface Layout
```
┌─────────────────────────────────────────┐
│ 🔐 GCP Security Assistant               │
├─────────────────────────────────────────┤
│ 📊 Asset Inventory Overview             │
│ Total Assets: 42 | Security Findings: 7 │
├─────────────────────────────────────────┤
│ 💭 Conversation Context                 │
│ Topic: Storage Analysis | Agents: 2     │
├─────────────────────────────────────────┤
│                                         │
│ User: Tell me about my storage buckets  │
│                                         │
│ Assistant: I found 10 storage buckets   │
│ in your project. Here's the analysis... │
│ 🤖 Delegated to: AssetDiscoveryAgent    │
│                                         │
├─────────────────────────────────────────┤
│ Ask about your GCP assets, security... │
│ [Type your message here...]             │
└─────────────────────────────────────────┘
```

## 4. Using the Chat Interface

### 4.1 Basic Queries

#### 4.1.1 Asset Discovery
Ask about your resources using natural language:

**Examples:**
- "Show me my compute instances"
- "What storage buckets do I have?"
- "List my cloud functions"
- "Tell me about my databases"
- "What Kubernetes clusters are running?"

**Sample Interaction:**
```
You: Show me my compute instances

Agent: I found 2 compute instances in your project:

**mgm-web-server-01**
- Type: e2-medium
- Zone: us-central1-a  
- Status: RUNNING
- Security Score: 72/100
- Issues: Public IP exposed, OS Login disabled

**mgm-database-server**
- Type: n1-standard-2
- Zone: us-central1-b
- Status: RUNNING
- Security Score: 85/100
- Issues: Shielded VM features disabled

🤖 Delegated to: AssetDiscoveryAgent
⏱️ Response time: 1.2s

❓ Suggestions:
- "Show me more details about the high-risk instance"
- "How can I improve the security of these instances?"
- "What are the recommended security settings?"
```

#### 4.1.2 Security Analysis
Get comprehensive security analysis:

**Examples:**
- "Analyze my security posture"
- "What security vulnerabilities do I have?"
- "Check my compliance with SOC2"
- "Show me high-risk assets"
- "What are my security findings?"

**Sample Interaction:**
```
You: Analyze my security posture

Agent: Here's your current security analysis:

**Overall Security Score: 76/100**

**Risk Distribution:**
- 🔴 High Risk: 3 assets
- 🟡 Medium Risk: 8 assets  
- 🟢 Low Risk: 31 assets

**Top Security Findings:**
1. **Storage Bucket Public Access** (HIGH)
   - Asset: mgm-data-lake-raw
   - Risk: Data exposure to internet
   
2. **Instance Missing OS Login** (MEDIUM)
   - Asset: mgm-web-server-01
   - Risk: Weak access control

3. **Unencrypted Database** (MEDIUM)
   - Asset: customer-db
   - Risk: Data protection issue

**Compliance Status:**
- SOC2: 83% compliant (7 findings)
- ISO27001: 78% compliant (12 findings)

🤖 Delegated to: SecurityAgent
⏱️ Response time: 2.8s

❓ Suggestions:
- "How do I fix the public bucket access?"
- "Show me detailed compliance requirements"
- "What's the impact of these security issues?"
```

#### 4.1.3 Getting Recommendations
Ask for security improvement suggestions:

**Examples:**
- "What security improvements do you recommend?"
- "How can I improve my compliance score?"
- "Show me the highest priority fixes"
- "What are best practices for my setup?"

### 4.2 Advanced Queries

#### 4.2.1 Complex Analysis
Combine multiple aspects in one query:

**Examples:**
- "Analyze the security of my storage buckets and provide compliance recommendations"
- "Show me compute instances with public IPs and suggest security improvements"
- "Compare the security posture of my dev and prod environments"

#### 4.2.2 Follow-up Questions
Build on previous conversations:

**Examples:**
- "Tell me more about that first finding"
- "How do I implement that recommendation?"
- "What's the priority of these issues?"
- "Show me similar problems in other resources"

### 4.3 Quick Actions

#### 4.3.1 Pre-built Queries
Use quick action buttons for common tasks:

- **🪣 Check Buckets**: Analyze storage bucket security
- **🔐 Review IAM**: Examine identity and access management
- **📋 Check Compliance**: Run compliance assessment
- **🌐 Network Security**: Analyze network configuration
- **💰 Cost Analysis**: Review cost optimization opportunities
- **💡 Get Recommendations**: Show prioritized improvements

#### 4.3.2 Using Quick Actions
1. Click any quick action button
2. The system automatically formulates and sends the query
3. Results appear in the chat interface
4. Follow up with additional questions as needed

## 5. Understanding Results

### 5.1 Asset Information
When you query about assets, you'll see:

#### 5.1.1 Basic Asset Details
- **Name**: Resource identifier
- **Type**: GCP service type
- **Location**: Zone or region
- **Status**: Current operational state
- **Creation Date**: When resource was created

#### 5.1.2 Security Metrics
- **Security Score**: 0-100 rating
- **Risk Level**: LOW, MEDIUM, HIGH, CRITICAL
- **Findings Count**: Number of security issues
- **Compliance Status**: Framework alignment

#### 5.1.3 Recommendations
- **Priority**: HIGH, MEDIUM, LOW
- **Effort**: Implementation difficulty
- **Impact**: Security improvement potential
- **Implementation Steps**: How to fix

### 5.2 Security Findings

#### 5.2.1 Finding Types
- **Configuration Issues**: Misconfigurations
- **Access Control**: Permission problems
- **Encryption**: Data protection gaps
- **Network Security**: Exposure risks
- **Compliance Violations**: Standards non-compliance

#### 5.2.2 Severity Levels
- **🔴 CRITICAL**: Immediate action required
- **🟠 HIGH**: Address within 24 hours
- **🟡 MEDIUM**: Address within 1 week
- **🟢 LOW**: Address when convenient

### 5.3 Recommendations

#### 5.3.1 Recommendation Categories
- **Security Hardening**: Improve defenses
- **Access Control**: Refine permissions
- **Compliance**: Meet standards
- **Cost Optimization**: Reduce expenses
- **Performance**: Improve efficiency

#### 5.3.2 Implementation Guidance
Each recommendation includes:
- **Description**: What needs to be done
- **Rationale**: Why it's important
- **Steps**: How to implement
- **Resources**: Links to documentation
- **Validation**: How to verify success

## 6. Best Practices

### 6.1 Effective Querying

#### 6.1.1 Be Specific
Instead of: "Check security"
Try: "Analyze security of my storage buckets"

#### 6.1.2 Use Context
Follow up questions work well:
- First: "Show me my databases"
- Then: "Which ones have public access?"
- Finally: "How do I secure the public ones?"

#### 6.1.3 Ask for Priorities
- "What's the highest priority security issue?"
- "Which recommendation should I implement first?"
- "What's the biggest risk in my environment?"

### 6.2 Regular Usage Patterns

#### 6.2.1 Weekly Security Check
1. "Give me a security overview"
2. "What new findings do we have?"
3. "Show me this week's recommendations"

#### 6.2.2 Before Major Changes
1. "Analyze current security baseline"
2. "What are potential risks of deployment?"
3. "How will this change affect compliance?"

#### 6.2.3 Incident Investigation
1. "Show me recent security changes"
2. "What assets were modified today?"
3. "Analyze security of affected resources"

### 6.3 Interpreting Results

#### 6.3.1 Security Scores
- **90-100**: Excellent security posture
- **75-89**: Good with minor improvements needed
- **60-74**: Moderate security, several issues to address
- **Below 60**: Poor security, immediate attention required

#### 6.3.2 Prioritizing Actions
1. **Critical Findings**: Address immediately
2. **High-Impact, Low-Effort**: Quick wins
3. **Compliance Requirements**: Regulatory needs
4. **Long-term Improvements**: Strategic enhancements

## 7. Troubleshooting

### 7.1 Common Issues

#### 7.1.1 "No Assets Found"
**Possible Causes:**
- Insufficient permissions
- Wrong project selected
- No resources in project

**Solutions:**
- Verify IAM permissions
- Check project ID
- Ask admin to grant access

#### 7.1.2 Slow Response Times
**Possible Causes:**
- Large number of assets
- Complex analysis requested
- Network latency

**Solutions:**
- Be more specific in queries
- Wait for processing to complete
- Try simpler queries first

#### 7.1.3 "Permission Denied"
**Possible Causes:**
- Missing IAM roles
- Expired credentials
- Project access revoked

**Solutions:**
- Contact your GCP administrator
- Refresh browser and try again
- Verify project permissions

### 7.2 Getting Help

#### 7.2.1 In-App Help
- Type "help" in the chat for basic guidance
- Use "what can you do?" to see capabilities
- Ask "how do I..." for specific guidance

#### 7.2.2 Error Messages
The system provides helpful error messages:
- Read the message carefully
- Follow suggested actions
- Ask follow-up questions for clarification

## 8. Security and Privacy

### 8.1 Data Handling
- **No Data Storage**: Your GCP data stays in your environment
- **Real-time Analysis**: Information is analyzed on-demand
- **Secure Connections**: All communication uses HTTPS
- **Access Logging**: All queries are logged for security

### 8.2 Permissions
- **Read-Only Access**: System only reads, never modifies resources
- **Least Privilege**: Minimal required permissions
- **User Context**: Operates within your existing permissions
- **Audit Trail**: All actions are logged and traceable

### 8.3 Best Security Practices
- **Regular Reviews**: Check permissions quarterly
- **Monitor Usage**: Review access logs regularly
- **Report Issues**: Contact support for security concerns
- **Stay Updated**: Keep informed about security updates

## 9. Tips for Success

### 9.1 Getting the Most Value
1. **Start Broad**: Begin with general security overview
2. **Drill Down**: Focus on specific areas of concern
3. **Take Action**: Implement recommendations promptly
4. **Monitor Progress**: Track improvements over time
5. **Stay Informed**: Regular security health checks

### 9.2 Building Security Habits
- **Daily**: Quick security status check
- **Weekly**: Review new findings and recommendations
- **Monthly**: Comprehensive security analysis
- **Quarterly**: Compliance assessment and planning

### 9.3 Advanced Usage
- **Integration**: Use API for automated security checks
- **Customization**: Configure for your specific needs
- **Collaboration**: Share findings with your team
- **Automation**: Set up regular security reports

## 10. Support and Resources

### 10.1 Getting Support
- **Documentation**: Comprehensive guides and references
- **In-App Help**: Context-sensitive assistance
- **Community**: User forums and discussions
- **Professional Support**: Enterprise support options

### 10.2 Additional Resources
- **Security Best Practices**: Google Cloud security guides
- **Compliance Frameworks**: Standards documentation
- **Training**: Security certification programs
- **Updates**: Release notes and new features

### 10.3 Feedback
Help us improve:
- **Feature Requests**: Suggest new capabilities
- **Bug Reports**: Report issues you encounter
- **Usage Feedback**: Share your experience
- **Success Stories**: Tell us how you've improved security

---

**Remember**: The GCP Security Agent is your intelligent partner in maintaining strong cloud security. Use it regularly, act on its recommendations, and your security posture will continuously improve.