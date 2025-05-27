#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
from requests.auth import HTTPBasicAuth
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_ai_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JiraAIChatbot")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Jira Configuration
JIRA_URL = os.environ.get("JIRA_URL", "https://cmegroup.atlassian.net/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "lakshya.vijay@cmegroup.com")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "ATATTxEfGFdPemKZpWezVd58xSZ5242xlMJMRvelfVdjcrmm7G3G0KQeaBdBRw1aATINeOAwHPNmwy8qUQ2oL4kyJRWjQhIKygnc55R7O.lw.3wl1EkopOz1lFRKCYPEhR0A6ZHZNZwMCRjrYuj1Blh5r5legJFD3dGhlcwhBz0mJ1KXZE-reZDUWM=851BA963")

class SafeJiraClient:
    """Ultra-safe Jira client with bulletproof error handling"""
    
    def __init__(self):
        self.base_url = JIRA_URL.rstrip('/')
        self.auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self._metadata = None
        logger.info("Jira client initialized")
    
    def safe_get(self, obj, key, default=None):
        """Ultra-safe getter that never fails"""
        try:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            if hasattr(obj, key):
                return getattr(obj, key, default)
            return default
        except:
            return default
    
    def make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make safe HTTP request"""
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        
        try:
            logger.info(f"Making request to: {endpoint}")
            response = requests.get(
                url,
                headers=self.headers,
                params=params or {},
                auth=self.auth,
                verify=False,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    def get_jira_metadata(self) -> Dict:
        """Get Jira metadata for AI context"""
        if self._metadata:
            return self._metadata
        
        logger.info("Loading Jira metadata...")
        metadata = {
            'projects': [],
            'users': [],
            'statuses': [],
            'priorities': [],
            'issue_types': []
        }
        
        try:
            # Get projects
            projects = self.make_request('project') or []
            for project in projects:
                if project:
                    metadata['projects'].append({
                        'key': self.safe_get(project, 'key', ''),
                        'name': self.safe_get(project, 'name', '')
                    })
            
            # Get sample users
            users = self.make_request('users/search', {'maxResults': 50}) or []
            for user in users:
                if user:
                    metadata['users'].append({
                        'displayName': self.safe_get(user, 'displayName', ''),
                        'emailAddress': self.safe_get(user, 'emailAddress', '')
                    })
            
            # Get statuses
            statuses = self.make_request('status') or []
            for status in statuses:
                if status:
                    metadata['statuses'].append(self.safe_get(status, 'name', ''))
            
            # Get priorities
            priorities = self.make_request('priority') or []
            for priority in priorities:
                if priority:
                    metadata['priorities'].append(self.safe_get(priority, 'name', ''))
            
            # Get issue types
            issue_types = self.make_request('issuetype') or []
            for issue_type in issue_types:
                if issue_type:
                    metadata['issue_types'].append(self.safe_get(issue_type, 'name', ''))
            
            self._metadata = metadata
            logger.info(f"Loaded metadata: {len(metadata['projects'])} projects, {len(metadata['users'])} users")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
        
        return metadata
    
    def execute_jql(self, jql: str, max_results: int = 100) -> List[Dict]:
        """Execute JQL query safely"""
        if not jql or not jql.strip():
            return []
        
        try:
            logger.info(f"Executing JQL: {jql}")
            params = {
                "jql": jql,
                "maxResults": max_results,
                "fields": "*all"
            }
            
            result = self.make_request('search', params)
            if result:
                issues = self.safe_get(result, 'issues', [])
                logger.info(f"JQL returned {len(issues)} issues")
                return [issue for issue in issues if issue]
            
            return []
            
        except Exception as e:
            logger.error(f"JQL execution error: {e}")
            return []
    
    def get_issue_details(self, issue_key: str) -> Optional[Dict]:
        """Get detailed issue information"""
        try:
            issue = self.make_request(f'issue/{issue_key}')
            if issue:
                # Get comments
                comments = self.make_request(f'issue/{issue_key}/comment')
                if comments:
                    issue['all_comments'] = self.safe_get(comments, 'comments', [])
                return issue
            return None
        except Exception as e:
            logger.error(f"Error getting issue {issue_key}: {e}")
            return None

class AIJiraAnalyzer:
    """AI-powered Jira query analyzer and response generator"""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info("AI analyzer initialized")
    
    def analyze_query_and_build_jql(self, user_query: str, metadata: Dict) -> List[str]:
        """Let AI analyze the query and build perfect JQL"""
        
        context = f"""
AVAILABLE JIRA PROJECTS: {[p['key'] + ' (' + p['name'] + ')' for p in metadata.get('projects', [])]}
AVAILABLE STATUSES: {metadata.get('statuses', [])}
AVAILABLE PRIORITIES: {metadata.get('priorities', [])}
AVAILABLE ISSUE TYPES: {metadata.get('issue_types', [])}
SAMPLE USERS: {[u['displayName'] + ' (' + u['emailAddress'] + ')' for u in metadata.get('users', [])[:10]]}
"""

        prompt = f"""You are a Jira JQL expert. Analyze this user query and create 3-5 different JQL queries that will find what they're looking for.

USER QUERY: "{user_query}"

JIRA CONTEXT:
{context}

IMPORTANT JQL GUIDELINES:
1. Use text search: text ~ "term" (searches all fields)
2. Use summary search: summary ~ "term" 
3. Use description search: description ~ "term"
4. Use comment search: comment ~ "term"
5. For users, try both displayName and email patterns
6. For dates, use formats like: created >= -7d, updated >= "2024-01-01"
7. Always end with: ORDER BY updated DESC
8. Use AND, OR operators appropriately
9. For project search: project = "KEY" or project in ("KEY1", "KEY2")
10. For assignee: assignee ~ "name" or assignee = "email"

EXAMPLES:
- "rollout restart issues" → text ~ "rollout restart" OR summary ~ "rollout restart"
- "tickets assigned to john" → assignee ~ "john" OR assignee ~ "John"
- "high priority bugs" → priority = "High" AND issuetype = "Bug"
- "bamps project tickets" → project = "BAMPS"
- "recent issues" → updated >= -7d

CREATE MULTIPLE JQL QUERIES:
1. Main search query (most specific)
2. Broader search query (less specific) 
3. Fallback search query (very broad)
4. Alternative search approach
5. Recent items query (if no specific criteria)

OUTPUT FORMAT - Return only JQL queries, one per line:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )
            
            if hasattr(response, 'text'):
                jql_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                jql_text = response.candidates[0].text
            else:
                return self._fallback_jql(user_query)
            
            # Parse JQL queries
            jql_queries = []
            for line in jql_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Clean up the line
                    line = line.replace('```', '').replace('sql', '').replace('jql', '')
                    line = line.strip('- ').strip('* ').strip('1. ').strip('2. ').strip('3. ').strip('4. ').strip('5. ')
                    
                    if len(line) > 10 and ('=' in line or '~' in line or 'ORDER BY' in line):
                        jql_queries.append(line.strip())
            
            if jql_queries:
                logger.info(f"AI generated {len(jql_queries)} JQL queries")
                return jql_queries[:5]  # Limit to 5 queries
            else:
                return self._fallback_jql(user_query)
                
        except Exception as e:
            logger.error(f"AI JQL generation error: {e}")
            return self._fallback_jql(user_query)
    
    def _fallback_jql(self, user_query: str) -> List[str]:
        """Simple fallback JQL if AI fails"""
        # Extract words from query
        words = [w for w in user_query.lower().split() if len(w) > 2]
        
        queries = []
        if words:
            # Text search with main terms
            main_terms = ' '.join(words[:3])
            queries.append(f'text ~ "{main_terms}" ORDER BY updated DESC')
            
            # Individual word searches
            for word in words[:2]:
                queries.append(f'summary ~ "{word}" ORDER BY updated DESC')
        
        # Always add a recent items fallback
        queries.append('updated >= -30d ORDER BY updated DESC')
        
        return queries
    
    def generate_response(self, user_query: str, jira_data: List[Dict]) -> str:
        """Generate comprehensive response from Jira data"""
        
        # Format data for AI
        formatted_data = self._format_data_for_ai(jira_data)
        
        prompt = f"""You are JiraGPT, an expert Jira analyst. Provide a comprehensive response to the user's query based on the Jira data found.

USER QUERY: "{user_query}"

JIRA DATA FOUND:
{formatted_data}

RESPONSE REQUIREMENTS:
1. Start with a clear SUMMARY of what was found
2. Provide KEY STATISTICS (counts, distributions)
3. List TOP IMPORTANT TICKETS with details
4. Include INSIGHTS and PATTERNS you observe
5. Add ACTIONABLE RECOMMENDATIONS if applicable
6. Use emojis and clear formatting for readability
7. Include ticket links: [TICKET-KEY](https://cmegroup.atlassian.net/browse/TICKET-KEY)
8. If no data found, provide helpful suggestions

Make it comprehensive, professional, and actionable:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4000,
                )
            )
            
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].text
            else:
                return self._create_simple_response(user_query, jira_data)
                
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return self._create_simple_response(user_query, jira_data)
    
    def _format_data_for_ai(self, issues: List[Dict]) -> str:
        """Format Jira data for AI consumption"""
        if not issues:
            return "No Jira issues found."
        
        formatted = f"TOTAL ISSUES FOUND: {len(issues)}\n\n"
        
        # Add statistics
        stats = self._calculate_basic_stats(issues)
        formatted += "STATISTICS:\n"
        for key, value in stats.items():
            formatted += f"- {key}: {value}\n"
        formatted += "\n"
        
        # Add issue details (limit to first 20 for context)
        formatted += "ISSUE DETAILS:\n"
        for i, issue in enumerate(issues[:20], 1):
            if issue:
                formatted += self._format_single_issue(issue, i)
                formatted += "\n"
        
        if len(issues) > 20:
            formatted += f"... and {len(issues) - 20} more issues\n"
        
        return formatted
    
    def _calculate_basic_stats(self, issues: List[Dict]) -> Dict:
        """Calculate basic statistics safely"""
        stats = {"Total Issues": len(issues)}
        
        try:
            status_counts = {}
            priority_counts = {}
            type_counts = {}
            assignee_counts = {}
            
            for issue in issues:
                if not issue:
                    continue
                
                # Safe field extraction
                fields = issue.get('fields') or {}
                
                # Status
                status = fields.get('status') or {}
                status_name = status.get('name', 'Unknown')
                status_counts[status_name] = status_counts.get(status_name, 0) + 1
                
                # Priority
                priority = fields.get('priority') or {}
                priority_name = priority.get('name', 'Unknown')
                priority_counts[priority_name] = priority_counts.get(priority_name, 0) + 1
                
                # Issue Type
                issuetype = fields.get('issuetype') or {}
                type_name = issuetype.get('name', 'Unknown')
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                
                # Assignee
                assignee = fields.get('assignee') or {}
                assignee_name = assignee.get('displayName', 'Unassigned')
                assignee_counts[assignee_name] = assignee_counts.get(assignee_name, 0) + 1
            
            if status_counts:
                stats["Status Distribution"] = status_counts
            if priority_counts:
                stats["Priority Distribution"] = priority_counts
            if type_counts:
                stats["Type Distribution"] = type_counts
            if assignee_counts:
                # Show top 5 assignees
                top_assignees = dict(sorted(assignee_counts.items(), key=lambda x: x[1], reverse=True)[:5])
                stats["Top Assignees"] = top_assignees
                
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
        
        return stats
    
    def _format_single_issue(self, issue: Dict, index: int) -> str:
        """Format single issue safely"""
        try:
            key = issue.get('key', 'Unknown')
            fields = issue.get('fields') or {}
            
            summary = fields.get('summary', 'No summary')
            status = (fields.get('status') or {}).get('name', 'Unknown')
            priority = (fields.get('priority') or {}).get('name', 'Unknown')
            assignee = (fields.get('assignee') or {}).get('displayName', 'Unassigned')
            created = fields.get('created', 'Unknown')
            updated = fields.get('updated', 'Unknown')
            
            return f"{index}. **{key}** - {summary[:100]}\n   Status: {status} | Priority: {priority} | Assignee: {assignee}\n   Created: {created[:10]} | Updated: {updated[:10]}"
            
        except Exception as e:
            logger.error(f"Error formatting issue: {e}")
            return f"{index}. Error formatting issue"
    
    def _create_simple_response(self, user_query: str, jira_data: List[Dict]) -> str:
        """Create simple response if AI fails"""
        if not jira_data:
            return f"""🔍 **No Results Found for "{user_query}"**

I searched comprehensively but couldn't find matching tickets.

💡 **Try:**
• Different keywords
• Broader search terms  
• Check spelling of names/projects
• Ask for "recent tickets" to see what's available

📝 **Examples:**
• "show me recent tickets"
• "high priority issues"  
• "tickets from last week"
"""

        # Create basic response
        response = f"""📊 **Results for "{user_query}"**

🎯 **Found {len(jira_data)} tickets**

📋 **Top Tickets:**
"""
        
        for i, issue in enumerate(jira_data[:10], 1):
            if issue:
                key = issue.get('key', 'Unknown')
                fields = issue.get('fields') or {}
                summary = fields.get('summary', 'No summary')[:80]
                status = (fields.get('status') or {}).get('name', 'Unknown')
                
                response += f"{i}. **[{key}](https://cmegroup.atlassian.net/browse/{key})** - {summary}\n   Status: {status}\n\n"
        
        if len(jira_data) > 10:
            response += f"... and {len(jira_data) - 10} more tickets found."
        
        return response

class TrulyVersatileJiraChatbot:
    """AI-first, truly versatile Jira chatbot"""
    
    def __init__(self):
        self.jira_client = SafeJiraClient()
        self.ai_analyzer = AIJiraAnalyzer()
        logger.info("Versatile Jira chatbot initialized")
    
    def answer_any_question(self, user_query: str) -> str:
        """Answer ANY Jira question using pure AI analysis"""
        start_time = datetime.now()
        logger.info(f"Processing: '{user_query}'")
        
        try:
            # Handle basic interactions
            basic_response = self._handle_basic_interactions(user_query)
            if basic_response:
                return basic_response
            
            # Get Jira metadata for context
            logger.info("Loading Jira metadata...")
            metadata = self.jira_client.get_jira_metadata()
            
            # Let AI analyze the query and build JQL
            logger.info("AI analyzing query and building JQL...")
            jql_queries = self.ai_analyzer.analyze_query_and_build_jql(user_query, metadata)
            
            # Execute all JQL queries
            all_results = []
            for jql in jql_queries:
                results = self.jira_client.execute_jql(jql, max_results=50)
                all_results.extend(results)
            
            # Remove duplicates
            unique_results = self._deduplicate_by_key(all_results)
            
            logger.info(f"Found {len(unique_results)} unique results")
            
            # Let AI generate the response
            logger.info("AI generating response...")
            response = self.ai_analyzer.generate_response(user_query, unique_results)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Query completed in {elapsed:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in answer_any_question: {e}")
            return f"❌ **Error:** {str(e)}\n\nPlease try rephrasing your question."
    
    def _deduplicate_by_key(self, issues: List[Dict]) -> List[Dict]:
        """Remove duplicate issues by key"""
        seen_keys = set()
        unique_issues = []
        
        for issue in issues:
            if not issue:
                continue
            key = issue.get('key')
            if key and key not in seen_keys:
                seen_keys.add(key)
                unique_issues.append(issue)
        
        return unique_issues
    
    def _handle_basic_interactions(self, query: str) -> Optional[str]:
        """Handle basic greetings and help"""
        query_lower = query.lower().strip()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return """👋 **Hello! I'm your AI-powered Jira assistant!**

🧠 **I use pure AI to understand ANY question about Jira:**
• Just ask naturally - no special syntax needed
• I'll analyze your question and build perfect search queries
• Get comprehensive results with insights and statistics

💡 **Try asking:**
• "bamps tickets"
• "tickets assigned to lakshya vijay"  
• "rollout restart issues"
• "high priority bugs from last week"

Ask me anything about your Jira data! 🚀"""

        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            return "👋 Goodbye! Come back anytime for Jira insights!"
        
        if any(word in query_lower for word in ['help', 'what can you do']):
            return """🤖 **I'm your AI-powered Jira expert!**

✨ **How I work:**
1. **You ask anything** in natural language
2. **AI analyzes** your question intelligently  
3. **AI builds** perfect JQL queries automatically
4. **AI generates** comprehensive responses with insights

🎯 **I guarantee results** for any Jira question!

💡 **No need to learn JQL or special syntax - just ask naturally!**"""
        
        return None
    
    def run_chat(self):
        """Run the interactive chatbot"""
        print("\n" + "="*80)
        print("🧠 AI-POWERED VERSATILE JIRA CHATBOT")
        print("="*80)
        print("🚀 Pure AI analysis - Ask ANY question about Jira in natural language!")
        print("\n💡 **How it works:**")
        print("   1. AI analyzes your question")
        print("   2. AI builds perfect JQL queries") 
        print("   3. AI generates comprehensive responses")
        print("\n📝 **Test with:**")
        print("   • 'bamps tickets'")
        print("   • 'tickets assigned to lakshya vijay'")
        print("   • 'rollout restart issues'")
        print("   • 'recent high priority bugs'")
        print("\n💬 Type 'exit' to quit")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n🗣️  Ask anything: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\n🤖 **AI JiraGPT:** 👋 Thanks for using AI-powered Jira chatbot!")
                    break
                
                print(f"\n🤖 **AI JiraGPT:** ", end="")
                response = self.answer_any_question(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 **AI JiraGPT:** 👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def test_system():
    """Test the system"""
    try:
        print("🔄 Testing AI-powered Jira system...")
        
        # Test Jira connection
        client = SafeJiraClient()
        metadata = client.get_jira_metadata()
        print(f"✅ Jira connected: {len(metadata['projects'])} projects found")
        
        # Test AI
        analyzer = AIJiraAnalyzer()
        test_jql = analyzer.analyze_query_and_build_jql("test query", metadata)
        print(f"✅ AI working: Generated {len(test_jql)} JQL queries")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main function"""
    print("🧠 Starting AI-Powered Versatile Jira Chatbot...")
    
    if not test_system():
        print("Please check your configuration and try again.")
        return
    
    try:
        chatbot = TrulyVersatileJiraChatbot()
        print("🎯 AI-powered chatbot ready!")
        chatbot.run_chat()
    except Exception as e:
        print(f"❌ Failed to start: {e}")

if __name__ == "__main__":
    main()
