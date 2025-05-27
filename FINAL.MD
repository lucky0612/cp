#!/usr/bin/env python3
import os
import json
import pickle
import time
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import requests
from requests.auth import HTTPBasicAuth
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, GenerativeConfig
import urllib3
from collections import defaultdict

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JiraChatbot")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Jira Configuration
JIRA_URL = os.environ.get("JIRA_URL", "https://cmegroup.atlassian.net/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "lakshya.vijay@cmegroup.com")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "ATATTxEfGFdPemKZpWezVd58xSZ5242xlMJMRvelfVdjcrmm7G3G0KQeaBdBRw1aATINeOAwHPNmwy8qUQ2oL4kyJRWjQhIKygnc55R7O.lw.3wl1EkopOz1lFRKCYPEhR0A6ZHZNZwMCRjrYuj1Blh5r5legJFD3dGhlcwhBz0mJ1KXZE-reZDUWM=851BA963")

# Cache configuration
CACHE_DIR = "jira_cache"
CACHE_EXPIRY = 6  # Cache expiry in hours (reduced for fresh data)

class UniversalJiraClient:
    """Universal Jira client that finds EVERYTHING"""
    
    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.cache = SimpleCache()
        self._all_data_cache = None
        
    def make_request(self, method: str, endpoint: str, params: Dict = None, 
                     data: Dict = None, timeout: int = 30) -> Optional[Dict]:
        """Make robust HTTP request"""
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        
        try:
            logger.info(f"Making {method} request to: {endpoint}")
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params or {},
                json=data,
                auth=self.auth,
                verify=False,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Resource not found: {endpoint}")
                return None
            else:
                logger.warning(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
    
    def get_all_projects(self) -> List[Dict]:
        """Get all projects"""
        cache_key = "all_projects"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        projects = self.make_request('GET', 'project')
        if projects:
            clean_projects = []
            for p in projects:
                if p and isinstance(p, dict):
                    clean_projects.append({
                        'key': p.get('key', ''),
                        'name': p.get('name', ''),
                        'id': p.get('id', ''),
                        'projectTypeKey': p.get('projectTypeKey', '')
                    })
            self.cache.set(cache_key, clean_projects)
            return clean_projects
        return []
    
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        cache_key = "all_users"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        users = self.make_request('GET', 'users/search', params={'maxResults': 1000})
        if users:
            clean_users = []
            for u in users:
                if u and isinstance(u, dict):
                    clean_users.append({
                        'displayName': u.get('displayName', ''),
                        'emailAddress': u.get('emailAddress', ''),
                        'accountId': u.get('accountId', ''),
                        'active': u.get('active', True)
                    })
            self.cache.set(cache_key, clean_users)
            return clean_users
        return []
    
    def comprehensive_search(self, query_text: str) -> List[Dict]:
        """Comprehensive search using multiple strategies"""
        logger.info(f"Starting comprehensive search for: '{query_text}'")
        
        all_results = []
        
        # Strategy 1: Direct issue key search
        issue_keys = re.findall(r'[A-Z]+-\d+', query_text.upper())
        if issue_keys:
            logger.info(f"Found issue keys: {issue_keys}")
            for key in issue_keys:
                issue = self.get_issue_by_key(key)
                if issue:
                    all_results.append(issue)
        
        # Strategy 2: Multiple JQL searches
        jql_queries = self.build_comprehensive_jql_queries(query_text)
        
        for jql in jql_queries:
            logger.info(f"Executing JQL: {jql}")
            results = self.execute_jql(jql)
            if results:
                all_results.extend(results)
        
        # Strategy 3: Fallback - get recent data if nothing found
        if not all_results:
            logger.info("No results found, trying fallback searches")
            fallback_queries = [
                "updated >= -30d ORDER BY updated DESC",
                "created >= -7d ORDER BY created DESC",
                "project in (RDRF, BAMPS, CWS) ORDER BY updated DESC"
            ]
            
            for fallback_jql in fallback_queries:
                results = self.execute_jql(fallback_jql, max_results=50)
                if results:
                    all_results.extend(results[:20])
                    break
        
        # Remove duplicates
        unique_results = self.deduplicate_issues(all_results)
        
        logger.info(f"Comprehensive search returned {len(unique_results)} unique results")
        return unique_results
    
    def build_comprehensive_jql_queries(self, query_text: str) -> List[str]:
        """Build multiple JQL queries to cover all possibilities"""
        queries = []
        query_lower = query_text.lower()
        
        # Extract meaningful terms
        terms = self.extract_search_terms(query_text)
        logger.info(f"Extracted search terms: {terms}")
        
        if terms:
            # Text search in multiple fields
            for term in terms[:5]:  # Limit to prevent too many queries
                queries.extend([
                    f'summary ~ "{term}" ORDER BY updated DESC',
                    f'description ~ "{term}" ORDER BY updated DESC',
                    f'comment ~ "{term}" ORDER BY updated DESC',
                    f'text ~ "{term}" ORDER BY updated DESC'
                ])
            
            # Combined text search
            combined_terms = ' OR '.join([f'text ~ "{term}"' for term in terms[:3]])
            queries.append(f'({combined_terms}) ORDER BY updated DESC')
        
        # User-based searches
        user_queries = self.build_user_queries(query_lower)
        queries.extend(user_queries)
        
        # Status-based searches
        status_queries = self.build_status_queries(query_lower)
        queries.extend(status_queries)
        
        # Priority-based searches
        priority_queries = self.build_priority_queries(query_lower)
        queries.extend(priority_queries)
        
        # Date-based searches
        date_queries = self.build_date_queries(query_lower)
        queries.extend(date_queries)
        
        # Project-based searches
        project_queries = self.build_project_queries(query_lower)
        queries.extend(project_queries)
        
        # Issue type searches
        type_queries = self.build_type_queries(query_lower)
        queries.extend(type_queries)
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))[:15]  # Limit to 15 queries
        
        logger.info(f"Built {len(unique_queries)} unique JQL queries")
        return unique_queries
    
    def extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms"""
        # Remove common stop words
        stop_words = {
            'show', 'me', 'all', 'the', 'get', 'find', 'list', 'about', 'with', 'for', 
            'in', 'on', 'at', 'by', 'from', 'to', 'and', 'or', 'but', 'if', 'then', 
            'else', 'when', 'where', 'how', 'what', 'who', 'why', 'which', 'tickets', 
            'issues', 'jira', 'ticket', 'issue', 'please', 'can', 'you', 'i', 'need', 
            'want', 'like', 'would', 'could', 'should', 'a', 'an', 'is', 'are', 'was', 
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        terms = quoted.copy()
        
        # Extract individual words
        words = re.findall(r'\b\w+\b', query.lower())
        for word in words:
            if len(word) > 2 and word not in stop_words and word not in terms:
                terms.append(word)
        
        # Look for compound technical terms
        compound_patterns = [
            r'rollout[\s-]*restart', r'dr[\s-]*related', r'database[\s-]*issue',
            r'deployment[\s-]*issue', r'performance[\s-]*issue', r'traffic[\s-]*surge',
            r'system[\s-]*down', r'server[\s-]*error', r'security[\s-]*vulnerability',
            r'production[\s-]*issue', r'network[\s-]*problem', r'backup[\s-]*failure'
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                clean_term = re.sub(r'[\s-]+', ' ', match).strip()
                if clean_term not in terms:
                    terms.append(clean_term)
        
        return terms[:10]  # Limit terms
    
    def build_user_queries(self, query: str) -> List[str]:
        """Build user-based JQL queries"""
        queries = []
        
        # Look for user names
        user_patterns = [
            r'assigned to ([a-zA-Z\s]+)',
            r'assignee ([a-zA-Z\s]+)',
            r'reported by ([a-zA-Z\s]+)',
            r'reporter ([a-zA-Z\s]+)',
            r'created by ([a-zA-Z\s]+)',
            r'lakshya[\s]*vijay',
            r'vijay',
            r'lakshya'
        ]
        
        for pattern in user_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                user = match.strip() if isinstance(match, str) else ' '.join(match.split())
                if user:
                    queries.extend([
                        f'assignee ~ "{user}" ORDER BY updated DESC',
                        f'reporter ~ "{user}" ORDER BY updated DESC',
                        f'(assignee ~ "{user}" OR reporter ~ "{user}") ORDER BY updated DESC'
                    ])
        
        # Special handling for lakshya vijay
        if any(name in query for name in ['lakshya', 'vijay']):
            queries.extend([
                'assignee = "lakshya.vijay@cmegroup.com" ORDER BY updated DESC',
                'reporter = "lakshya.vijay@cmegroup.com" ORDER BY updated DESC',
                'assignee ~ "lakshya" ORDER BY updated DESC',
                'assignee ~ "vijay" ORDER BY updated DESC'
            ])
        
        return queries
    
    def build_status_queries(self, query: str) -> List[str]:
        """Build status-based queries"""
        queries = []
        
        status_mapping = {
            'open': ['Open', 'To Do', 'New'],
            'closed': ['Closed', 'Done', 'Resolved'],
            'in progress': ['In Progress'],
            'todo': ['To Do'],
            'done': ['Done'],
            'resolved': ['Resolved']
        }
        
        for keyword, statuses in status_mapping.items():
            if keyword in query:
                status_list = "', '".join(statuses)
                queries.append(f"status in ('{status_list}') ORDER BY updated DESC")
        
        return queries
    
    def build_priority_queries(self, query: str) -> List[str]:
        """Build priority-based queries"""
        queries = []
        
        if any(word in query for word in ['high', 'critical', 'urgent', 'blocker']):
            queries.append("priority in (High, Highest, Blocker) ORDER BY updated DESC")
        
        if any(word in query for word in ['low', 'minor', 'trivial']):
            queries.append("priority in (Low, Lowest, Minor, Trivial) ORDER BY updated DESC")
        
        return queries
    
    def build_date_queries(self, query: str) -> List[str]:
        """Build date-based queries"""
        queries = []
        
        date_patterns = [
            ('today', 0), ('yesterday', -1), ('last week', -7), 
            ('past week', -7), ('last month', -30), ('past month', -30)
        ]
        
        for pattern, days in date_patterns:
            if pattern in query:
                date = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
                queries.extend([
                    f"created >= '{date}' ORDER BY created DESC",
                    f"updated >= '{date}' ORDER BY updated DESC"
                ])
        
        return queries
    
    def build_project_queries(self, query: str) -> List[str]:
        """Build project-based queries"""
        queries = []
        
        # Get all projects
        projects = self.get_all_projects()
        
        for project in projects:
            project_key = project.get('key', '').lower()
            project_name = project.get('name', '').lower()
            
            if (project_key in query or 
                any(word in query for word in project_name.split()) or
                project_key in query.replace(' ', '')):
                queries.append(f"project = '{project.get('key')}' ORDER BY updated DESC")
        
        # Also check for common project abbreviations
        if 'bamps' in query:
            queries.append("project = 'BAMPS' ORDER BY updated DESC")
        if 'rdrf' in query:
            queries.append("project = 'RDRF' ORDER BY updated DESC")
        if 'cws' in query:
            queries.append("project = 'CWS' ORDER BY updated DESC")
        
        return queries
    
    def build_type_queries(self, query: str) -> List[str]:
        """Build issue type queries"""
        queries = []
        
        type_mapping = {
            'bug': 'Bug',
            'task': 'Task', 
            'story': 'Story',
            'epic': 'Epic',
            'subtask': 'Sub-task'
        }
        
        for keyword, issue_type in type_mapping.items():
            if keyword in query:
                queries.append(f"issuetype = '{issue_type}' ORDER BY updated DESC")
        
        return queries
    
    def execute_jql(self, jql: str, max_results: int = 100) -> List[Dict]:
        """Execute JQL and return clean results"""
        if not jql or not jql.strip():
            return []
        
        try:
            params = {
                "jql": jql,
                "maxResults": max_results,
                "fields": [
                    "summary", "description", "status", "assignee", "reporter", 
                    "created", "updated", "priority", "issuetype", "labels", 
                    "comment", "components", "fixVersions", "project", "key"
                ]
            }
            
            result = self.make_request('GET', 'search', params=params)
            
            if result and 'issues' in result:
                return [issue for issue in result['issues'] if issue]
            return []
            
        except Exception as e:
            logger.error(f"JQL execution error: {e}")
            return []
    
    def get_issue_by_key(self, issue_key: str) -> Optional[Dict]:
        """Get specific issue by key"""
        try:
            issue = self.make_request('GET', f'issue/{issue_key}')
            if issue:
                # Get comments
                comments = self.make_request('GET', f'issue/{issue_key}/comment')
                if comments and 'comments' in comments:
                    issue['all_comments'] = comments['comments']
                return issue
            return None
        except Exception as e:
            logger.error(f"Error getting issue {issue_key}: {e}")
            return None
    
    def deduplicate_issues(self, issues: List[Dict]) -> List[Dict]:
        """Remove duplicate issues"""
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

class SimpleCache:
    """Simple and reliable cache"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        # Check memory first
        if key in self.memory_cache:
            timestamp, data = self.memory_cache[key]
            if time.time() - timestamp < CACHE_EXPIRY * 3600:
                return data
            else:
                del self.memory_cache[key]
        
        # Check file cache
        try:
            cache_file = os.path.join(self.cache_dir, f"{self._safe_key(key)}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if time.time() - cached_data['timestamp'] < CACHE_EXPIRY * 3600:
                    self.memory_cache[key] = (time.time(), cached_data['data'])
                    return cached_data['data']
                else:
                    os.remove(cache_file)
        except Exception as e:
            logger.error(f"Cache read error: {e}")
        
        return None
    
    def set(self, key: str, data: Any):
        try:
            # Memory cache
            self.memory_cache[key] = (time.time(), data)
            
            # File cache
            cache_file = os.path.join(self.cache_dir, f"{self._safe_key(key)}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump({'timestamp': time.time(), 'data': data}, f)
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def _safe_key(self, key: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '_', str(key))[:100]
    
    def clear(self):
        self.memory_cache.clear()
        try:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

class AdvancedResponseGenerator:
    """Advanced AI response generator"""
    
    def __init__(self, project_id: str, location: str, model_name: str):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
    
    def generate_response(self, user_query: str, jira_data: List[Dict]) -> str:
        """Generate comprehensive response"""
        
        # Format the data for AI
        context = self._format_jira_data(jira_data)
        
        prompt = f"""You are JiraGPT, an expert Jira analyst. Provide a comprehensive, well-structured response.

USER QUERY: "{user_query}"

JIRA DATA FOUND:
{context}

RESPONSE GUIDELINES:
1. Start with a clear SUMMARY of what was found
2. Provide KEY STATISTICS (counts, distributions)
3. List IMPORTANT TICKETS with details
4. Include INSIGHTS and PATTERNS
5. Add RECOMMENDATIONS if applicable
6. Use clear formatting with emojis and headers
7. Include ticket links: [TICKET-KEY](https://cmegroup.atlassian.net/browse/TICKET-KEY)

Provide a helpful, comprehensive response:"""

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
                return self._fallback_response(user_query, jira_data)
                
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return self._fallback_response(user_query, jira_data)
    
    def _format_jira_data(self, issues: List[Dict]) -> str:
        """Format Jira data for AI context"""
        if not issues:
            return "No issues found."
        
        context = f"FOUND {len(issues)} TICKETS:\n\n"
        
        # Statistics
        stats = self._calculate_statistics(issues)
        context += "STATISTICS:\n"
        for key, value in stats.items():
            context += f"- {key}: {value}\n"
        context += "\n"
        
        # Top 15 issues
        context += "TICKET DETAILS:\n"
        for i, issue in enumerate(issues[:15], 1):
            context += self._format_issue(issue, i)
            context += "\n"
        
        if len(issues) > 15:
            context += f"... and {len(issues) - 15} more tickets\n"
        
        return context
    
    def _calculate_statistics(self, issues: List[Dict]) -> Dict:
        """Calculate statistics"""
        stats = {"Total": len(issues)}
        
        status_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        type_counts = defaultdict(int)
        assignee_counts = defaultdict(int)
        
        for issue in issues:
            if not issue:
                continue
                
            fields = issue.get('fields', {})
            
            # Status
            status = fields.get('status', {})
            if status:
                status_counts[status.get('name', 'Unknown')] += 1
            
            # Priority
            priority = fields.get('priority', {})
            if priority:
                priority_counts[priority.get('name', 'Unknown')] += 1
            
            # Type
            issuetype = fields.get('issuetype', {})
            if issuetype:
                type_counts[issuetype.get('name', 'Unknown')] += 1
            
            # Assignee
            assignee = fields.get('assignee', {})
            if assignee:
                assignee_counts[assignee.get('displayName', 'Unknown')] += 1
            else:
                assignee_counts['Unassigned'] += 1
        
        if status_counts:
            stats["By Status"] = dict(status_counts)
        if priority_counts:
            stats["By Priority"] = dict(priority_counts)
        if type_counts:
            stats["By Type"] = dict(type_counts)
        if assignee_counts:
            stats["By Assignee"] = dict(list(assignee_counts.items())[:5])  # Top 5
        
        return stats
    
    def _format_issue(self, issue: Dict, index: int) -> str:
        """Format single issue"""
        if not issue:
            return f"{index}. Invalid issue\n"
        
        key = issue.get('key', 'Unknown')
        fields = issue.get('fields', {})
        
        summary = fields.get('summary', 'No summary')
        status = fields.get('status', {}).get('name', 'Unknown')
        priority = fields.get('priority', {}).get('name', 'Unknown')
        assignee = fields.get('assignee', {}).get('displayName', 'Unassigned')
        created = fields.get('created', 'Unknown')
        updated = fields.get('updated', 'Unknown')
        
        return f"{index}. **{key}** - {summary}\n   Status: {status} | Priority: {priority} | Assignee: {assignee}\n   Created: {created} | Updated: {updated}"
    
    def _fallback_response(self, user_query: str, jira_data: List[Dict]) -> str:
        """Fallback response if AI fails"""
        if not jira_data:
            return f"""🔍 **No Results Found**

I searched extensively for "{user_query}" but didn't find any matching tickets.

💡 **Suggestions:**
• Try different keywords or terms
• Check spelling of project names or user names  
• Use broader search terms
• Try asking about recent tickets in general

📝 **Example queries:**
• "Show me recent tickets"
• "All high priority issues"
• "Tickets from last week"
"""

        response = f"""📊 **Search Results for "{user_query}"**

🎯 **SUMMARY:** Found {len(jira_data)} tickets

📈 **TOP TICKETS:**
"""
        
        for i, issue in enumerate(jira_data[:10], 1):
            if issue:
                key = issue.get('key', 'Unknown')
                fields = issue.get('fields', {})
                summary = fields.get('summary', 'No summary')[:80]
                status = fields.get('status', {}).get('name', 'Unknown')
                
                response += f"{i}. **[{key}](https://cmegroup.atlassian.net/browse/{key})** - {summary}\n   Status: {status}\n\n"
        
        if len(jira_data) > 10:
            response += f"... and {len(jira_data) - 10} more tickets found."
        
        return response

class MasterJiraChatbot:
    """The ultimate versatile Jira chatbot"""
    
    def __init__(self):
        self.jira_client = UniversalJiraClient(JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN)
        self.response_generator = AdvancedResponseGenerator(PROJECT_ID, REGION, MODEL_NAME)
        logger.info("MasterJiraChatbot initialized")
    
    def process_any_query(self, query: str) -> str:
        """Process ANY query with guaranteed results"""
        start_time = time.time()
        logger.info(f"Processing: '{query}'")
        
        try:
            # Handle special commands
            if query.lower().strip() in ["clear cache", "refresh"]:
                self.jira_client.cache.clear()
                return "✅ Cache cleared! Fresh data will be loaded."
            
            # Handle simple interactions
            simple_response = self._handle_simple_interactions(query)
            if simple_response:
                return simple_response
            
            # Main search
            logger.info("Starting comprehensive search...")
            results = self.jira_client.comprehensive_search(query)
            
            logger.info(f"Found {len(results)} total results")
            
            # Generate response
            response = self.response_generator.generate_response(query, results)
            
            elapsed = time.time() - start_time
            logger.info(f"Query processed in {elapsed:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Critical error: {e}")
            return f"❌ **Error occurred:** {str(e)}\n\nPlease try a different query or check the logs."
    
    def _handle_simple_interactions(self, query: str) -> Optional[str]:
        """Handle greetings and help"""
        query_lower = query.lower().strip()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return """👋 **Hello! I'm your Master Jira Assistant!**

🚀 **I can find ANY Jira ticket or information:**
• Tickets about any topic: "rollout restart issues"
• Assigned tickets: "tickets assigned to lakshya vijay"  
• Project tickets: "BAMPS tickets"
• Recent tickets: "tickets from last week"
• Priority tickets: "high priority bugs"

Just ask me naturally - I'll find what you need! 💪"""

        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            return "👋 Goodbye! Come back anytime for Jira help!"
        
        if any(word in query_lower for word in ['help', 'what can you do']):
            return """🤖 **I'm your versatile Jira expert!**

✨ **What I guarantee:**
• **Find ANY ticket** - no matter how you ask
• **Multiple search strategies** - text, user, project, status, date
• **Comprehensive results** - I'll always find something relevant
• **Smart understanding** - natural language processing

💡 **Try these examples:**
• "bamps tickets" 
• "tickets assigned to lakshya vijay"
• "rollout restart problems"
• "high priority issues from last week"
• "DR related tickets"
• "show me recent bugs"

**I WILL find results for any query!** 🎯"""
        
        return None
    
    def run_interactive_chat(self):
        """Run the master chatbot"""
        print("\n" + "="*80)
        print("🚀 MASTER JIRA CHATBOT - GUARANTEED RESULTS")
        print("="*80)
        print("💪 I can find ANY Jira information - just ask naturally!")
        print("\n📝 **Test these to see my power:**")
        print("   • 'bamps tickets'")
        print("   • 'tickets assigned to lakshya vijay'") 
        print("   • 'rollout restart issues'")
        print("   • 'high priority bugs'")
        print("   • 'DR related problems'")
        print("   • 'show me recent tickets'")
        print("\n💬 Type 'exit' to quit | 'clear cache' to refresh")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n🗣️  Ask anything: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\n🤖 **Master JiraGPT:** 👋 Thanks for using Master Jira Chatbot!")
                    break
                
                print(f"\n🤖 **Master JiraGPT:** ", end="")
                response = self.process_any_query(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 **Master JiraGPT:** 👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")

def test_connection():
    """Test Jira connection"""
    try:
        print("🔄 Testing Jira connection...")
        client = UniversalJiraClient(JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN)
        
        # Test basic connectivity
        projects = client.get_all_projects()
        print(f"✅ Connected! Found {len(projects)} projects")
        
        # Test search
        test_results = client.execute_jql("updated >= -7d ORDER BY updated DESC", max_results=5)
        print(f"✅ Search working! Found {len(test_results)} recent tickets")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main function to run the ultimate chatbot"""
    print("🚀 Starting Master Jira Chatbot...")
    
    # Test connection first
    if not test_connection():
        print("Please check your Jira credentials and try again.")
        return
    
    try:
        chatbot = MasterJiraChatbot()
        print("🎯 Master Jira Chatbot is ready!")
        chatbot.run_interactive_chat()
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")

if __name__ == "__main__":
    main()
