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
from vertexai.generative_models import GenerativeModel, GenerativeConfig, Part, Content
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
CACHE_EXPIRY = 24  # Cache expiry in hours

class EnhancedJiraClient:
    """Enhanced Jira client with intelligent querying capabilities"""
    
    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.cache = JiraCache()
        self._projects_cache = None
        self._users_cache = None
        
    def make_request(self, method: str, endpoint: str, params: Dict = None, 
                     data: Dict = None, cache_key: str = None, force_refresh: bool = False) -> Dict:
        """Make HTTP request with enhanced error handling and caching"""
        if method.lower() == 'get' and cache_key and not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Using cached data for: {cache_key}")
                return cached_data
        
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                auth=self.auth,
                verify=False,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if method.lower() == 'get' and cache_key:
                self.cache.set(cache_key, result)
                
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error for {endpoint}: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {}
    
    def get_all_projects(self) -> List[Dict]:
        """Get all available projects"""
        if self._projects_cache is None:
            cache_key = "all_projects"
            projects = self.make_request('GET', 'project', cache_key=cache_key)
            self._projects_cache = projects if isinstance(projects, list) else []
        return self._projects_cache
    
    def get_all_users(self) -> List[Dict]:
        """Get all users (limited to first 1000)"""
        if self._users_cache is None:
            cache_key = "all_users"
            users = self.make_request('GET', 'users/search', params={'maxResults': 1000}, cache_key=cache_key)
            self._users_cache = users if isinstance(users, list) else []
        return self._users_cache
    
    def intelligent_search(self, query_text: str, max_results: int = 100) -> List[Dict]:
        """Perform intelligent search across all Jira data"""
        logger.info(f"Performing intelligent search for: {query_text}")
        
        # Get all projects to search across
        projects = self.get_all_projects()
        all_issues = []
        
        # Build comprehensive JQL queries
        jql_queries = self._build_intelligent_jql(query_text, projects)
        
        for jql in jql_queries:
            try:
                logger.info(f"Executing JQL: {jql}")
                params = {
                    "jql": jql,
                    "maxResults": max_results,
                    "fields": [
                        "summary", "description", "status", "assignee", "reporter", 
                        "created", "updated", "priority", "issuetype", "labels", 
                        "comment", "components", "fixVersions", "versions", 
                        "resolution", "resolutiondate", "project"
                    ]
                }
                
                cache_key = f"search_{hash(jql)}_{max_results}"
                result = self.make_request('GET', 'search', params=params, cache_key=cache_key)
                
                if result.get('issues'):
                    all_issues.extend(result['issues'])
                    
            except Exception as e:
                logger.error(f"Error executing JQL {jql}: {e}")
                continue
        
        # Remove duplicates based on issue key
        seen_keys = set()
        unique_issues = []
        for issue in all_issues:
            key = issue.get('key')
            if key and key not in seen_keys:
                seen_keys.add(key)
                unique_issues.append(issue)
        
        # Sort by relevance (updated date desc, then created date desc)
        unique_issues.sort(key=lambda x: (
            x.get('fields', {}).get('updated', ''),
            x.get('fields', {}).get('created', '')
        ), reverse=True)
        
        return unique_issues[:max_results]
    
    def _build_intelligent_jql(self, query_text: str, projects: List[Dict]) -> List[str]:
        """Build multiple JQL queries to capture different aspects of the search"""
        query_lower = query_text.lower()
        jql_queries = []
        
        # Extract specific terms and patterns
        specific_terms = self._extract_search_terms(query_text)
        
        # Query 1: Text search across summary, description, and comments
        if specific_terms:
            text_conditions = []
            for term in specific_terms:
                text_conditions.append(f'(summary ~ "{term}" OR description ~ "{term}" OR comment ~ "{term}")')
            
            if text_conditions:
                jql_queries.append(" OR ".join(text_conditions) + " ORDER BY updated DESC")
        
        # Query 2: Status-based search
        status_mapping = {
            'open': ['Open', 'To Do', 'New', 'Reopened'],
            'closed': ['Closed', 'Done', 'Resolved'],
            'in progress': ['In Progress', 'In Review'],
            'todo': ['To Do', 'Open'],
            'done': ['Done', 'Closed', 'Resolved'],
            'resolved': ['Resolved', 'Done']
        }
        
        for status_key, statuses in status_mapping.items():
            if status_key in query_lower:
                status_conditions = " OR ".join([f'status = "{s}"' for s in statuses])
                jql_queries.append(f"({status_conditions}) ORDER BY updated DESC")
        
        # Query 3: Priority-based search
        if any(p in query_lower for p in ['high', 'critical', 'blocker', 'urgent']):
            jql_queries.append('priority in (High, Highest, Blocker) ORDER BY updated DESC')
        elif any(p in query_lower for p in ['low', 'minor', 'trivial']):
            jql_queries.append('priority in (Low, Lowest, Minor, Trivial) ORDER BY updated DESC')
        
        # Query 4: Date-based search
        date_jql = self._build_date_jql(query_lower)
        if date_jql:
            jql_queries.append(date_jql)
        
        # Query 5: Issue type search
        issue_types = ['bug', 'task', 'story', 'epic', 'subtask']
        for issue_type in issue_types:
            if issue_type in query_lower:
                jql_queries.append(f'issuetype = "{issue_type.title()}" ORDER BY updated DESC')
        
        # Query 6: User-based search (assignee/reporter)
        user_jql = self._build_user_jql(query_lower)
        if user_jql:
            jql_queries.extend(user_jql)
        
        # Query 7: Project-specific search
        project_jql = self._build_project_jql(query_lower, projects)
        if project_jql:
            jql_queries.extend(project_jql)
        
        # Query 8: Generic search if no specific patterns found
        if not jql_queries and specific_terms:
            generic_terms = " OR ".join([f'text ~ "{term}"' for term in specific_terms[:3]])
            jql_queries.append(f"({generic_terms}) ORDER BY updated DESC")
        
        # Fallback: Get recent issues if nothing else matches
        if not jql_queries:
            jql_queries.append("updated >= -7d ORDER BY updated DESC")
        
        return jql_queries
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms from the query"""
        # Remove common stop words
        stop_words = {
            'show', 'get', 'find', 'list', 'tell', 'me', 'about', 'all', 'the', 
            'with', 'for', 'of', 'in', 'on', 'at', 'by', 'from', 'to', 'and',
            'or', 'but', 'if', 'then', 'else', 'when', 'where', 'how', 'what',
            'who', 'why', 'which', 'tickets', 'issues', 'jira', 'ticket', 'issue',
            'please', 'can', 'you', 'i', 'need', 'want', 'like', 'would', 'could',
            'should', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        # Extract quoted phrases first
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        terms = quoted_phrases.copy()
        
        # Remove quoted phrases from query for further processing
        query_without_quotes = re.sub(r'"[^"]+"', '', query)
        
        # Extract words
        words = re.findall(r'\b\w+\b', query_without_quotes.lower())
        
        # Add meaningful words
        for word in words:
            if (len(word) > 2 and 
                word not in stop_words and 
                not word.isdigit() and
                word not in terms):
                terms.append(word)
        
        # Look for compound terms and technical phrases
        compound_patterns = [
            r'rollout[\s-]*restart',
            r'dr[\s-]*related',
            r'database[\s-]*issue',
            r'deployment[\s-]*issue',
            r'performance[\s-]*issue',
            r'security[\s-]*vulnerability',
            r'production[\s-]*issue',
            r'traffic[\s-]*surge',
            r'system[\s-]*down',
            r'server[\s-]*error'
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                clean_term = re.sub(r'[\s-]+', ' ', match).strip()
                if clean_term not in terms:
                    terms.append(clean_term)
        
        return terms[:10]  # Limit to avoid too many terms
    
    def _build_date_jql(self, query: str) -> Optional[str]:
        """Build date-based JQL from query"""
        today = datetime.now()
        
        date_patterns = [
            (r'today', 0),
            (r'yesterday', -1),
            (r'last week', -7),
            (r'past week', -7),
            (r'last month', -30),
            (r'past month', -30),
            (r'last year', -365),
            (r'past year', -365)
        ]
        
        for pattern, days_offset in date_patterns:
            if pattern in query:
                date = today + timedelta(days=days_offset)
                if 'created' in query or 'reported' in query:
                    return f"created >= '{date.strftime('%Y-%m-%d')}' ORDER BY created DESC"
                elif 'updated' in query or 'modified' in query:
                    return f"updated >= '{date.strftime('%Y-%m-%d')}' ORDER BY updated DESC"
                else:
                    return f"updated >= '{date.strftime('%Y-%m-%d')}' ORDER BY updated DESC"
        
        # Look for specific dates
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
        if date_match:
            date_str = date_match.group(1)
            return f"created >= '{date_str}' ORDER BY created DESC"
        
        return None
    
    def _build_user_jql(self, query: str) -> List[str]:
        """Build user-based JQL queries"""
        jql_queries = []
        
        # Extract user names/emails
        user_patterns = [
            r'assigned to (\w+)',
            r'assignee (\w+)',
            r'reported by (\w+)',
            r'reporter (\w+)',
            r'created by (\w+)',
            r'user (\w+)',
            r'person (\w+)'
        ]
        
        for pattern in user_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if 'assigned' in pattern or 'assignee' in pattern:
                    jql_queries.append(f"assignee = '{match}' ORDER BY updated DESC")
                elif 'reported' in pattern or 'reporter' in pattern or 'created' in pattern:
                    jql_queries.append(f"reporter = '{match}' ORDER BY updated DESC")
                else:
                    jql_queries.append(f"(assignee = '{match}' OR reporter = '{match}') ORDER BY updated DESC")
        
        return jql_queries
    
    def _build_project_jql(self, query: str, projects: List[Dict]) -> List[str]:
        """Build project-specific JQL queries"""
        jql_queries = []
        
        # Look for project keys or names in the query
        for project in projects:
            project_key = project.get('key', '').lower()
            project_name = project.get('name', '').lower()
            
            if (project_key in query or 
                project_name in query or 
                any(word in query for word in project_name.split())):
                jql_queries.append(f"project = '{project.get('key')}' ORDER BY updated DESC")
        
        return jql_queries
    
    def get_issue_details(self, issue_key: str) -> Dict:
        """Get comprehensive issue details including comments and attachments"""
        cache_key = f"issue_details_{issue_key}"
        
        # Get basic issue data
        issue = self.make_request('GET', f'issue/{issue_key}', cache_key=cache_key)
        
        if issue:
            # Get comments
            comments = self.make_request('GET', f'issue/{issue_key}/comment', 
                                       cache_key=f"comments_{issue_key}")
            if comments:
                issue['comments'] = comments.get('comments', [])
            
            # Get worklog
            worklog = self.make_request('GET', f'issue/{issue_key}/worklog', 
                                      cache_key=f"worklog_{issue_key}")
            if worklog:
                issue['worklog'] = worklog.get('worklogs', [])
            
            # Get transitions
            transitions = self.make_request('GET', f'issue/{issue_key}/transitions', 
                                          cache_key=f"transitions_{issue_key}")
            if transitions:
                issue['transitions'] = transitions.get('transitions', [])
        
        return issue

class JiraCache:
    """Enhanced caching system"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}
        self.max_memory_items = 100
    
    def _get_cache_path(self, key: str) -> str:
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', key)[:100]
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def get(self, key: str) -> Optional[Any]:
        # Check memory cache first
        if key in self.memory_cache:
            cache_time, data = self.memory_cache[key]
            if time.time() - cache_time < CACHE_EXPIRY * 3600:
                return data
            else:
                del self.memory_cache[key]
        
        # Check file cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                cache_time = cached_data.get('timestamp', 0)
                if time.time() - cache_time < CACHE_EXPIRY * 3600:
                    # Store in memory cache for faster access
                    self._store_in_memory(key, cached_data.get('data'))
                    return cached_data.get('data')
                else:
                    os.remove(cache_path)
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        
        return None
    
    def set(self, key: str, data: Any):
        timestamp = time.time()
        
        # Store in memory cache
        self._store_in_memory(key, data)
        
        # Store in file cache
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({'timestamp': timestamp, 'data': data}, f)
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def _store_in_memory(self, key: str, data: Any):
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k][0])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = (time.time(), data)
    
    def clear(self):
        self.memory_cache.clear()
        for filename in os.listdir(self.cache_dir):
            try:
                os.remove(os.path.join(self.cache_dir, filename))
            except Exception as e:
                logger.error(f"Error removing cache file: {e}")

class IntelligentGeminiChat:
    """Enhanced Gemini chat with better context understanding"""
    
    def __init__(self, project_id: str, location: str, model_name: str):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        
        self.system_prompt = """You are JiraGPT, an advanced AI assistant specialized in analyzing and explaining Jira project data. You have access to comprehensive information about Jira tickets, projects, users, and workflows.

Your core capabilities include:

1. **Data Analysis & Summarization**: Analyze complex Jira data and provide clear, actionable insights
2. **Natural Language Understanding**: Understand any type of question about Jira, from simple lookups to complex analytical queries
3. **Comprehensive Reporting**: Provide detailed summaries with key metrics, trends, and patterns
4. **Technical Explanation**: Explain technical issues, workflows, and processes in understandable terms
5. **Contextual Awareness**: Remember conversation context and provide relevant follow-up information

When responding:
- Always provide a clear summary at the beginning if dealing with multiple tickets
- Include relevant statistics (counts, percentages, trends)
- Highlight important information (high priority items, overdue tickets, etc.)
- Provide actionable insights and recommendations when appropriate
- Include direct links to Jira tickets using the format: [TICKET-ID](URL)
- Format your responses for easy scanning (use headers, bullet points, tables when helpful)

Response Structure for Complex Queries:
1. **Summary**: Brief overview of findings
2. **Key Details**: Main information requested
3. **Statistics**: Relevant numbers and metrics
4. **Insights**: Analysis and patterns observed
5. **Recommendations**: Suggested actions (if applicable)

Always be comprehensive yet concise, professional yet approachable."""
        
        self.conversation_context = []
    
    def generate_comprehensive_response(self, user_query: str, jira_data: Dict = None) -> str:
        """Generate comprehensive response with full context"""
        try:
            # Build context
            context = f"{self.system_prompt}\n\n"
            
            if jira_data:
                context += "=== JIRA DATA CONTEXT ===\n"
                context += self._format_comprehensive_context(jira_data)
                context += "\n=== END JIRA DATA ===\n\n"
            
            # Add conversation history for context
            if self.conversation_context:
                context += "=== RECENT CONVERSATION ===\n"
                for exchange in self.conversation_context[-3:]:
                    context += f"User: {exchange['user']}\n"
                    context += f"Assistant: {exchange['assistant'][:200]}...\n\n"
                context += "=== END CONVERSATION ===\n\n"
            
            full_prompt = f"{context}Current User Query: {user_query}\n\nProvide a comprehensive response:"
            
            generation_config = GenerationConfig(
                temperature=0.3,
                top_p=0.95,
                max_output_tokens=8192,
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            
            if hasattr(response, 'text'):
                result = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                result = response.candidates[0].text
            else:
                result = "I couldn't generate a response. Please try rephrasing your question."
            
            # Add to conversation context
            self.conversation_context.append({
                'user': user_query,
                'assistant': result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 5 exchanges
            if len(self.conversation_context) > 5:
                self.conversation_context = self.conversation_context[-5:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _format_comprehensive_context(self, jira_data: Dict) -> str:
        """Format Jira data comprehensively for better AI understanding"""
        context = ""
        
        if jira_data.get('issues'):
            issues = jira_data['issues']
            context += f"FOUND {len(issues)} JIRA TICKETS:\n\n"
            
            # Statistics
            stats = self._calculate_statistics(issues)
            context += "STATISTICS:\n"
            for key, value in stats.items():
                context += f"- {key}: {value}\n"
            context += "\n"
            
            # Detailed issue information
            context += "DETAILED TICKET INFORMATION:\n"
            for idx, issue in enumerate(issues, 1):
                context += self._format_single_issue(issue, idx)
                context += "\n" + "="*50 + "\n"
        
        if jira_data.get('projects'):
            context += f"\nAVAILABLE PROJECTS ({len(jira_data['projects'])}):\n"
            for project in jira_data['projects']:
                context += f"- {project.get('key')}: {project.get('name')}\n"
        
        return context
    
    def _calculate_statistics(self, issues: List[Dict]) -> Dict[str, Union[int, str]]:
        """Calculate comprehensive statistics from issues"""
        stats = {}
        
        # Basic counts
        stats["Total Issues"] = len(issues)
        
        # Status distribution
        status_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        type_counts = defaultdict(int)
        assignee_counts = defaultdict(int)
        
        for issue in issues:
            fields = issue.get('fields', {})
            
            # Status
            status = fields.get('status', {}).get('name', 'Unknown')
            status_counts[status] += 1
            
            # Priority
            priority = fields.get('priority', {}).get('name', 'Unknown')
            priority_counts[priority] += 1
            
            # Issue Type
            issue_type = fields.get('issuetype', {}).get('name', 'Unknown')
            type_counts[issue_type] += 1
            
            # Assignee
            assignee = fields.get('assignee', {}).get('displayName', 'Unassigned')
            assignee_counts[assignee] += 1
        
        # Add distributions to stats
        stats["Status Distribution"] = dict(status_counts)
        stats["Priority Distribution"] = dict(priority_counts)
        stats["Type Distribution"] = dict(type_counts)
        stats["Assignee Distribution"] = dict(assignee_counts)
        
        return stats
    
    def _format_single_issue(self, issue: Dict, index: int) -> str:
        """Format a single issue with comprehensive details"""
        fields = issue.get('fields', {})
        key = issue.get('key', 'Unknown')
        
        formatted = f"TICKET #{index}: {key}\n"
        formatted += f"Summary: {fields.get('summary', 'No summary')}\n"
        formatted += f"Status: {fields.get('status', {}).get('name', 'Unknown')}\n"
        formatted += f"Priority: {fields.get('priority', {}).get('name', 'Unknown')}\n"
        formatted += f"Type: {fields.get('issuetype', {}).get('name', 'Unknown')}\n"
        formatted += f"Assignee: {fields.get('assignee', {}).get('displayName', 'Unassigned')}\n"
        formatted += f"Reporter: {fields.get('reporter', {}).get('displayName', 'Unknown')}\n"
        formatted += f"Created: {fields.get('created', 'Unknown')}\n"
        formatted += f"Updated: {fields.get('updated', 'Unknown')}\n"
        
        if fields.get('description'):
            desc = fields['description'][:300]
            formatted += f"Description: {desc}{'...' if len(fields['description']) > 300 else ''}\n"
        
        if fields.get('labels'):
            formatted += f"Labels: {', '.join(fields['labels'])}\n"
        
        if fields.get('components'):
            components = [c.get('name', '') for c in fields['components']]
            formatted += f"Components: {', '.join(components)}\n"
        
        # Comments
        if issue.get('comments'):
            comments = issue['comments']
            formatted += f"Comments: {len(comments)} total\n"
            if comments:
                latest_comment = comments[-1]
                comment_body = latest_comment.get('body', '')[:200]
                formatted += f"Latest Comment: {comment_body}{'...' if len(latest_comment.get('body', '')) > 200 else ''}\n"
        
        formatted += f"Jira Link: {JIRA_URL}browse/{key}\n"
        
        return formatted

class SmartJiraChatbot:
    """Main intelligent chatbot class"""
    
    def __init__(self):
        self.jira_client = EnhancedJiraClient(JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN)
        self.gemini_chat = IntelligentGeminiChat(PROJECT_ID, REGION, MODEL_NAME)
        logger.info("SmartJiraChatbot initialized successfully")
    
    def process_any_query(self, query: str) -> str:
        """Process any type of query intelligently"""
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        try:
            # Handle special commands
            if query.lower().strip() in ["clear cache", "refresh cache", "reset cache"]:
                self.jira_client.cache.clear()
                return "✅ Cache cleared successfully! Fresh data will be fetched for your next queries."
            
            # Handle greeting/farewell messages
            greeting_responses = self._handle_simple_interactions(query)
            if greeting_responses:
                return greeting_responses
            
            # Check if specific issue keys are mentioned
            issue_keys = re.findall(r'[A-Z]+-\d+', query.upper())
            
            jira_data = {}
            
            if issue_keys:
                # Get specific issues mentioned
                logger.info(f"Found specific issue keys: {issue_keys}")
                issues_data = []
                for issue_key in issue_keys:
                    issue_details = self.jira_client.get_issue_details(issue_key)
                    if issue_details:
                        issues_data.append(issue_details)
                
                if issues_data:
                    jira_data['issues'] = issues_data
                    jira_data['query_type'] = 'specific_issues'
            else:
                # Perform intelligent search
                logger.info("Performing intelligent search")
                search_results = self.jira_client.intelligent_search(query, max_results=50)
                
                if search_results:
                    jira_data['issues'] = search_results
                    jira_data['query_type'] = 'search_results'
                    
                    # Get additional details for top results if needed
                    if any(word in query.lower() for word in ['detail', 'comment', 'attachment', 'history']):
                        detailed_issues = []
                        for issue in search_results[:10]:  # Get details for top 10
                            detailed_issue = self.jira_client.get_issue_details(issue.get('key'))
                            if detailed_issue:
                                detailed_issues.append(detailed_issue)
                        if detailed_issues:
                            jira_data['issues'] = detailed_issues
                
                # Add project information if query seems project-related
                if any(word in query.lower() for word in ['project', 'projects']):
                    jira_data['projects'] = self.jira_client.get_all_projects()
            
            # If no data found, still try to provide a helpful response
            if not jira_data.get('issues') and not jira_data.get('projects'):
                logger.info("No specific data found, generating general response")
                jira_data['no_results'] = True
                jira_data['original_query'] = query
                
                # Try a broader search
                broad_search = self.jira_client.intelligent_search("", max_results=20)
                if broad_search:
                    jira_data['recent_issues'] = broad_search[:5]
            
            # Generate comprehensive response
            response = self.gemini_chat.generate_comprehensive_response(query, jira_data)
            
            # Log processing time
            elapsed_time = time.time() - start_time
            logger.info(f"Query processed in {elapsed_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"❌ I encountered an error while processing your question. Please try again or rephrase your question.\n\nTechnical details: {str(e)}"
    
    def _handle_simple_interactions(self, query: str) -> Optional[str]:
        """Handle simple greetings and common interactions"""
        query_lower = query.lower().strip()
        
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        farewells = ['bye', 'goodbye', 'see you', 'thanks', 'thank you']
        help_requests = ['help', 'what can you do', 'how do you work', 'capabilities']
        
        if any(greeting in query_lower for greeting in greetings):
            return """👋 Hello! I'm your intelligent Jira assistant. I can help you with:

🎯 **Any question about your Jira tickets** - just ask naturally!
📊 **Examples of what you can ask:**
   • "Show me tickets about rollout restart"
   • "High priority bugs from last week"
   • "All tickets assigned to John"
   • "DR related issues this month"
   • "Tell me about RDRF-2606"
   • "Traffic surge problems"
   • "Open tasks in my project"

Just ask me anything about your Jira data - I'll understand and help! 🚀"""

        if any(farewell in query_lower for farewell in farewells):
            return "👋 Goodbye! Feel free to ask me anything about your Jira tickets anytime. Have a great day!"
        
        if any(help_req in query_lower for help_req in help_requests):
            return """🤖 **I'm your intelligent Jira assistant!** Here's what I can do:

🔍 **Smart Search**: Ask me anything about tickets using natural language
📊 **Data Analysis**: Get summaries, statistics, and insights from your Jira data
🎯 **Flexible Queries**: I understand context and can handle any type of question

💡 **Try asking me:**
   • "Show me all rollout restart issues"
   • "High priority tickets assigned to me"
   • "Bugs reported this week"
   • "Tell me about project RDRF"
   • "DR related problems"
   • "Traffic surge alerts"
   • "Tickets updated today"

Just type your question naturally - I'll figure out what you need! 🚀"""
        
        return None
    
    def get_usage_stats(self) -> str:
        """Get chatbot usage statistics"""
        try:
            cache_files = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')])
            return f"📊 **Chatbot Stats:**\n- Cached items: {cache_files}\n- Session queries: {len(self.gemini_chat.conversation_context)}"
        except:
            return "📊 Stats unavailable"
    
    def run_interactive_chat(self):
        """Run the chatbot in interactive mode"""
        print("\n" + "="*60)
        print("🤖 SMART JIRA CHATBOT - ENHANCED VERSION")
        print("="*60)
        print("💡 Ask me ANYTHING about your Jira tickets!")
        print("📝 Examples:")
        print("   • 'Show me rollout restart tickets'")
        print("   • 'High priority bugs from last week'")
        print("   • 'Tell me about RDRF-2606'")
        print("   • 'DR related issues this month'")
        print("\n💬 Type 'exit' to quit, 'clear cache' to refresh data")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🗣️  You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\n🤖 Chatbot: 👋 Goodbye! Thanks for using Smart Jira Chatbot!")
                    break
                
                if user_input.lower() == "stats":
                    print(f"\n🤖 Chatbot: {self.get_usage_stats()}")
                    continue
                
                print("\n🤖 Chatbot: ", end="")
                response = self.process_any_query(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 Chatbot: 👋 Goodbye! Thanks for using Smart Jira Chatbot!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

def main():
    """Main function to run the chatbot"""
    try:
        # Test connection first
        print("🔄 Initializing Smart Jira Chatbot...")
        chatbot = SmartJiraChatbot()
        print("✅ Successfully connected to Jira!")
        
        # Run interactive chat
        chatbot.run_interactive_chat()
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("Please check your Jira credentials and network connection.")

if __name__ == "__main__":
    main()
