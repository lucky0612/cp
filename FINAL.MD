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

class SmartJiraClient:
    """Ultra-smart Jira client that understands context"""
    
    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.cache = JiraCache()
        self._metadata = None
        
    def safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        """Safely get value from object, handling None cases"""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default
    
    def make_request(self, method: str, endpoint: str, params: Dict = None, 
                     data: Dict = None, cache_key: str = None, force_refresh: bool = False) -> Optional[Dict]:
        """Make HTTP request with enhanced error handling"""
        if method.lower() == 'get' and cache_key and not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached data for: {cache_key}")
                return cached_data
        
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
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if method.lower() == 'get' and cache_key:
                    self.cache.set(cache_key, result)
                return result
            else:
                logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error for {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def get_jira_metadata(self) -> Dict:
        """Get comprehensive Jira metadata for query building"""
        if self._metadata is not None:
            return self._metadata
            
        metadata = {}
        
        try:
            # Get projects
            projects = self.make_request('GET', 'project', cache_key="all_projects") or []
            metadata['projects'] = [
                {
                    'key': self.safe_get(p, 'key', ''),
                    'name': self.safe_get(p, 'name', ''),
                    'id': self.safe_get(p, 'id', '')
                }
                for p in projects if p is not None
            ]
            
            # Get issue types
            issue_types = self.make_request('GET', 'issuetype', cache_key="issue_types") or []
            metadata['issue_types'] = [
                {
                    'name': self.safe_get(it, 'name', ''),
                    'id': self.safe_get(it, 'id', '')
                }
                for it in issue_types if it is not None
            ]
            
            # Get priorities
            priorities = self.make_request('GET', 'priority', cache_key="priorities") or []
            metadata['priorities'] = [
                {
                    'name': self.safe_get(p, 'name', ''),
                    'id': self.safe_get(p, 'id', '')
                }
                for p in priorities if p is not None
            ]
            
            # Get statuses
            statuses = self.make_request('GET', 'status', cache_key="statuses") or []
            metadata['statuses'] = [
                {
                    'name': self.safe_get(s, 'name', ''),
                    'id': self.safe_get(s, 'id', '')
                }
                for s in statuses if s is not None
            ]
            
            # Get users (sample)
            users = self.make_request('GET', 'users/search', params={'maxResults': 50}, cache_key="sample_users") or []
            metadata['users'] = [
                {
                    'displayName': self.safe_get(u, 'displayName', ''),
                    'emailAddress': self.safe_get(u, 'emailAddress', ''),
                    'accountId': self.safe_get(u, 'accountId', '')
                }
                for u in users if u is not None
            ]
            
            self._metadata = metadata
            logger.info(f"Loaded metadata: {len(metadata.get('projects', []))} projects, {len(metadata.get('issue_types', []))} issue types")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self._metadata = {
                'projects': [],
                'issue_types': [],
                'priorities': [],
                'statuses': [],
                'users': []
            }
        
        return self._metadata
    
    def execute_jql_query(self, jql: str, max_results: int = 100) -> List[Dict]:
        """Execute JQL query and return results safely"""
        if not jql or jql.strip() == "":
            logger.warning("Empty JQL query provided")
            return []
            
        try:
            logger.info(f"Executing JQL: {jql}")
            params = {
                "jql": jql,
                "maxResults": max_results,
                "fields": [
                    "summary", "description", "status", "assignee", "reporter", 
                    "created", "updated", "priority", "issuetype", "labels", 
                    "comment", "components", "fixVersions", "versions", 
                    "resolution", "resolutiondate", "project", "key"
                ]
            }
            
            cache_key = f"jql_{hash(jql)}_{max_results}"
            result = self.make_request('GET', 'search', params=params, cache_key=cache_key)
            
            if result is not None and 'issues' in result:
                issues = result['issues']
                logger.info(f"JQL returned {len(issues)} issues")
                return [issue for issue in issues if issue is not None]
            else:
                logger.warning("JQL query returned no results or failed")
                return []
                
        except Exception as e:
            logger.error(f"Error executing JQL query: {e}")
            return []
    
    def get_issue_details(self, issue_key: str) -> Optional[Dict]:
        """Get detailed information about a specific issue"""
        if not issue_key:
            return None
            
        try:
            cache_key = f"issue_{issue_key}"
            issue = self.make_request('GET', f'issue/{issue_key}', cache_key=cache_key)
            
            if issue is not None:
                # Get comments
                comments = self.make_request('GET', f'issue/{issue_key}/comment', 
                                           cache_key=f"comments_{issue_key}")
                if comments is not None and 'comments' in comments:
                    issue['all_comments'] = comments['comments']
                
                return issue
            return None
            
        except Exception as e:
            logger.error(f"Error getting issue details for {issue_key}: {e}")
            return None

class JiraCache:
    """Simple and reliable caching system"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}
    
    def _get_cache_path(self, key: str) -> str:
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', str(key))[:100]
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def get(self, key: str) -> Optional[Any]:
        try:
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
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                cache_time = cached_data.get('timestamp', 0)
                if time.time() - cache_time < CACHE_EXPIRY * 3600:
                    # Store in memory for faster access
                    self.memory_cache[key] = (time.time(), cached_data.get('data'))
                    return cached_data.get('data')
                else:
                    os.remove(cache_path)
        except Exception as e:
            logger.error(f"Cache read error: {e}")
        
        return None
    
    def set(self, key: str, data: Any):
        try:
            # Store in memory
            self.memory_cache[key] = (time.time(), data)
            
            # Store in file
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f)
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def clear(self):
        self.memory_cache.clear()
        try:
            for filename in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, filename))
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

class AIQueryAnalyzer:
    """AI-powered query analyzer that understands intent and builds perfect JQL"""
    
    def __init__(self, project_id: str, location: str, model_name: str):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
    
    def analyze_and_build_jql(self, user_query: str, jira_metadata: Dict) -> List[str]:
        """Analyze user query and build optimal JQL queries"""
        
        analysis_prompt = f"""You are a Jira JQL expert. Analyze the user's natural language query and create the perfect JQL queries to get exactly what they want.

USER QUERY: "{user_query}"

AVAILABLE JIRA METADATA:
Projects: {[p['key'] + ':' + p['name'] for p in jira_metadata.get('projects', [])]}
Issue Types: {[it['name'] for it in jira_metadata.get('issue_types', [])]}
Priorities: {[p['name'] for p in jira_metadata.get('priorities', [])]}
Statuses: {[s['name'] for s in jira_metadata.get('statuses', [])]}

INSTRUCTIONS:
1. Understand what the user wants (tickets about what topic, with what criteria)
2. Create 2-3 different JQL queries that capture different aspects of their request
3. Use text search (summary ~ "term" OR description ~ "term" OR comment ~ "term") for content matching
4. Include relevant filters for status, priority, type, dates, etc.
5. Order results by relevance (usually "ORDER BY updated DESC")

EXAMPLES:
- "rollout restart issues" → summary ~ "rollout restart" OR description ~ "rollout restart" OR comment ~ "rollout restart"
- "high priority bugs" → priority = High AND issuetype = Bug
- "tickets assigned to john" → assignee = john
- "DR related problems" → summary ~ "DR" OR description ~ "DR" OR labels = DR

OUTPUT FORMAT:
Return ONLY the JQL queries, one per line, no explanations:
query1
query2
query3

JQL QUERIES:"""

        try:
            response = self.model.generate_content(
                analysis_prompt,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )
            
            if hasattr(response, 'text'):
                jql_text = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                jql_text = response.candidates[0].text.strip()
            else:
                return self._fallback_jql(user_query)
            
            # Parse JQL queries from response
            jql_queries = []
            for line in jql_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and 'jql' not in line.lower():
                    # Clean up the query
                    line = re.sub(r'^[-*•]\s*', '', line)  # Remove bullet points
                    line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbers
                    if len(line) > 10:  # Reasonable minimum length
                        jql_queries.append(line)
            
            if not jql_queries:
                return self._fallback_jql(user_query)
            
            logger.info(f"AI generated {len(jql_queries)} JQL queries")
            return jql_queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            logger.error(f"Error in AI query analysis: {e}")
            return self._fallback_jql(user_query)
    
    def _fallback_jql(self, user_query: str) -> List[str]:
        """Fallback JQL generation if AI fails"""
        logger.info("Using fallback JQL generation")
        
        # Extract meaningful terms
        terms = []
        words = re.findall(r'\b\w+\b', user_query.lower())
        stop_words = {'show', 'me', 'all', 'the', 'get', 'find', 'list', 'about', 'with', 'for', 'in', 'on', 'at'}
        
        for word in words:
            if len(word) > 2 and word not in stop_words:
                terms.append(word)
        
        jql_queries = []
        
        if terms:
            # Text search query
            text_conditions = []
            for term in terms[:3]:  # Limit to 3 terms
                text_conditions.append(f'summary ~ "{term}" OR description ~ "{term}"')
            
            if text_conditions:
                jql_queries.append(f"({' OR '.join(text_conditions)}) ORDER BY updated DESC")
        
        # Add a recent items query as fallback
        jql_queries.append("updated >= -7d ORDER BY updated DESC")
        
        return jql_queries

class SmartResponseGenerator:
    """AI-powered response generator with comprehensive analysis"""
    
    def __init__(self, project_id: str, location: str, model_name: str):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
    
    def generate_response(self, user_query: str, jira_data: Dict) -> str:
        """Generate comprehensive response based on Jira data"""
        
        system_prompt = """You are JiraGPT, an expert Jira analyst. Provide comprehensive, well-structured responses about Jira tickets and data.

RESPONSE STRUCTURE:
1. **SUMMARY** - Brief overview of what was found
2. **KEY FINDINGS** - Main insights and important information  
3. **DETAILED BREAKDOWN** - Specific ticket information
4. **STATISTICS** - Numbers, counts, distributions
5. **RECOMMENDATIONS** - Actionable next steps (if applicable)

FORMATTING GUIDELINES:
- Use emojis and clear headers for better readability
- Include ticket links: [TICKET-KEY](https://cmegroup.atlassian.net/browse/TICKET-KEY)
- Highlight important information in **bold**
- Use bullet points for lists
- Provide specific statistics and counts
- Be comprehensive but concise"""

        # Format the Jira data
        context = self._format_jira_context(jira_data)
        
        full_prompt = f"""{system_prompt}

USER QUERY: "{user_query}"

JIRA DATA:
{context}

Provide a comprehensive response addressing the user's query:"""

        try:
            response = self.model.generate_content(
                full_prompt,
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
                return "I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error generating response: {str(e)}"
    
    def _format_jira_context(self, jira_data: Dict) -> str:
        """Format Jira data for AI context"""
        if not jira_data:
            return "No Jira data available."
        
        context = ""
        
        issues = jira_data.get('issues', [])
        if issues:
            context += f"FOUND {len(issues)} TICKETS:\n\n"
            
            # Add statistics
            stats = self._calculate_stats(issues)
            context += "STATISTICS:\n"
            for key, value in stats.items():
                context += f"- {key}: {value}\n"
            context += "\n"
            
            # Add ticket details
            context += "TICKET DETAILS:\n"
            for i, issue in enumerate(issues[:20], 1):  # Limit to 20 for context
                context += self._format_issue(issue, i)
                context += "\n"
        else:
            context += "No tickets found matching the criteria.\n"
        
        return context
    
    def _calculate_stats(self, issues: List[Dict]) -> Dict:
        """Calculate statistics from issues"""
        stats = {"Total Tickets": len(issues)}
        
        # Status distribution
        status_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for issue in issues:
            if issue is None:
                continue
                
            fields = self.safe_get(issue, 'fields', {})
            
            status = self.safe_get(self.safe_get(fields, 'status', {}), 'name', 'Unknown')
            status_counts[status] += 1
            
            priority = self.safe_get(self.safe_get(fields, 'priority', {}), 'name', 'Unknown')
            priority_counts[priority] += 1
            
            issue_type = self.safe_get(self.safe_get(fields, 'issuetype', {}), 'name', 'Unknown')
            type_counts[issue_type] += 1
        
        stats["By Status"] = dict(status_counts)
        stats["By Priority"] = dict(priority_counts)
        stats["By Type"] = dict(type_counts)
        
        return stats
    
    def _format_issue(self, issue: Dict, index: int) -> str:
        """Format a single issue for context"""
        if issue is None:
            return f"{index}. Invalid issue data\n"
        
        fields = self.safe_get(issue, 'fields', {})
        key = self.safe_get(issue, 'key', 'Unknown')
        
        formatted = f"{index}. **{key}**\n"
        formatted += f"   Summary: {self.safe_get(fields, 'summary', 'No summary')}\n"
        formatted += f"   Status: {self.safe_get(self.safe_get(fields, 'status', {}), 'name', 'Unknown')}\n"
        formatted += f"   Priority: {self.safe_get(self.safe_get(fields, 'priority', {}), 'name', 'Unknown')}\n"
        formatted += f"   Assignee: {self.safe_get(self.safe_get(fields, 'assignee', {}), 'displayName', 'Unassigned')}\n"
        formatted += f"   Updated: {self.safe_get(fields, 'updated', 'Unknown')}\n"
        
        return formatted
    
    def safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        """Safely get value from object"""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

class UltraSmartJiraChatbot:
    """The ultimate Jira chatbot that understands everything"""
    
    def __init__(self):
        self.jira_client = SmartJiraClient(JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN)
        self.query_analyzer = AIQueryAnalyzer(PROJECT_ID, REGION, MODEL_NAME)
        self.response_generator = SmartResponseGenerator(PROJECT_ID, REGION, MODEL_NAME)
        logger.info("UltraSmartJiraChatbot initialized successfully")
    
    def process_query(self, user_query: str) -> str:
        """Process any query intelligently"""
        start_time = time.time()
        logger.info(f"Processing query: {user_query}")
        
        try:
            # Handle special commands
            if user_query.lower().strip() in ["clear cache", "refresh cache", "reset cache"]:
                self.jira_client.cache.clear()
                return "✅ Cache cleared! Fresh data will be fetched for your next queries."
            
            # Handle simple interactions
            simple_response = self._handle_simple_interactions(user_query)
            if simple_response:
                return simple_response
            
            # Check for specific issue keys
            issue_keys = re.findall(r'[A-Z]+-\d+', user_query.upper())
            
            jira_data = {'issues': []}
            
            if issue_keys:
                # Get specific issues
                logger.info(f"Getting specific issues: {issue_keys}")
                for issue_key in issue_keys:
                    issue = self.jira_client.get_issue_details(issue_key)
                    if issue:
                        jira_data['issues'].append(issue)
            else:
                # Use AI to analyze query and build JQL
                logger.info("Using AI to analyze query and build JQL")
                metadata = self.jira_client.get_jira_metadata()
                jql_queries = self.query_analyzer.analyze_and_build_jql(user_query, metadata)
                
                all_issues = []
                for jql in jql_queries:
                    issues = self.jira_client.execute_jql_query(jql, max_results=50)
                    all_issues.extend(issues)
                
                # Remove duplicates
                seen_keys = set()
                unique_issues = []
                for issue in all_issues:
                    if issue is None:
                        continue
                    key = self.jira_client.safe_get(issue, 'key')
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        unique_issues.append(issue)
                
                jira_data['issues'] = unique_issues
            
            # Generate response
            response = self.response_generator.generate_response(user_query, jira_data)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Query processed in {elapsed_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}\n\nPlease try rephrasing your question or check the logs for more details."
    
    def _handle_simple_interactions(self, query: str) -> Optional[str]:
        """Handle greetings and simple interactions"""
        query_lower = query.lower().strip()
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return """👋 **Hello! I'm your Ultra-Smart Jira Assistant!**

🚀 **I can understand ANY question about your Jira tickets:**
• "Show me rollout restart issues"
• "High priority bugs from last week" 
• "DR related tickets assigned to John"
• "Traffic surge problems this month"
• "Tell me about RDRF-2606"

Just ask me naturally - I'll understand and find exactly what you need! 💡"""

        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            return "👋 Goodbye! Come back anytime you need help with your Jira tickets!"
        
        if any(word in query_lower for word in ['help', 'what can you do']):
            return """🤖 **I'm your intelligent Jira assistant!**

✨ **What makes me special:**
• **Natural Language Understanding**: Ask questions naturally, no rigid syntax needed
• **AI-Powered Search**: I analyze your question and build perfect queries automatically  
• **Comprehensive Responses**: Get summaries, statistics, and insights
• **Flexible Queries**: Handle any type of Jira question you can think of

💡 **Just ask me anything about your Jira tickets!**"""
        
        return None
    
    def run_chat(self):
        """Run the interactive chat"""
        print("\n" + "="*70)
        print("🚀 ULTRA-SMART JIRA CHATBOT")
        print("="*70)
        print("💡 Ask me ANYTHING about your Jira tickets - I understand natural language!")
        print("\n📝 Examples:")
        print("   • 'Show me all rollout restart issues'")
        print("   • 'High priority DR tickets from last month'") 
        print("   • 'Bugs assigned to John about database'")
        print("   • 'Traffic surge alerts this week'")
        print("\n💬 Type 'exit' to quit, 'clear cache' to refresh")
        print("="*70)
        
        while True:
            try:
                user_input = input("\n🗣️  Ask me: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\n🤖 Goodbye! Thanks for using Ultra-Smart Jira Chatbot! 👋")
                    break
                
                print(f"\n🤖 **JiraGPT**: ", end="")
                response = self.process_query(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 Goodbye! Thanks for using Ultra-Smart Jira Chatbot! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main function"""
    try:
        print("🔄 Initializing Ultra-Smart Jira Chatbot...")
        chatbot = UltraSmartJiraChatbot()
        print("✅ Ready to help with your Jira queries!")
        chatbot.run_chat()
    except Exception as e:
        print(f"❌ Failed to start: {e}")

if __name__ == "__main__":
    main()
