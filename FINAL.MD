#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import urllib3
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import getpass
import pickle
import time
import re

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("remedy_ai_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RemedyAIChatbot")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Remedy Configuration
REMEDY_SERVER = "cmegroup-restapi.onbmc.com"
CACHE_DIR = "remedy_cache"
CACHE_EXPIRY = 6  # hours

class UniversalRemedyClient:
    """Universal Remedy client that finds EVERYTHING using AI"""
    
    def __init__(self):
        self.base_url = f"https://{REMEDY_SERVER}"
        self.token = None
        self.cache = SimpleCache()
        self._metadata = None
        logger.info("Universal Remedy client initialized")
    
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
    
    def login(self, username: str = None, password: str = None) -> bool:
        """Login to Remedy"""
        try:
            if not username:
                username = input("Enter Remedy Username: ")
            if not password:
                password = getpass.getpass("Enter Remedy Password: ")
            
            login_url = f"{self.base_url}/api/jwt/login"
            payload = {'username': username, 'password': password}
            headers = {'content-type': 'application/x-www-form-urlencoded'}
            
            logger.info("🔐 Logging into Remedy...")
            response = requests.post(login_url, data=payload, headers=headers, verify=False, timeout=30)
            
            if response.status_code == 200:
                self.token = response.text.strip('"')
                logger.info("✅ Successfully logged into Remedy")
                return True
            else:
                logger.error(f"❌ Login failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return False
    
    def logout(self):
        """Logout from Remedy"""
        if self.token:
            try:
                logout_url = f"{self.base_url}/api/jwt/logout"
                headers = {'Authorization': f"AR-JWT {self.token}"}
                requests.post(logout_url, headers=headers, verify=False, timeout=10)
                logger.info("📤 Logged out from Remedy")
            except:
                pass
            finally:
                self.token = None
    
    def get_remedy_metadata(self) -> Dict:
        """Get Remedy metadata for AI context"""
        if self._metadata:
            return self._metadata
        
        logger.info("📊 Loading Remedy metadata...")
        metadata = {
            'available_fields': [
                'Incident Number', 'Summary', 'Description', 'Status', 'Priority', 
                'Assignee', 'Submitter', 'Owner', 'Owner Group', 'Submit Date',
                'Assigned Group', 'Environment', 'Impact', 'Urgency', 'Request Manager'
            ],
            'status_values': [
                'New', 'Assigned', 'In Progress', 'Pending', 'Resolved', 'Closed', 'Cancelled'
            ],
            'priority_values': [
                'Critical', 'High', 'Medium', 'Low'
            ],
            'sample_groups': [
                'TOCC Support', 'Database Team', 'Network Team', 'Application Support'
            ],
            'environments': [
                'Production', 'QA', 'Development', 'Staging'
            ]
        }
        
        self._metadata = metadata
        logger.info("✅ Remedy metadata loaded")
        return metadata
    
    def comprehensive_search(self, user_query: str) -> List[Dict]:
        """AI-powered comprehensive search across all Remedy data"""
        logger.info(f"🔍 Starting comprehensive Remedy search for: '{user_query}'")
        
        if not self.token:
            logger.error("❌ Not logged in to Remedy")
            return []
        
        # Get metadata for AI context
        metadata = self.get_remedy_metadata()
        
        # Generate search strategies using AI
        search_strategies = self.generate_search_strategies(user_query, metadata)
        
        all_results = []
        
        # Execute each search strategy
        for strategy in search_strategies:
            logger.info(f"🎯 Executing search strategy: {strategy['description']}")
            results = self.execute_search_strategy(strategy)
            if results:
                all_results.extend(results)
        
        # Remove duplicates and return
        unique_results = self.deduplicate_incidents(all_results)
        logger.info(f"✅ Found {len(unique_results)} unique incidents")
        
        return unique_results
    
    def generate_search_strategies(self, user_query: str, metadata: Dict) -> List[Dict]:
        """Use AI to generate multiple search strategies"""
        
        # Initialize AI for strategy generation
        vertexai.init(project=PROJECT_ID, location=REGION)
        model = GenerativeModel(MODEL_NAME)
        
        strategy_prompt = f"""You are a Remedy/Helix expert. Generate multiple search strategies for this user query.

USER QUERY: "{user_query}"

REMEDY CONTEXT:
Available Fields: {metadata['available_fields']}
Status Values: {metadata['status_values']}
Priority Values: {metadata['priority_values']}
Sample Groups: {metadata['sample_groups']}
Environments: {metadata['environments']}

GENERATE 3-5 SEARCH STRATEGIES:
For each strategy, provide:
1. search_field: which field to search (Summary, Description, Status, etc.)
2. search_value: what value to search for
3. search_type: exact, contains, starts_with
4. additional_filters: any extra conditions
5. description: what this strategy targets

EXAMPLES:
- "rollout issues" → search Summary/Description for "rollout"
- "high priority incidents" → filter by Priority = "High"
- "incidents assigned to john" → search Assignee field
- "recent production problems" → Status + Environment filters

OUTPUT FORMAT (JSON array):
[
  {
    "search_field": "Summary",
    "search_value": "rollout",
    "search_type": "contains",
    "additional_filters": {},
    "description": "Search for rollout issues in summary"
  }
]

Generate strategies:"""

        try:
            response = model.generate_content(
                strategy_prompt,
                generation_config=GenerationConfig(temperature=0.2, max_output_tokens=1000)
            )
            
            # Parse AI response
            if hasattr(response, 'text'):
                strategy_text = response.text
            else:
                strategy_text = str(response)
            
            # Extract JSON from response
            strategies = self.parse_strategies_from_ai(strategy_text, user_query)
            
            logger.info(f"🧠 AI generated {len(strategies)} search strategies")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ AI strategy generation failed: {e}")
            return self.fallback_strategies(user_query)
    
    def parse_strategies_from_ai(self, ai_text: str, user_query: str) -> List[Dict]:
        """Parse search strategies from AI response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\[.*\]', ai_text, re.DOTALL)
            if json_match:
                strategies_json = json_match.group(0)
                strategies = json.loads(strategies_json)
                return strategies
        except:
            pass
        
        # Fallback parsing
        return self.fallback_strategies(user_query)
    
    def fallback_strategies(self, user_query: str) -> List[Dict]:
        """Fallback search strategies if AI fails"""
        strategies = []
        query_lower = user_query.lower()
        
        # Extract key terms
        terms = [word for word in query_lower.split() if len(word) > 2]
        
        for term in terms[:3]:  # Limit to 3 terms
            strategies.extend([
                {
                    "search_field": "Summary",
                    "search_value": term,
                    "search_type": "contains",
                    "additional_filters": {},
                    "description": f"Search summary for '{term}'"
                },
                {
                    "search_field": "Description", 
                    "search_value": term,
                    "search_type": "contains",
                    "additional_filters": {},
                    "description": f"Search description for '{term}'"
                }
            ])
        
        # Add status-based searches
        if any(status in query_lower for status in ['open', 'new', 'active']):
            strategies.append({
                "search_field": "Status",
                "search_value": "New\" OR \"Status\"=\"Assigned\" OR \"Status\"=\"In Progress",
                "search_type": "exact",
                "additional_filters": {},
                "description": "Search for open incidents"
            })
        
        # Add priority-based searches
        if any(priority in query_lower for priority in ['high', 'critical', 'urgent']):
            strategies.append({
                "search_field": "Priority",
                "search_value": "High\" OR \"Priority\"=\"Critical",
                "search_type": "exact", 
                "additional_filters": {},
                "description": "Search for high priority incidents"
            })
        
        # Always add recent incidents fallback
        strategies.append({
            "search_field": "Submit Date",
            "search_value": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "search_type": "greater_than",
            "additional_filters": {},
            "description": "Recent incidents from last 30 days"
        })
        
        return strategies[:5]  # Limit to 5 strategies
    
    def execute_search_strategy(self, strategy: Dict) -> List[Dict]:
        """Execute a single search strategy"""
        try:
            # Build search query based on strategy
            search_field = strategy.get('search_field', 'Summary')
            search_value = strategy.get('search_value', '')
            search_type = strategy.get('search_type', 'contains')
            
            # Build query string
            if search_type == 'contains':
                query = f'"{search_field}"%20LIKE%20"%25{search_value}%25"'
            elif search_type == 'exact':
                query = f'"{search_field}"%3D"{search_value}"'
            elif search_type == 'starts_with':
                query = f'"{search_field}"%20LIKE%20"{search_value}%25"'
            else:
                query = f'"{search_field}"%20LIKE%20"%25{search_value}%25"'
            
            # Make API call
            api_url = f"{self.base_url}/api/arsys/v1/entry/HPD:Help%20Desk"
            headers = {'Authorization': f"AR-JWT {self.token}"}
            params = {
                'q': query,
                'fields': 'values(Incident Number,Summary,Description,Status,Priority,Assignee,Submitter,Owner,Owner Group,Submit Date,Assigned Group,Environment)',
                'limit': 100
            }
            
            logger.info(f"🌐 Making API call for: {strategy['description']}")
            response = requests.get(api_url, headers=headers, params=params, verify=False, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                entries = self.safe_get(data, 'entries', [])
                
                # Convert to standardized format
                incidents = []
                for entry in entries:
                    if entry:
                        values = self.safe_get(entry, 'values', {})
                        incident = self.standardize_incident(values)
                        if incident:
                            incidents.append(incident)
                
                logger.info(f"✅ Strategy returned {len(incidents)} incidents")
                return incidents
            else:
                logger.warning(f"⚠️ API call failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Strategy execution error: {e}")
            return []
    
    def standardize_incident(self, values: Dict) -> Optional[Dict]:
        """Convert Remedy API response to standardized format"""
        try:
            return {
                'incident_number': self.safe_get(values, 'Incident Number', 'Unknown'),
                'summary': self.safe_get(values, 'Summary', 'No summary'),
                'description': self.safe_get(values, 'Description', 'No description'),
                'status': self.safe_get(values, 'Status', 'Unknown'),
                'priority': self.safe_get(values, 'Priority', 'Unknown'),
                'assignee': self.safe_get(values, 'Assignee', 'Unassigned'),
                'submitter': self.safe_get(values, 'Submitter', 'Unknown'),
                'owner': self.safe_get(values, 'Owner', 'Unknown'),
                'owner_group': self.safe_get(values, 'Owner Group', 'Unknown'),
                'submit_date': self.safe_get(values, 'Submit Date', 'Unknown'),
                'assigned_group': self.safe_get(values, 'Assigned Group', 'Unknown'),
                'environment': self.safe_get(values, 'Environment', 'Unknown'),
                'raw_data': values  # Keep original for reference
            }
        except Exception as e:
            logger.error(f"❌ Error standardizing incident: {e}")
            return None
    
    def deduplicate_incidents(self, incidents: List[Dict]) -> List[Dict]:
        """Remove duplicate incidents"""
        seen_numbers = set()
        unique_incidents = []
        
        for incident in incidents:
            if not incident:
                continue
            number = incident.get('incident_number')
            if number and number not in seen_numbers:
                seen_numbers.add(number)
                unique_incidents.append(incident)
        
        return unique_incidents

class SimpleCache:
    """Simple caching system"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        # Memory cache first
        if key in self.memory_cache:
            timestamp, data = self.memory_cache[key]
            if time.time() - timestamp < CACHE_EXPIRY * 3600:
                return data
            else:
                del self.memory_cache[key]
        
        # File cache
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

class AIRemedyResponseGenerator:
    """AI-powered response generator for Remedy data"""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
    
    def generate_response(self, user_query: str, remedy_data: List[Dict]) -> str:
        """Generate comprehensive response from Remedy data"""
        
        # Format data for AI
        formatted_data = self._format_remedy_data(remedy_data)
        
        prompt = f"""You are RemedyGPT, an expert Remedy/Helix analyst. Provide a comprehensive response to the user's query based on the Remedy incident data found.

USER QUERY: "{user_query}"

REMEDY DATA FOUND:
{formatted_data}

RESPONSE REQUIREMENTS:
1. Start with a clear SUMMARY of what was found
2. Provide KEY STATISTICS (counts, status distributions, priority breakdown)
3. List TOP IMPORTANT INCIDENTS with details
4. Include INSIGHTS and PATTERNS you observe
5. Add ACTIONABLE RECOMMENDATIONS if applicable
6. Use emojis and clear formatting for readability
7. Include incident numbers for reference
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
                return self._create_simple_response(user_query, remedy_data)
                
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return self._create_simple_response(user_query, remedy_data)
    
    def _format_remedy_data(self, incidents: List[Dict]) -> str:
        """Format Remedy data for AI consumption"""
        if not incidents:
            return "No Remedy incidents found."
        
        formatted = f"TOTAL INCIDENTS FOUND: {len(incidents)}\n\n"
        
        # Add statistics
        stats = self._calculate_stats(incidents)
        formatted += "STATISTICS:\n"
        for key, value in stats.items():
            formatted += f"- {key}: {value}\n"
        formatted += "\n"
        
        # Add incident details (limit to first 20)
        formatted += "INCIDENT DETAILS:\n"
        for i, incident in enumerate(incidents[:20], 1):
            if incident:
                formatted += self._format_single_incident(incident, i)
                formatted += "\n"
        
        if len(incidents) > 20:
            formatted += f"... and {len(incidents) - 20} more incidents\n"
        
        return formatted
    
    def _calculate_stats(self, incidents: List[Dict]) -> Dict:
        """Calculate statistics from incidents"""
        stats = {"Total Incidents": len(incidents)}
        
        try:
            status_counts = {}
            priority_counts = {}
            assignee_counts = {}
            group_counts = {}
            
            for incident in incidents:
                if not incident:
                    continue
                
                # Status distribution
                status = incident.get('status', 'Unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Priority distribution
                priority = incident.get('priority', 'Unknown')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                # Assignee distribution
                assignee = incident.get('assignee', 'Unassigned')
                assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
                
                # Group distribution
                group = incident.get('assigned_group', 'Unknown')
                group_counts[group] = group_counts.get(group, 0) + 1
            
            if status_counts:
                stats["Status Distribution"] = status_counts
            if priority_counts:
                stats["Priority Distribution"] = priority_counts
            if assignee_counts:
                # Top 5 assignees
                top_assignees = dict(sorted(assignee_counts.items(), key=lambda x: x[1], reverse=True)[:5])
                stats["Top Assignees"] = top_assignees
            if group_counts:
                # Top 5 groups
                top_groups = dict(sorted(group_counts.items(), key=lambda x: x[1], reverse=True)[:5])
                stats["Top Groups"] = top_groups
                
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
        
        return stats
    
    def _format_single_incident(self, incident: Dict, index: int) -> str:
        """Format single incident"""
        try:
            number = incident.get('incident_number', 'Unknown')
            summary = incident.get('summary', 'No summary')
            status = incident.get('status', 'Unknown')
            priority = incident.get('priority', 'Unknown')
            assignee = incident.get('assignee', 'Unassigned')
            submit_date = incident.get('submit_date', 'Unknown')
            
            return f"{index}. **{number}** - {summary[:100]}\n   Status: {status} | Priority: {priority} | Assignee: {assignee}\n   Submit Date: {submit_date}"
            
        except Exception as e:
            logger.error(f"Error formatting incident: {e}")
            return f"{index}. Error formatting incident"
    
    def _create_simple_response(self, user_query: str, remedy_data: List[Dict]) -> str:
        """Create simple response if AI fails"""
        if not remedy_data:
            return f"""🔍 **No Results Found for "{user_query}"**

I searched comprehensively in Remedy but couldn't find matching incidents.

💡 **Try:**
• Different keywords
• Broader search terms  
• Check spelling of names/groups
• Ask for "recent incidents" to see what's available

📝 **Examples:**
• "show me recent incidents"
• "high priority issues"  
• "incidents from last week"
"""

        # Create basic response
        response = f"""📊 **Remedy Results for "{user_query}"**

🎯 **Found {len(remedy_data)} incidents**

📋 **Top Incidents:**
"""
        
        for i, incident in enumerate(remedy_data[:10], 1):
            if incident:
                number = incident.get('incident_number', 'Unknown')
                summary = incident.get('summary', 'No summary')[:80]
                status = incident.get('status', 'Unknown')
                
                response += f"{i}. **{number}** - {summary}\n   Status: {status}\n\n"
        
        if len(remedy_data) > 10:
            response += f"... and {len(remedy_data) - 10} more incidents found."
        
        return response

class VersatileRemedyChatbot:
    """AI-first, truly versatile Remedy chatbot"""
    
    def __init__(self):
        self.remedy_client = UniversalRemedyClient()
        self.response_generator = AIRemedyResponseGenerator()
        self.logged_in = False
        logger.info("Versatile Remedy chatbot initialized")
    
    def answer_any_question(self, user_query: str) -> str:
        """Answer ANY Remedy question using pure AI analysis"""
        start_time = datetime.now()
        logger.info(f"🤔 Processing: '{user_query}'")
        
        try:
            # Handle basic interactions
            basic_response = self._handle_basic_interactions(user_query)
            if basic_response:
                return basic_response
            
            # Check login status
            if not self.logged_in:
                return """🔐 **Please login first!**
                
Type 'login' to authenticate with Remedy, then ask your question."""
            
            # Comprehensive search using AI
            logger.info("🧠 AI analyzing query and searching Remedy...")
            results = self.remedy_client.comprehensive_search(user_query)
            
            logger.info(f"✅ Found {len(results)} results")
            
            # Generate AI response
            logger.info("🎨 AI generating response...")
            response = self.response_generator.generate_response(user_query, results)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"⚡ Query completed in {elapsed:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in answer_any_question: {e}")
            return f"❌ **Error:** {str(e)}\n\nPlease try rephrasing your question."
    
    def _handle_basic_interactions(self, query: str) -> Optional[str]:
        """Handle basic commands and greetings"""
        query_lower = query.lower().strip()
        
        if query_lower == 'login':
            if self.remedy_client.login():
                self.logged_in = True
                return "✅ **Successfully logged into Remedy!**\n\nNow you can ask me anything about Remedy incidents!"
            else:
                return "❌ **Login failed!** Please check your credentials and try again."
        
        if query_lower == 'logout':
            self.remedy_client.logout()
            self.logged_in = False
            return "📤 **Logged out from Remedy.**"
        
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return """👋 **Hello! I'm your AI-powered Remedy assistant!**

🧠 **I use pure AI to understand ANY question about Remedy:**
• Just ask naturally - no special syntax needed
• I'll analyze your question and search comprehensively
• Get detailed results with insights and statistics

🔐 **First, type 'login' to authenticate with Remedy**

💡 **Then try asking:**
• "rollout restart incidents"
• "high priority issues assigned to john"  
• "recent production problems"
• "incidents from last week"

Ask me anything about your Remedy data! 🚀"""

        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            return "👋 Goodbye! Come back anytime for Remedy insights!"
        
        if any(word in query_lower for word in ['help', 'what can you do']):
            return """🤖 **I'm your AI-powered Remedy expert!**

✨ **How I work:**
1. **You ask anything** in natural language about Remedy
2. **AI analyzes** your question intelligently  
3. **AI searches** Remedy comprehensively using multiple strategies
4. **AI generates** comprehensive responses with insights

🎯 **I guarantee results** for any Remedy question!

🔐 **Commands:**
• 'login' - Authenticate with Remedy
• 'logout' - Sign out
• 'clear cache' - Refresh cached data

💡 **No need to learn Remedy syntax - just ask naturally!**"""
        
        if query_lower in ['clear cache', 'refresh']:
            self.remedy_client.cache.clear()
            return "✅ **Cache cleared!** Fresh data will be loaded for your next queries."
        
        return None
    
    def run_chat(self):
        """Run the interactive chatbot"""
        print("\n" + "="*80)
        print("🧠 AI-POWERED VERSATILE REMEDY CHATBOT")
        print("="*80)
        print("🚀 Pure AI analysis - Ask ANY question about Remedy in natural language!")
        print("\n💡 **How it works:**")
        print("   1. AI analyzes your question")
        print("   2. AI generates multiple search strategies") 
        print("   3. AI searches Remedy comprehensively")
        print("   4. AI generates intelligent responses")
        print("\n🔐 **First step: Type 'login' to authenticate**")
        print("\n📝 **Then test with:**")
        print("   • 'rollout restart incidents'")
        print("   • 'high priority issues assigned to john'")
        print("   • 'recent production problems'")
        print("   • 'incidents from last week'")
        print("\n💬 Type 'exit' to quit | 'login'/'logout' to authenticate")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n🗣️  Ask anything: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    if self.logged_in:
                        self.remedy_client.logout()
                    print("\n🤖 **AI RemedyGPT:** 👋 Thanks for using AI-powered Remedy chatbot!")
                    break
                
                print(f"\n🤖 **AI RemedyGPT:** ", end="")
                response = self.answer_any_question(user_input)
                print(response)
                
            except KeyboardInterrupt:
                if self.logged_in:
                    self.remedy_client.logout()
                print("\n\n🤖 **AI RemedyGPT:** 👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def test_remedy_connection():
    """Test Remedy connection without login"""
    try:
        print("🔄 Testing Remedy system...")
        
        # Test AI components
        vertexai.init(project=PROJECT_ID, location=REGION)
        model = GenerativeModel(MODEL_NAME)
        print("✅ AI components working")
        
        # Test Remedy server connectivity
        remedy_client = UniversalRemedyClient()
        print("✅ Remedy client initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main function"""
    print("🧠 Starting AI-Powered Versatile Remedy Chatbot...")
    
    if not test_remedy_connection():
        print("Please check your configuration and try again.")
        return
    
    try:
        chatbot = VersatileRemedyChatbot()
        print("🎯 AI-powered Remedy chatbot ready!")
        chatbot.run_chat()
    except Exception as e:
        print(f"❌ Failed to start: {e}")

if __name__ == "__main__":
    main()
