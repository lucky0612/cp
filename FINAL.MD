#!/usr/bin/env python3
import os
import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Web framework
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# HTTP requests
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup

# AI/ML
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedAssistant")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Confluence Configuration
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "lakshya.vijay@cmegroup.com")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "RE")

# Jira Configuration
JIRA_URL = os.environ.get("JIRA_URL", "https://cmegroup.atlassian.net/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "lakshya.vijay@cmegroup.com")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "")

# Performance settings
REQUEST_TIMEOUT = 15

class ConfluenceAssistant:
    """Complete Confluence Assistant - Extracted from your working script."""
    
    def __init__(self):
        self.base_url = CONFLUENCE_URL.rstrip('/')
        self.auth = (CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "RE-AI-Python-Agent"
        })
        self.space_pages = {}
        self.page_content_cache = {}
        
        # Initialize space content
        self._load_space_content()
        
        logger.info("Confluence Assistant initialized")
    
    def test_connection(self):
        """Test connection to Confluence."""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                params={"limit": 1},
                timeout=REQUEST_TIMEOUT,
                verify=False
            )
            connected = response.status_code == 200
            logger.info(f"Confluence connection: {'SUCCESS' if connected else 'FAILED'}")
            return connected
        except Exception as e:
            logger.error(f"Confluence connection test failed: {e}")
            return False
    
    def _load_space_content(self):
        """Load metadata for all pages in the specified space."""
        try:
            if not self.test_connection():
                logger.error("Cannot load space content - connection failed")
                return
            
            logger.info(f"Loading pages from space: {CONFLUENCE_SPACE}")
            pages = self._get_all_pages_in_space(CONFLUENCE_SPACE)
            if pages:
                self.space_pages[CONFLUENCE_SPACE] = pages
                logger.info(f"Loaded {len(pages)} pages from space {CONFLUENCE_SPACE}")
        except Exception as e:
            logger.error(f"Error loading space content: {e}")
    
    def _get_all_pages_in_space(self, space_key, batch_size=50):
        """Get all pages in a space."""
        all_pages = []
        start = 0
        
        while True:
            try:
                params = {
                    "spaceKey": space_key,
                    "expand": "history",
                    "limit": batch_size,
                    "start": start
                }
                
                response = self.session.get(
                    f"{self.base_url}/rest/api/content",
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                    verify=False
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch pages: {response.status_code}")
                    break
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                all_pages.extend(results)
                
                if len(results) < batch_size:
                    break
                
                start += batch_size
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching pages: {e}")
                break
        
        return all_pages
    
    def _get_page_content(self, page_id):
        """Get content of a page."""
        try:
            if page_id in self.page_content_cache:
                return self.page_content_cache[page_id]
            
            response = self.session.get(
                f"{self.base_url}/rest/api/content/{page_id}",
                params={"expand": "body.storage"},
                timeout=REQUEST_TIMEOUT,
                verify=False
            )
            
            if response.status_code == 200:
                page_data = response.json()
                
                title = page_data.get("title", "Unknown")
                body = page_data.get("body", {})
                storage = body.get("storage", {})
                html_content = storage.get("value", "")
                
                # Extract clean text from HTML
                if html_content:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    # Remove scripts and styles
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text_content = soup.get_text(separator='\n', strip=True)
                    
                    # Clean up the text
                    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                    cleaned_text = '\n'.join(lines)
                    
                    result = {
                        "title": title,
                        "content": cleaned_text[:3000],  # Limit content size
                        "url": f"{self.base_url}/pages/viewpage.action?pageId={page_id}"
                    }
                    
                    self.page_content_cache[page_id] = result
                    return result
            
            return None
        except Exception as e:
            logger.error(f"Error getting page content: {e}")
            return None
    
    def extract_relevant_content(self, query):
        """Extract content from pages that is most relevant to the query - from your original script."""
        if not self.space_pages:
            return "No pages found in the specified Confluence space."
        
        all_pages = []
        for space_key, pages in self.space_pages.items():
            all_pages.extend(pages)
        
        # Score pages based on title relevance
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Add domain-specific terms if query is short
        domain_terms = {
            "RE", "SRE", "LOGS", "MAINTENANCE", "API", "REST", "DATABASE",
            "VIEW", "TABLE", "FIELD", "ENDPOINT", "MAPPING", "SCHEMA"
        }
        
        if len(query_words) < 3:
            query_words.update(domain_terms)
        
        # Initial scoring based on title
        candidates = []
        for page in all_pages:
            title = page.get("title", "").lower()
            title_score = sum(1 for word in query_words if word in title)
            
            if title_score > 0 or any(term in title for term in domain_terms):
                candidates.append((page, title_score))
        
        # Sort by score and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:10]
        
        if not top_candidates:
            # If no candidates, take most recently updated pages
            recent_pages = sorted(all_pages, 
                                key=lambda p: p.get("history", {}).get("lastUpdated", {}).get("when", "2000-01-01"),
                                reverse=True)[:5]
            top_candidates = [(p, 0) for p in recent_pages]
        
        logger.info(f"Selected {len(top_candidates)} candidate pages for analysis")
        
        # Get content for top candidates
        relevant_content = []
        for page, score in top_candidates[:5]:  # Limit to top 5
            page_content = self._get_page_content(page["id"])
            if page_content:
                content_block = f"--- FROM: {page_content['title']} ---\n\n"
                content_block += f"# {page_content['title']}\n\n"
                content_block += page_content['content'] + "\n\n"
                content_block += f"Source: {page_content['url']}\n"
                relevant_content.append(content_block)
        
        return "\n\n".join(relevant_content) if relevant_content else "No relevant content found."

class JiraAssistant:
    """Complete Jira Assistant - Extracted from your working script."""
    
    def __init__(self):
        self.base_url = JIRA_URL.rstrip('/')
        self.auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self._metadata = None
        logger.info("Jira Assistant initialized")
    
    def test_connection(self):
        """Test connection to Jira."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/myself",
                headers=self.headers,
                auth=self.auth,
                verify=False,
                timeout=REQUEST_TIMEOUT
            )
            connected = response.status_code == 200
            logger.info(f"Jira connection: {'SUCCESS' if connected else 'FAILED'}")
            return connected
        except Exception as e:
            logger.error(f"Jira connection test failed: {e}")
            return False
    
    def safe_get(self, obj, key, default=None):
        """Ultra-safe getter that never fails."""
        try:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default
        except:
            return default
    
    def get_jira_metadata(self):
        """Get Jira metadata for AI context."""
        if self._metadata:
            return self._metadata
        
        logger.info("Loading Jira metadata...")
        metadata = {
            'projects': [],
            'statuses': [],
            'priorities': [],
            'issue_types': []
        }
        
        try:
            # Get projects
            response = requests.get(
                f"{self.base_url}/rest/api/3/project",
                headers=self.headers,
                auth=self.auth,
                verify=False,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                projects = response.json()
                for project in projects:
                    if project:
                        metadata['projects'].append({
                            'key': self.safe_get(project, 'key', ''),
                            'name': self.safe_get(project, 'name', '')
                        })
            
            # Get statuses
            response = requests.get(
                f"{self.base_url}/rest/api/3/status",
                headers=self.headers,
                auth=self.auth,
                verify=False,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                statuses = response.json()
                for status in statuses:
                    if status:
                        metadata['statuses'].append(self.safe_get(status, 'name', ''))
            
            self._metadata = metadata
            logger.info(f"Loaded metadata: {len(metadata['projects'])} projects")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
        
        return metadata
    
    def analyze_query_and_build_jql(self, user_query):
        """Build JQL queries based on user query - from your original script."""
        words = [w for w in user_query.lower().split() if len(w) > 2]
        
        jql_queries = []
        
        if words:
            # Main search query with multiple terms
            main_terms = ' '.join(words[:3])
            jql_queries.append(f'text ~ "{main_terms}" ORDER BY updated DESC')
            
            # Individual word searches
            for word in words[:2]:
                jql_queries.append(f'summary ~ "{word}" ORDER BY updated DESC')
                jql_queries.append(f'description ~ "{word}" ORDER BY updated DESC')
        
        # Fallback: recent issues
        jql_queries.append('updated >= -30d ORDER BY updated DESC')
        
        return jql_queries[:5]  # Limit to 5 queries
    
    def execute_jql(self, jql, max_results=50):
        """Execute JQL query safely."""
        try:
            logger.info(f"Executing JQL: {jql}")
            
            params = {
                "jql": jql,
                "maxResults": max_results,
                "fields": "summary,status,priority,assignee,created,updated,description"
            }
            
            response = requests.get(
                f"{self.base_url}/rest/api/3/search",
                headers=self.headers,
                params=params,
                auth=self.auth,
                verify=False,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                issues = data.get("issues", [])
                logger.info(f"JQL returned {len(issues)} issues")
                return issues
            else:
                logger.warning(f"JQL query failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"JQL execution error: {e}")
            return []
    
    def search_issues(self, user_query):
        """Search Jira issues - complete method from your original script."""
        logger.info(f"Searching Jira for: '{user_query}'")
        
        # Build JQL queries
        jql_queries = self.analyze_query_and_build_jql(user_query)
        
        # Execute all queries and collect results
        all_issues = []
        for jql in jql_queries:
            issues = self.execute_jql(jql, max_results=20)
            all_issues.extend(issues)
        
        # Remove duplicates by key
        unique_issues = []
        seen_keys = set()
        for issue in all_issues:
            if issue:
                key = issue.get('key')
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    unique_issues.append(issue)
        
        logger.info(f"Found {len(unique_issues)} unique Jira issues")
        return unique_issues[:20]  # Limit to 20 results
    
    def format_issues_for_ai(self, issues):
        """Format Jira issues for AI processing."""
        if not issues:
            return "No Jira issues found."
        
        formatted = f"JIRA RESULTS ({len(issues)} tickets found):\n\n"
        
        for i, issue in enumerate(issues[:10], 1):  # Limit to 10 for context
            key = issue.get('key', 'Unknown')
            fields = issue.get('fields', {})
            summary = fields.get('summary', 'No summary')
            status = (fields.get('status') or {}).get('name', 'Unknown')
            priority = (fields.get('priority') or {}).get('name', 'Unknown')
            assignee_obj = fields.get('assignee')
            assignee = assignee_obj.get('displayName', 'Unassigned') if assignee_obj else 'Unassigned'
            
            formatted += f"{i}. **[{key}](https://cmegroup.atlassian.net/browse/{key})** - {summary}\n"
            formatted += f"   Status: {status} | Priority: {priority} | Assignee: {assignee}\n\n"
        
        if len(issues) > 10:
            formatted += f"... and {len(issues) - 10} more tickets found.\n"
        
        return formatted

class UnifiedAIAssistant:
    """AI Assistant that combines both Confluence and Jira data."""
    
    def __init__(self):
        try:
            vertexai.init(project=PROJECT_ID, location=REGION)
            self.model = GenerativeModel(MODEL_NAME)
            logger.info(f"AI Assistant initialized with {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
            self.model = None
    
    def generate_unified_response(self, user_query, confluence_data, jira_data, selected_sources):
        """Generate unified response combining both sources."""
        if not self.model:
            return "AI service is not available. Please check the configuration."
        
        try:
            logger.info("Generating AI response...")
            
            # Build the prompt
            prompt = f"""You are the friendly CME Knowledge Assistant, an expert on both documentation and project tracking.

Your personality:
- Conversational and approachable - use a casual, helpful tone while maintaining workplace professionalism
- Explain concepts clearly, as if speaking to a colleague
- Use examples and context to clarify complex ideas
- Be concise but thorough - focus on answering the question directly first, then add helpful context

USER QUESTION: "{user_query}"
SELECTED SOURCES: {selected_sources}

CONFLUENCE DOCUMENTATION:
{confluence_data if confluence_data else "No Confluence data available"}

JIRA TICKETS:
{jira_data if jira_data else "No Jira data available"}

Provide a comprehensive response that:
1. Directly answers the user's question
2. Combines information from available sources intelligently
3. Uses clear formatting with sections and bullet points
4. Includes relevant ticket links for Jira items (already formatted)
5. Is professional but conversational
6. Highlights key insights and actionable information

If both sources have relevant information, show how they complement each other.
If only one source has information, focus on that but mention what was searched.

Response:"""

            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=3000
                )
            )
            
            result = response.text if hasattr(response, 'text') else response.candidates[0].text
            logger.info("AI response generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            # Return a helpful fallback response
            fallback = f"I found information for your query about '{user_query}':\n\n"
            
            if confluence_data and "No Confluence data" not in confluence_data:
                fallback += "**From Confluence Documentation:**\n"
                fallback += confluence_data[:1000] + "...\n\n"
            
            if jira_data and "No Jira data" not in jira_data:
                fallback += "**From Jira Tickets:**\n"
                fallback += jira_data[:1000] + "...\n\n"
            
            fallback += "Note: I had trouble formatting this response, but the information above should help answer your question."
            return fallback

class UnifiedKnowledgeAssistant:
    """Main unified assistant that orchestrates everything."""
    
    def __init__(self):
        logger.info("🚀 Initializing Unified Knowledge Assistant...")
        
        self.confluence = None
        self.jira = None
        self.ai = UnifiedAIAssistant()
        
        # Initialize clients based on available tokens
        if CONFLUENCE_API_TOKEN:
            try:
                self.confluence = ConfluenceAssistant()
                logger.info("✅ Confluence client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Confluence: {e}")
        else:
            logger.warning("⚠️  Confluence API token not provided")
        
        if JIRA_API_TOKEN:
            try:
                self.jira = JiraAssistant()
                logger.info("✅ Jira client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Jira: {e}")
        else:
            logger.warning("⚠️  Jira API token not provided")
        
        logger.info("🎉 Unified Knowledge Assistant ready!")
    
    def process_query(self, user_query, selected_sources):
        """Process user query and return unified response."""
        start_time = time.time()
        logger.info(f"🔍 Processing query: '{user_query}' with sources: {selected_sources}")
        
        try:
            confluence_data = ""
            jira_data = ""
            
            # Search Confluence if selected and available
            if ("confluence" in selected_sources or "both" in selected_sources) and self.confluence:
                logger.info("📚 Searching Confluence...")
                try:
                    confluence_data = self.confluence.extract_relevant_content(user_query)
                    logger.info("✅ Confluence search completed")
                except Exception as e:
                    logger.error(f"❌ Confluence search failed: {e}")
                    confluence_data = f"Error searching Confluence: {str(e)}"
            else:
                logger.info("⏭️  Skipping Confluence (not selected or unavailable)")
            
            # Search Jira if selected and available
            if ("jira" in selected_sources or "both" in selected_sources) and self.jira:
                logger.info("🎫 Searching Jira...")
                try:
                    jira_issues = self.jira.search_issues(user_query)
                    jira_data = self.jira.format_issues_for_ai(jira_issues)
                    logger.info("✅ Jira search completed")
                except Exception as e:
                    logger.error(f"❌ Jira search failed: {e}")
                    jira_data = f"Error searching Jira: {str(e)}"
            else:
                logger.info("⏭️  Skipping Jira (not selected or unavailable)")
            
            # Generate unified response
            logger.info("🤖 Generating AI response...")
            response = self.ai.generate_unified_response(user_query, confluence_data, jira_data, selected_sources)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Query completed in {elapsed:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return f"I encountered an error while processing your query: {str(e)}. Please try again."
    
    def get_status(self):
        """Get system status."""
        logger.info("📊 Checking system status...")
        
        confluence_status = False
        jira_status = False
        confluence_pages = 0
        
        if self.confluence:
            confluence_status = self.confluence.test_connection()
            if confluence_status:
                confluence_pages = len(self.confluence.space_pages.get(CONFLUENCE_SPACE, []))
        
        if self.jira:
            jira_status = self.jira.test_connection()
        
        status = {
            'confluence': confluence_status,
            'jira': jira_status,
            'confluence_pages': confluence_pages,
            'status': 'healthy' if confluence_status or jira_status else 'degraded'
        }
        
        logger.info(f"📊 System status: {status}")
        return status

# Flask Application
app = Flask(__name__)
CORS(app)

# Initialize the assistant
logger.info("🚀 Starting CME Unified Knowledge Assistant...")
assistant = UnifiedKnowledgeAssistant()

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    logger.info("📨 Received chat request")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_query = data.get('message', '').strip()
        selected_sources = data.get('sources', [])
        
        logger.info(f"Query: '{user_query}', Sources: {selected_sources}")
        
        if not user_query:
            return jsonify({'error': 'No message provided'}), 400
        
        if not selected_sources:
            return jsonify({'error': 'No sources selected'}), 400
        
        # Handle basic greetings
        query_lower = user_query.lower()
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            response = """👋 **Hello! I'm your CME Unified Knowledge Assistant!**

🔍 **I can help you with:**
• **Confluence:** Documentation, procedures, how-to guides
• **Jira:** Tickets, issues, bugs, project tracking  
• **Both:** Combined insights across systems

💡 **Just select your data sources and ask naturally!**

🚀 **Try asking:**
• "Show me rollout restart tickets and how to solve them"
• "BAMPS project documentation"
• "Recent high priority issues"
• "How to handle system maintenance"

What would you like to explore? 🤔"""
            return jsonify({'response': response})
        
        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            response = "👋 **Goodbye!** Feel free to come back anytime for insights across CME's knowledge systems! 🚀"
            return jsonify({'response': response})
        
        # Process the query
        response = assistant.process_query(user_query, selected_sources)
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Check system status."""
    try:
        status_data = assistant.get_status()
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"❌ Error in status endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting CME Unified Knowledge Assistant...")
    print(f"🔗 Confluence: {CONFLUENCE_URL}")
    print(f"🎫 Jira: {JIRA_URL}")
    print(f"🤖 AI Model: {MODEL_NAME}")
    print(f"📊 Confluence Token: {'✅ SET' if CONFLUENCE_API_TOKEN else '❌ MISSING'}")
    print(f"📊 Jira Token: {'✅ SET' if JIRA_API_TOKEN else '❌ MISSING'}")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
