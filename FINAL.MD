#!/usr/bin/env python3
import os
import json
import pickle
import time
import re
import logging
import requests
from requests.auth import HTTPBasicAuth
import base64
from datetime import datetime, timedelta
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError
import concurrent.futures
import urllib3

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

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Jira Configuration
JIRA_URL = os.environ.get("JIRA_URL", "https://cmegroup.atlassian.net/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "your.email@cmegroup.com")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "your-api-token")

# Cache configuration
CACHE_DIR = "jira_cache"
CACHE_EXPIRY = 24  # Cache expiry in hours

class JiraCache:
    """Class to handle caching of Jira data"""
    
    def __init__(self, cache_dir=CACHE_DIR, expiry_hours=CACHE_EXPIRY):
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        """Get the file path for a cache key"""
        # Create a safe filename from the cache key
        safe_key = re.sub(r'[^\w]', '_', key)
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def get(self, key):
        """Get data from cache if it exists and is not expired"""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is expired
            cache_time = cache_data.get('timestamp')
            if cache_time and datetime.now() - cache_time > timedelta(hours=self.expiry_hours):
                logger.info(f"Cache expired for key: {key}")
                return None
            
            logger.info(f"Cache hit for key: {key}")
            return cache_data.get('data')
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, key, data):
        """Save data to cache with current timestamp"""
        cache_path = self._get_cache_path(key)
        
        try:
            cache_data = {
                'data': data,
                'timestamp': datetime.now()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache set for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
            return False
    
    def clear(self, key=None):
        """Clear specific cache or all cache if key is None"""
        if key:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cleared cache for key: {key}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("Cleared all cache data")


class JiraClient:
    """Class to handle interactions with Jira API"""
    
    def __init__(self, base_url=JIRA_URL, email=JIRA_EMAIL, api_token=JIRA_API_TOKEN):
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.cache = JiraCache()
    
    def make_request(self, method, endpoint, params=None, data=None, cache_key=None, force_refresh=False):
        """Make an HTTP request to Jira API with caching support"""
        url = f"{self.base_url}{endpoint}"
        
        # Check cache if applicable
        if method.lower() == 'get' and cache_key and not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        try:
            response = requests.request(
                method,
                url,
                headers=self.headers,
                params=params,
                json=data,
                auth=self.auth,
                verify=False  # Disable SSL verification as requested
            )
            
            # Handle HTTP errors
            response.raise_for_status()
            
            if response.status_code == 204:  # No content
                result = {"status": "success"}
            else:
                result = response.json()
            
            # Cache the result if applicable
            if method.lower() == 'get' and cache_key:
                self.cache.set(cache_key, result)
            
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            return {"error": str(e)}
    
    def get_myself(self, force_refresh=False):
        """Get information about the authenticated user"""
        return self.make_request('GET', '/rest/api/3/myself', cache_key='myself', force_refresh=force_refresh)
    
    def search_issues(self, jql, fields=None, max_results=50, force_refresh=False):
        """Search for issues using JQL"""
        if fields is None:
            fields = ["summary", "description", "status", "assignee", "reporter", "created", "updated", "priority", "issuetype", "labels", "comment"]
        
        params = {
            "jql": jql,
            "maxResults": max_results,
            "fields": ",".join(fields),
            "expand": "renderedFields,names,schema,transitions,operations,editmeta,changelog"
        }
        
        cache_key = f"search_issues_{jql}_{max_results}"
        return self.make_request('GET', '/rest/api/3/search', params=params, cache_key=cache_key, force_refresh=force_refresh)
    
    def get_issue(self, issue_key, fields=None, expand=None, force_refresh=False):
        """Get detailed information about a specific issue"""
        params = {}
        if fields:
            params["fields"] = ",".join(fields)
        if expand:
            params["expand"] = expand
        
        cache_key = f"issue_{issue_key}"
        return self.make_request('GET', f'/rest/api/3/issue/{issue_key}', params=params, cache_key=cache_key, force_refresh=force_refresh)
    
    def get_issue_comments(self, issue_key, force_refresh=False):
        """Get comments for a specific issue"""
        cache_key = f"comments_{issue_key}"
        return self.make_request('GET', f'/rest/api/3/issue/{issue_key}/comment', cache_key=cache_key, force_refresh=force_refresh)
    
    def get_issue_attachments(self, issue_key, force_refresh=False):
        """Get attachments for a specific issue"""
        cache_key = f"attachments_{issue_key}"
        return self.make_request('GET', f'/rest/api/3/issue/{issue_key}/attachment', cache_key=cache_key, force_refresh=force_refresh)
    
    def get_projects(self, force_refresh=False):
        """Get all accessible projects"""
        cache_key = "projects"
        return self.make_request('GET', '/rest/api/3/project', cache_key=cache_key, force_refresh=force_refresh)
    
    def get_project(self, project_key, force_refresh=False):
        """Get detailed information about a specific project"""
        cache_key = f"project_{project_key}"
        return self.make_request('GET', f'/rest/api/3/project/{project_key}', cache_key=cache_key, force_refresh=force_refresh)
    
    def get_all_issue_types(self, force_refresh=False):
        """Get all issue types"""
        cache_key = "issue_types"
        return self.make_request('GET', '/rest/api/3/issuetype', cache_key=cache_key, force_refresh=force_refresh)
    
    def get_all_fields(self, force_refresh=False):
        """Get all available fields"""
        cache_key = "fields"
        return self.make_request('GET', '/rest/api/3/field', cache_key=cache_key, force_refresh=force_refresh)
    
    def download_attachment(self, attachment_url, filename=None, force_refresh=False):
        """Download an attachment file"""
        if not filename:
            filename = attachment_url.split('/')[-1]
        
        cache_key = f"attachment_data_{filename}"
        cached_data = None if force_refresh else self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            response = requests.get(
                attachment_url,
                auth=self.auth,
                verify=False  # Disable SSL verification
            )
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            content = response.content
            
            if 'image' in content_type:
                # For images, encode as base64 for sending to Gemini
                encoded_data = base64.b64encode(content).decode('utf-8')
                result = {
                    'content_type': content_type,
                    'filename': filename,
                    'data': encoded_data,
                    'is_image': True
                }
            else:
                # For text-based files, try to decode
                try:
                    decoded_content = content.decode('utf-8')
                    result = {
                        'content_type': content_type,
                        'filename': filename,
                        'data': decoded_content,
                        'is_image': False
                    }
                except UnicodeDecodeError:
                    # Binary file, encode as base64
                    encoded_data = base64.b64encode(content).decode('utf-8')
                    result = {
                        'content_type': content_type,
                        'filename': filename,
                        'data': encoded_data,
                        'is_image': False,
                        'is_binary': True
                    }
            
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error downloading attachment: {e}")
            return {"error": str(e)}
    
    def extract_tables_from_html(self, html_content):
        """Extract table data from HTML content"""
        # Simple regex-based table extraction
        table_pattern = r'<table[^>]*>(.*?)</table>'
        tables = re.findall(table_pattern, html_content, re.DOTALL)
        
        extracted_tables = []
        for table_html in tables:
            rows = []
            row_pattern = r'<tr[^>]*>(.*?)</tr>'
            for row_html in re.findall(row_pattern, table_html, re.DOTALL):
                # Extract header cells
                headers = re.findall(r'<th[^>]*>(.*?)</th>', row_html, re.DOTALL)
                if headers:
                    row = [re.sub(r'<[^>]+>', '', cell).strip() for cell in headers]
                    rows.append(row)
                
                # Extract regular cells
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
                if cells:
                    row = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
                    rows.append(row)
            
            if rows:
                extracted_tables.append(rows)
        
        return extracted_tables


class GeminiChat:
    """Class for interacting with Gemini models via Vertex AI"""
    
    def __init__(self, project_id=PROJECT_ID, location=REGION, model_name=MODEL_NAME):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.model_name)
        
        # System prompt - comprehensive instructions for Gemini
        self.system_prompt = """
You are JiraGPT, an intelligent assistant designed to help answer questions about Jira projects, tickets, and tasks.

Key capabilities:
1. Provide clear, concise, and accurate answers about Jira tickets, projects, and workflows
2. Format responses appropriately based on the query (tables, bullet points, paragraphs)
3. Include relevant links to Jira pages when applicable
4. Ask clarifying questions when needed to provide more accurate answers
5. Understand both technical and non-technical questions
6. Analyze tables and images from Jira tickets
7. Be capable of explaining Jira concepts in simple terms
8. Stay professional but friendly in your tone

When you don't know the answer:
- Clearly state that you don't have enough information
- Ask specific questions to help narrow down what the user is looking for
- Suggest alternative approaches or places to look for the information

When providing links, always use the format [Title](URL) for better readability.

When a user asks about specific tickets:
- Summarize key information like status, assignee, priority, and description
- Highlight recent updates or comments
- Include a direct link to the ticket

Always respond in a structured way that makes information easy to scan and understand.
"""
    
    def generate_response(self, user_query, jira_context=None, image_data=None):
        """
        Generate a response using Gemini based on the user query and Jira context
        
        Args:
            user_query (str): The user's question
            jira_context (dict): Relevant Jira data to help answer the question
            image_data (list): List of image data to include in the query
            
        Returns:
            str: The generated response
        """
        try:
            # Prepare context for Gemini
            context = f"{self.system_prompt}\n\n"
            
            if jira_context:
                context += "Here is relevant information from Jira to help answer the question:\n"
                context += json.dumps(jira_context, indent=2)
                context += "\n\n"
            
            # Combine context and user query
            full_prompt = f"{context}User question: {user_query}\n\nYour response:"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.2,  # Lower temperature for more factual responses
                top_p=0.95,
                max_output_tokens=8192,
            )
            
            # Generate response - with or without images
            if image_data:
                # Handle multimodal content (text + images)
                from vertexai.generative_models import Part, Content
                
                content_parts = [Part.from_text(full_prompt)]
                
                # Add image parts
                for img in image_data:
                    if img.get('mime_type') and img.get('data'):
                        image_part = Part.from_data(
                            data=base64.b64decode(img['data']),
                            mime_type=img['mime_type']
                        )
                        content_parts.append(image_part)
                
                content = Content(
                    role="user",
                    parts=content_parts
                )
                
                response = self.model.generate_content(
                    content,
                    generation_config=generation_config,
                )
            else:
                # Text-only response
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )
            
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].text
            else:
                return "I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your question. Please try again or rephrase your question. Technical details: {str(e)}"


class JiraChatbot:
    """Main class that integrates Jira client with Gemini AI for a chatbot experience"""
    
    def __init__(self):
        self.jira_client = JiraClient()
        self.gemini_chat = GeminiChat()
        # Store conversation history for context
        self.conversation_history = []
        # Maximum history to keep
        self.max_history_length = 10
    
    def add_to_history(self, user_query, bot_response):
        """Add an exchange to the conversation history"""
        self.conversation_history.append({
            "user": user_query,
            "bot": bot_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the most recent exchanges
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def extract_jira_issue_keys(self, text):
        """Extract Jira issue keys from text using regex"""
        # Pattern for Jira issue keys like PROJECT-123
        pattern = r'[A-Z]+-\d+'
        return re.findall(pattern, text)
    
    def get_relevant_jira_data(self, query):
        """
        Get relevant Jira data based on the user query
        
        This function analyzes the query and fetches appropriate data from Jira
        """
        relevant_data = {}
        image_data = []
        
        try:
            # Extract issue keys mentioned in the query
            issue_keys = self.extract_jira_issue_keys(query)
            
            if issue_keys:
                # Get details for each mentioned issue
                issues_data = []
                for issue_key in issue_keys:
                    issue = self.jira_client.get_issue(issue_key)
                    if "error" not in issue:
                        issues_data.append(issue)
                        
                        # Get comments for the issue
                        comments = self.jira_client.get_issue_comments(issue_key)
                        if "error" not in comments:
                            issue["comments"] = comments
                        
                        # Get attachments for the issue
                        attachments = self.jira_client.get_issue_attachments(issue_key)
                        if "error" not in attachments and "values" in attachments:
                            issue["attachments"] = attachments
                            
                            # Download and process image attachments for Gemini
                            for attachment in attachments.get("values", []):
                                if "content" in attachment and "mimeType" in attachment:
                                    if attachment["mimeType"].startswith("image/"):
                                        attachment_data = self.jira_client.download_attachment(attachment["content"])
                                        if "error" not in attachment_data and attachment_data.get("is_image"):
                                            image_data.append({
                                                "mime_type": attachment_data["content_type"],
                                                "data": attachment_data["data"],
                                                "filename": attachment_data["filename"]
                                            })
                
                if issues_data:
                    relevant_data["issues"] = issues_data
            
            # Search for relevant issues if no specific issue keys were mentioned
            else:
                # Extract potential search terms
                search_terms = query.lower()
                # Remove common words that aren't useful for searching
                stop_words = ["the", "and", "is", "in", "it", "of", "for", "a", "on", "with", "about"]
                for word in stop_words:
                    search_terms = search_terms.replace(f" {word} ", " ")
                
                # Create a JQL query based on search terms
                jql = f'text ~ "{search_terms}" ORDER BY updated DESC'
                
                # Search for issues
                search_results = self.jira_client.search_issues(jql, max_results=5)
                if "error" not in search_results and "issues" in search_results:
                    relevant_data["search_results"] = search_results
            
            # Check for project-related queries
            project_pattern = r'project[s]?\s+([A-Z]+)'
            project_matches = re.findall(project_pattern, query, re.IGNORECASE)
            
            if project_matches:
                projects_data = []
                for project_key in project_matches:
                    project = self.jira_client.get_project(project_key.upper())
                    if "error" not in project:
                        projects_data.append(project)
                
                if projects_data:
                    relevant_data["projects"] = projects_data
            
            # If no specific data found yet, get general account info
            if not relevant_data:
                myself = self.jira_client.get_myself()
                if "error" not in myself:
                    relevant_data["user"] = myself
                
                # Get a list of projects the user has access to
                projects = self.jira_client.get_projects()
                if "error" not in projects:
                    relevant_data["available_projects"] = projects
            
            return relevant_data, image_data
            
        except Exception as e:
            logger.error(f"Error getting relevant Jira data: {e}")
            return {"error": str(e)}, []
    
    def process_query(self, query):
        """
        Process a user query and return a response
        
        This is the main method that handles the chatbot interaction
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        try:
            # Check if it's a query to clear the cache
            if query.lower().strip() in ["clear cache", "refresh cache", "reset cache"]:
                self.jira_client.cache.clear()
                response = "Cache cleared. I'll fetch fresh data for your next queries."
                self.add_to_history(query, response)
                return response
            
            # Check if it's an irrelevant query
            irrelevant_patterns = [
                r'hello|hi|hey',
                r'thank|thanks',
                r'bye|goodbye',
                r'how are you',
                r'what can you do',
                r'help me'
            ]
            
            for pattern in irrelevant_patterns:
                if re.match(f"^{pattern}$", query.lower().strip()):
                    # Handle common greetings and small talk
                    if pattern == r'hello|hi|hey':
                        response = "Hello! I'm JiraGPT, your Jira assistant. How can I help you with your Jira queries today?"
                    elif pattern == r'thank|thanks':
                        response = "You're welcome! Feel free to ask if you need any more help with Jira."
                    elif pattern == r'bye|goodbye':
                        response = "Goodbye! Feel free to come back when you need help with Jira."
                    elif pattern == r'how are you':
                        response = "I'm functioning well, thank you! How can I help you with Jira today?"
                    elif pattern == r'what can you do':
                        response = "I can help you with various Jira-related tasks, such as:\n\n" \
                                  "- Finding information about specific tickets (e.g., 'Tell me about RDRF-2606')\n" \
                                  "- Searching for tickets with specific criteria\n" \
                                  "- Explaining Jira concepts and workflows\n" \
                                  "- Analyzing ticket data, including images and tables\n" \
                                  "- Providing links to relevant Jira pages\n\n" \
                                  "How can I assist you today?"
                    elif pattern == r'help me':
                        response = "I'd be happy to help! To get started, try asking specific questions about your Jira tickets, projects, or workflows. For example:\n\n" \
                                  "- 'Tell me about ticket RDRF-2606'\n" \
                                  "- 'What are the open issues in my RDS-SPHERE project?'\n" \
                                  "- 'How do I create alerts in Jira?'\n\n" \
                                  "The more specific your question, the better I can assist you!"
                    
                    self.add_to_history(query, response)
                    return response
            
            # Get relevant data from Jira for the query
            relevant_data, image_data = self.get_relevant_jira_data(query)
            
            # Check if we need to ask a clarifying question
            if not relevant_data or ("error" in relevant_data and not relevant_data.get("user")):
                # Simplified logic for asking clarifying questions
                if "ticket" in query.lower() or "issue" in query.lower():
                    response = "I need more information to help you. Could you provide the specific ticket ID (e.g., RDRF-2606)?"
                elif "project" in query.lower():
                    response = "I need more information to help you. Could you provide the specific project key or name?"
                else:
                    # Generate a response with limited context
                    history_context = {
                        "conversation_history": self.conversation_history[-3:] if self.conversation_history else []
                    }
                    response = self.gemini_chat.generate_response(query, history_context)
            else:
                # Add conversation history to the context
                relevant_data["conversation_history"] = self.conversation_history[-3:] if self.conversation_history else []
                
                # Generate response using Gemini
                response = self.gemini_chat.generate_response(query, relevant_data, image_data)
            
            # Add the exchange to conversation history
            self.add_to_history(query, response)
            
            # Log processing time
            elapsed_time = time.time() - start_time
            logger.info(f"Query processed in {elapsed_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error while processing your question. Please try again or rephrase your question. Technical details: {str(e)}"


def run_chatbot():
    """Run the chatbot in an interactive loop"""
    print("\n===== JiraGPT Chatbot =====")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear cache' to refresh Jira data.")
    
    chatbot = JiraChatbot()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nChatbot: Goodbye! Have a great day.")
            break
        
        response = chatbot.process_query(user_input)
        print(f"\nChatbot: {response}")


if __name__ == "__main__":
    run_chatbot()
