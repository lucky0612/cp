#!/usr/bin/env python3
"""
Simple Jira-Gemini AI Chatbot
-----------------------------
A streamlined integration between Jira and Gemini AI.
"""

import logging
import os
import sys
import json
import re
import time
import pickle
import requests
from pathlib import Path
from typing import Dict, List, Any
import urllib3
from datetime import datetime, timedelta
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JiraBot")

# =====================================================================
# HARDCODED CREDENTIALS - REPLACE WITH YOUR ACTUAL VALUES
# =====================================================================
# Jira Credentials
JIRA_BASE_URL = "https://your-domain.atlassian.net"
JIRA_USERNAME = "your-username@example.com"
JIRA_API_TOKEN = "your-api-token"

# Google Cloud Configuration
PROJECT_ID = "prj-dv-cws-4363"
REGION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-001"

# Cache file path
CACHE_FILE = Path.home() / ".jira_cache.pickle"


class JiraClient:
    """Simple Jira client with caching"""
    
    def __init__(self, base_url, username, api_token):
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.verify = False  # Disable SSL verification
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk or create new"""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Cache loaded with {len(cache.get('issues', {}))} issues")
                return cache
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        # Return new empty cache
        return {
            "issues": {},
            "projects": {},
            "last_sync": None
        }
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info("Cache saved")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def request(self, endpoint, method="GET", params=None, data=None):
        """Make a request to the Jira API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code >= 400:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return None
            
            return response.json()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_projects(self):
        """Get all projects"""
        if "projects_list" in self.cache:
            return self.cache["projects_list"]
        
        logger.info("Fetching projects from Jira")
        result = self.request("/rest/api/3/project")
        
        if result:
            self.cache["projects_list"] = result
            self.cache["projects"] = {p["key"]: p for p in result}
            self._save_cache()
        
        return result or []
    
    def search_issues(self, jql, max_results=100):
        """Search for issues using JQL"""
        # Check cache first
        cache_key = f"search_{jql}_{max_results}"
        if cache_key in self.cache:
            logger.info(f"Using cached search results for: {jql}")
            return self.cache[cache_key]
        
        logger.info(f"Searching issues: {jql}")
        result = self.request("/rest/api/3/search", params={
            "jql": jql,
            "maxResults": max_results
        })
        
        # Cache the result
        if result:
            self.cache[cache_key] = result
            
            # Also cache individual issues
            if "issues" in result:
                for issue in result["issues"]:
                    issue_key = issue["key"]
                    self.cache["issues"][issue_key] = {
                        "basic_data": issue,
                        "timestamp": datetime.now().isoformat()
                    }
            
            self._save_cache()
        
        return result or {"issues": []}
    
    def get_issue(self, issue_key):
        """Get a specific issue by key"""
        # Check if we have full data cached
        if issue_key in self.cache["issues"] and "full_data" in self.cache["issues"][issue_key]:
            logger.info(f"Using cached data for {issue_key}")
            return self.cache["issues"][issue_key]["full_data"]
        
        logger.info(f"Fetching issue: {issue_key}")
        result = self.request(f"/rest/api/3/issue/{issue_key}", params={
            "expand": "renderedFields,names,schema,operations,editmeta,changelog"
        })
        
        # Cache the result
        if result:
            if issue_key not in self.cache["issues"]:
                self.cache["issues"][issue_key] = {}
            
            self.cache["issues"][issue_key]["full_data"] = result
            self.cache["issues"][issue_key]["timestamp"] = datetime.now().isoformat()
            self._save_cache()
        
        return result or {"key": issue_key, "error": "Failed to fetch issue"}
    
    def find_issues_for_query(self, query, max_results=10):
        """Find issues relevant to a user query"""
        # Extract ticket keys if mentioned directly
        ticket_pattern = r'([A-Z]+-\d+)'
        explicit_tickets = re.findall(ticket_pattern, query)
        
        if explicit_tickets:
            logger.info(f"Found explicit ticket keys: {explicit_tickets}")
            return explicit_tickets
        
        # Simple keyword extraction
        words = query.split()
        words = [w for w in words if len(w) > 2 and w.lower() not in {
            "the", "and", "for", "with", "what", "how", "when", "where", "who", "which", 
            "that", "this", "these", "those"
        }]
        
        # Use up to 3 keywords for search
        keywords = words[:3]
        if not keywords:
            # If no good keywords, search recent issues
            logger.info("No good keywords, using recent issues")
            result = self.search_issues("ORDER BY updated DESC", max_results)
            return [issue["key"] for issue in result.get("issues", [])]
        
        # Build JQL
        jql = " AND ".join([f'text ~ "{k}"' for k in keywords])
        jql += " ORDER BY updated DESC"
        
        logger.info(f"Searching with JQL: {jql}")
        result = self.search_issues(jql, max_results)
        
        return [issue["key"] for issue in result.get("issues", [])]
    
    def sync_recent_data(self):
        """Sync recent Jira data"""
        logger.info("Syncing recent Jira data")
        
        # Get projects
        self.get_projects()
        
        # Get recent issues
        result = self.search_issues("updated >= -30d ORDER BY updated DESC", 100)
        
        # Get full data for each issue
        if "issues" in result:
            for issue in result["issues"]:
                self.get_issue(issue["key"])
        
        self.cache["last_sync"] = datetime.now().isoformat()
        self._save_cache()
        logger.info("Sync completed")


class GeminiClient:
    """Client for Gemini AI"""
    
    def __init__(self, project_id, location, model_name):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
    
    def generate_response(self, prompt, temperature=0.7):
        """Generate a response from Gemini"""
        try:
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=8192
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if hasattr(response, 'text'):
                return response.text
            else:
                return "I couldn't generate a response to your query."
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return f"Error generating response: {str(e)}"


class JiraGeminiChatbot:
    """Main chatbot class"""
    
    def __init__(self):
        # Initialize clients
        self.jira_client = JiraClient(JIRA_BASE_URL, JIRA_USERNAME, JIRA_API_TOKEN)
        self.gemini_client = GeminiClient(PROJECT_ID, REGION, MODEL_NAME)
    
    def initialize(self):
        """Initialize the chatbot"""
        # Check if we need to sync data
        last_sync = self.jira_client.cache.get("last_sync")
        sync_needed = True
        
        if last_sync:
            last_sync_date = datetime.fromisoformat(last_sync)
            sync_needed = (datetime.now() - last_sync_date).days >= 1
        
        if sync_needed:
            print("Syncing Jira data... (this may take a minute)")
            self.jira_client.sync_recent_data()
            print("Sync completed")
        else:
            print("Using cached Jira data")
    
    def format_issue_for_prompt(self, issue_data):
        """Format an issue for inclusion in the prompt"""
        if not issue_data or "fields" not in issue_data:
            return ""
        
        fields = issue_data["fields"]
        text = f"--- {issue_data['key']} ---\n"
        text += f"Summary: {fields.get('summary', 'N/A')}\n"
        text += f"Status: {fields.get('status', {}).get('name', 'Unknown')}\n"
        
        # Add description
        description = fields.get('description', '')
        if description:
            if isinstance(description, dict) and "content" in description:
                desc_text = self._extract_text_from_adf(description)
                text += f"Description: {desc_text}\n"
            elif isinstance(description, str):
                text += f"Description: {description}\n"
        
        # Add link
        text += f"Link: {JIRA_BASE_URL}/browse/{issue_data['key']}\n\n"
        
        return text
    
    def _extract_text_from_adf(self, adf_doc):
        """Extract text from Atlassian Document Format"""
        if not isinstance(adf_doc, dict) or "content" not in adf_doc:
            return str(adf_doc)
        
        text_parts = []
        
        def extract_text_recursive(node):
            if isinstance(node, dict):
                if node.get("type") == "text" and "text" in node:
                    text_parts.append(node["text"])
                elif "content" in node and isinstance(node["content"], list):
                    for item in node["content"]:
                        extract_text_recursive(item)
            elif isinstance(node, list):
                for item in node:
                    extract_text_recursive(item)
        
        extract_text_recursive(adf_doc)
        return " ".join(text_parts)
    
    def answer_question(self, query):
        """Process and answer a question"""
        # Find relevant issues
        issue_keys = self.jira_client.find_issues_for_query(query)
        
        if not issue_keys:
            return "I couldn't find any relevant Jira issues for your query."
        
        # Get full data for each issue
        issues_data = []
        for key in issue_keys[:5]:  # Limit to 5 issues
            issue = self.jira_client.get_issue(key)
            if issue and "error" not in issue:
                issues_data.append(issue)
        
        if not issues_data:
            return "I found some issues but couldn't retrieve their details."
        
        # Format issues for the prompt
        context = "Relevant Jira issues:\n\n"
        for issue in issues_data:
            context += self.format_issue_for_prompt(issue)
        
        # Create the prompt
        prompt = f"""You are a helpful assistant for Jira who always provides direct, concise answers.

User query: {query}

{context}

Please provide a complete answer to the user's query based on the information above.
Include links to relevant Jira tickets. Format Jira ticket keys as [TICKET-123].
"""
        
        # Generate response
        response = self.gemini_client.generate_response(prompt)
        
        # Fix up ticket references
        ticket_pattern = r'\[([A-Z]+-\d+)\]'
        response = re.sub(
            ticket_pattern,
            lambda m: f"{m.group(1)} ({JIRA_BASE_URL}/browse/{m.group(1)})",
            response
        )
        
        return response
    
    def run(self):
        """Run the chatbot in interactive mode"""
        print("Jira-Gemini Chatbot initialized")
        print("Ask me anything about your Jira tickets (type 'exit' to quit)")
        
        while True:
            query = input("\nYou: ")
            
            if query.lower() in ('exit', 'quit', 'bye'):
                print("Goodbye!")
                break
            
            if query.lower() == 'sync':
                print("Syncing Jira data...")
                self.jira_client.sync_recent_data()
                print("Sync completed")
                continue
            
            print("\nThinking...")
            response = self.answer_question(query)
            print(f"\nAssistant: {response}")


def main():
    """Main function"""
    try:
        chatbot = JiraGeminiChatbot()
        chatbot.initialize()
        chatbot.run()
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()

















