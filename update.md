#!/usr/bin/env python3
import json
import requests
import logging
import os
import sys
import urllib3
import re
import base64
from datetime import datetime, timedelta
import time
import getpass
from urllib.parse import quote
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError
import pandas as pd
from tabulate import tabulate
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict
import string

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Disable SSL warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("remedy_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RemedyChatbot")

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
MULTIMODAL_MODEL = os.environ.get("MULTIMODAL_MODEL", "gemini-2.0-pro-vision")

# Hard-coded credentials (REPLACE THESE WITH YOUR ACTUAL CREDENTIALS)
DEFAULT_SERVER_URL = "https://cmegroup-restapi.onbmc.com"
DEFAULT_USERNAME = "your_username_here"  # Replace with your actual username
DEFAULT_PASSWORD = "your_password_here"  # Replace with your actual password

# Cache file locations
CACHE_DIR = "remedy_cache"
INCIDENTS_CACHE_FILE = os.path.join(CACHE_DIR, "incidents_cache.pkl")
CACHE_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

class RemedyClient:
    """
    Client for BMC Remedy REST API operations with comprehensive error handling and
    advanced querying capabilities.
    """
    def __init__(self, server_url=DEFAULT_SERVER_URL, username=DEFAULT_USERNAME, 
                 password=DEFAULT_PASSWORD, ssl_verify=False):
        """
        Initialize the Remedy client with server and authentication details.
        Args:
            server_url: The base URL of the Remedy server
            username: Username for authentication
            password: Password for authentication
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.token_type = "AR-JWT"
        self.last_login_time = None
        self.token_expiry = 60 * 60  # Default 1 hour (in seconds)
        self.session_active = False
        
        # Handle SSL verification
        if ssl_verify is False:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled.")
            self.ssl_verify = False
        else:
            self.ssl_verify = True
            
        logger.info(f"Initialized Remedy client for {self.server_url}")
        
        # Cached data storage
        self.incidents_cache = {}
        self.support_groups_cache = set()
        self.assignees_cache = set()
        self.last_cache_refresh = None
        
        # Ensure cache directory exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def fetch_and_cache_all_incidents(self, days_back=30, status=None):
        """
        Fetch and cache all incidents from the last X days.
        Args:
            days_back: Number of days to look back for incidents
            status: Optional status filter (e.g., "Open", "Closed")
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.ensure_logged_in():
            logger.error("Failed to log in. Cannot fetch incidents.")
            return False
            
        logger.info(f"Fetching all incidents from the past {days_back} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_datetime = start_date.strftime("%Y-%m-%d 00:00:00.000")
        end_datetime = end_date.strftime("%Y-%m-%d 23:59:59.999")
        
        # Create qualified query
        query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' <= \"{end_datetime}\""]
        
        # Add status filter if provided
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
            
        qualified_query = " AND ".join(query_parts)
        
        # Fields to retrieve (extensive list to have complete data)
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager", "Last Modified Date",
            "Last Modified By", "Resolution", "Notes"
        ]
        
        # Set a high limit to get as many incidents as possible
        limit = 10000
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            incidents = result["entries"]
            logger.info(f"Retrieved {len(incidents)} incidents from the past {days_back} days")
            
            # Process and store in cache
            self.incidents_cache = {}
            
            # Extract support groups and assignees
            self.support_groups_cache = set()
            self.assignees_cache = set()
            
            # Process each incident
            for incident in incidents:
                if "values" in incident and "Incident Number" in incident["values"]:
                    incident_id = incident["values"]["Incident Number"]
                    self.incidents_cache[incident_id] = incident
                    
                    # Extract support group
                    if "Support Group Name" in incident["values"] and incident["values"]["Support Group Name"]:
                        self.support_groups_cache.add(incident["values"]["Support Group Name"])
                        
                    # Extract assignee
                    if "Assignee" in incident["values"] and incident["values"]["Assignee"]:
                        self.assignees_cache.add(incident["values"]["Assignee"])
            
            # Save cache to disk
            self._save_cache_to_disk()
            
            # Update last refresh time
            self.last_cache_refresh = time.time()
            
            logger.info(f"Successfully cached {len(self.incidents_cache)} incidents")
            logger.info(f"Extracted {len(self.support_groups_cache)} support groups and {len(self.assignees_cache)} assignees")
            
            return True
        else:
            logger.error("Failed to retrieve incidents for caching")
            return False
            
    def _save_cache_to_disk(self):
        """Save the cached data to disk for persistence."""
        try:
            # Ensure cache directory exists
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
                
            # Save incidents cache
            with open(INCIDENTS_CACHE_FILE, 'wb') as f:
                pickle.dump({
                    'incidents': self.incidents_cache,
                    'support_groups': self.support_groups_cache,
                    'assignees': self.assignees_cache,
                    'timestamp': time.time()
                }, f)
                
            logger.info(f"Cache saved to {INCIDENTS_CACHE_FILE}")
            return True
        except Exception as e:
            logger.error(f"Error saving cache to disk: {str(e)}")
            return False
            
    def _load_cache_from_disk(self):
        """Load cached data from disk if available and not expired."""
        try:
            if os.path.exists(INCIDENTS_CACHE_FILE):
                with open(INCIDENTS_CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Check if cache is expired
                cache_timestamp = cache_data.get('timestamp', 0)
                if time.time() - cache_timestamp > CACHE_EXPIRY:
                    logger.info(f"Cache is expired (older than {CACHE_EXPIRY/3600} hours)")
                    return False
                    
                # Load cached data
                self.incidents_cache = cache_data.get('incidents', {})
                self.support_groups_cache = cache_data.get('support_groups', set())
                self.assignees_cache = cache_data.get('assignees', set())
                self.last_cache_refresh = cache_timestamp
                
                logger.info(f"Loaded {len(self.incidents_cache)} incidents from cache")
                return True
            else:
                logger.info("No cache file found")
                return False
        except Exception as e:
            logger.error(f"Error loading cache from disk: {str(e)}")
            return False
    
class NLPProcessor:
    """
    Processor for natural language processing tasks on incident data.
    Provides tokenization, lemmatization, and text preprocessing for better matching.
    """
    def __init__(self):
        """Initialize the NLP processor with required components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation_translator = str.maketrans('', '', string.punctuation)
        
        # Inverted index for fast searching
        self.incident_index = defaultdict(list)
        self.incidents_processed = False
        
    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing, removing stop words, and lemmatizing.
        Args:
            text: Text to preprocess
        Returns:
            list: Preprocessed tokens
        """
        if not text:
            return []
            
        # Convert to lowercase and remove punctuation
        text = text.lower().translate(self.punctuation_translator)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return tokens
        
    def build_index_from_incidents(self, incidents):
        """
        Build an inverted index from incident data for faster searching.
        Args:
            incidents: Dictionary of incidents {incident_id: incident_data}
        """
        logger.info("Building search index from incidents")
        self.incident_index = defaultdict(list)
        
        for incident_id, incident in incidents.items():
            if "values" in incident:
                values = incident["values"]
                
                # Process summary and description
                summary = values.get("Summary", "")
                description = values.get("Description", "")
                
                # Combine texts for processing
                combined_text = f"{summary} {description}"
                tokens = self.preprocess_text(combined_text)
                
                # Add to index (token -> incident_id)
                for token in tokens:
                    if token not in self.incident_index[token]:
                        self.incident_index[token].append(incident_id)
        
        self.incidents_processed = True
        logger.info(f"Built search index with {len(self.incident_index)} unique tokens")
        
    def search(self, query, incidents, max_results=20):
        """
        Search for incidents matching the query using the inverted index.
        Args:
            query: Search query
            incidents: Dictionary of incidents to search in
            max_results: Maximum number of results to return
        Returns:
            list: Matching incidents sorted by relevance
        """
        # Ensure index is built
        if not self.incidents_processed:
            self.build_index_from_incidents(incidents)
            
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        # Find matching incidents
        matching_incident_ids = []
        scores = {}
        
        for token in query_tokens:
            if token in self.incident_index:
                for incident_id in self.incident_index[token]:
                    if incident_id not in scores:
                        scores[incident_id] = 0
                    scores[incident_id] += 1
        
        # Sort by score (descending)
        matching_incident_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:max_results]
        
        # Get full incidents
        results = []
        for incident_id in matching_incident_ids:
            if incident_id in incidents:
                results.append(incidents[incident_id])
                
        return results
        
    def logout(self):
        """
        Log out from Remedy and invalidate the token.
        Returns:
            bool: True on success, False on failure
        """
        if not self.token:
            logger.warning("Cannot logout: No active token")
            return False
            
        logger.info("Logging out and invalidating token")
        url = f"{self.server_url}/api/jwt/logout"
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        try:
            r = requests.post(url, headers=headers, verify=self.ssl_verify)
            if r.status_code == 204 or r.status_code == 200:
                logger.info("Logout successful")
                self.token = None
                self.session_active = False
                return True
            else:
                logger.error(f"Logout failed with status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False

    def get_incident(self, incident_id):
        """
        Get a specific incident by its ID.
        Args:
            incident_id: The Incident Number (e.g., INC000001482087)
        Returns:
            dict: Incident data or None if not found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return None
            
        logger.info(f"Fetching incident: {incident_id}")
        
        # Create qualified query
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager", "Last Modified Date",
            "Last Modified By", "Resolution", "Notes"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            return result["entries"][0]
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None
            
    def get_incidents_by_date(self, date_str, status=None, owner_group=None):
        """
        Get all incidents submitted on a specific date.
        Support flexible date formats: today, yesterday, last week, specific date
        
        Args:
            date_str: The submission date in various formats (YYYY-MM-DD, today, yesterday, etc.)
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
            
        # Parse the date string to get a datetime object
        date_obj = self._parse_date_expression(date_str)
        if not date_obj:
            logger.error(f"Invalid date format or expression: {date_str}")
            return []
            
        logger.info(f"Fetching incidents for date: {date_obj.strftime('%Y-%m-%d')}")
        
        # Create date range (entire day)
        start_datetime = date_obj.strftime("%Y-%m-%d 00:00:00.000")
        end_datetime = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.000")
        
        # Create qualified query
        query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""]
        
        # Add status filter if provided
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
            
        # Add owner group filter if provided
        if owner_group:
            query_parts.append(f"'Owner Group'=\"{owner_group}\"")
            
        qualified_query = " AND ".join(query_parts)
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents for date {date_obj.strftime('%Y-%m-%d')}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found for date {date_obj.strftime('%Y-%m-%d')} or error occurred")
            return []
            
    def _parse_date_expression(self, date_expr):
        """
        Parse a date expression into a datetime object.
        Supports: 'today', 'yesterday', 'last week', specific dates in YYYY-MM-DD format.
        
        Args:
            date_expr: Date expression to parse
            
        Returns:
            datetime object or None if parsing failed
        """
        today = datetime.now()
        
        # Check for keywords
        if date_expr.lower() == 'today':
            return today
        elif date_expr.lower() == 'yesterday':
            return today - timedelta(days=1)
        elif date_expr.lower() == 'last week':
            return today - timedelta(days=7)
        elif date_expr.lower() == 'last month':
            # Approximate a month as 30 days
            return today - timedelta(days=30)
        elif date_expr.lower() == 'this week':
            # Return the beginning of the current week (Monday)
            return today - timedelta(days=today.weekday())
        
        # Try to parse as YYYY-MM-DD
        try:
            return datetime.strptime(date_expr, "%Y-%m-%d")
        except ValueError:
            # Try other common formats
            formats = [
                "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
                "%b %d %Y", "%d %b %Y", "%B %d %Y", "%d %B %Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_expr, fmt)
                except ValueError:
                    continue
                    
        return None
            
    def get_incidents_by_status(self, status, limit=100):
        """
        Get incidents by their status.
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching incidents with status: {status}")
        
        # Create qualified query
        qualified_query = f"'Status'=\"{status}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with status {status}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with status {status} or error occurred")
            return []
            
    def get_incidents_by_assignee(self, assignee, limit=100):
        """
        Get incidents assigned to a specific person.
        Args:
            assignee: The assignee name
            limit: Maximum number of incidents to retrieve
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching incidents assigned to: {assignee}")
        
        # Create qualified query
        qualified_query = f"'Assignee'=\"{assignee}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents assigned to {assignee}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found assigned to {assignee} or error occurred")
            return []
            
    def search_incidents(self, search_text, limit=50):
        """
        Search for incidents containing specific text in summary or description.
        
        Args:
            search_text: Text to search for in incident summary or description
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Searching for incidents with text: {search_text}")
        
        # Create qualified query with OR condition
        qualified_query = f"'Summary' LIKE \"%{search_text}%\" OR 'Description' LIKE \"%{search_text}%\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Found {len(result['entries'])} incidents containing '{search_text}'")
            return result["entries"]
        else:
            logger.warning(f"No incidents found containing '{search_text}' or error occurred")
            return []
            
    def get_incidents_by_support_group(self, support_group, limit=100):
        """
        Get incidents assigned to a specific support group.
        
        Args:
            support_group: The support group name
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching incidents for support group: {support_group}")
        
        # Create qualified query
        qualified_query = f"'Support Group Name'=\"{support_group}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents for support group {support_group}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found for support group {support_group} or error occurred")
            return []
            
    def get_recent_incidents(self, days=7, limit=100):
        """
        Get incidents from the past X days.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching incidents from the past {days} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_datetime = start_date.strftime("%Y-%m-%d 00:00:00.000")
        end_datetime = end_date.strftime("%Y-%m-%d 23:59:59.999")
        
        # Create qualified query
        qualified_query = f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' <= \"{end_datetime}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents from the past {days} days")
            return result["entries"]
        else:
            logger.warning(f"No incidents found from the past {days} days or error occurred")
            return []
            
    def query_form(self, form_name, qualified_query=None, fields=None, limit=100):
        """
        Query a Remedy form with optional filters and field selection.
        Args:
            form_name: The name of the form to query (e.g., "HPD:Help Desk")
            qualified_query: Optional qualified query string for filtering
            fields: Optional list of fields to retrieve
            limit: Maximum number of records to retrieve
        Returns:
            dict: Query result or None if error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return None
            
        logger.info(f"Querying form: {form_name}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/{form_name}"
        
        # Build headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Build query parameters
        params = {}
        if qualified_query:
            params["q"] = qualified_query
        if fields:
            params["fields"] = ",".join(fields)
        if limit:
            params["limit"] = limit
            
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully queried form {form_name} and got {len(result.get('entries', []))} results")
                return result
            else:
                logger.error(f"Query failed with status code: {r.status_code}")
                logger.error(f"Headers: {r.headers}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return None
            
    def create_incident(self, summary, description, impact="4-Minor/Localized",
                        urgency="4-Low", reported_source="Direct Input", 
                        service_type="User Service Restoration", assigned_group=None):
        """
        Create a new incident in Remedy.
        Args:
            summary: Incident summary/title
            description: Detailed description
            impact: Impact level (1-5)
            urgency: Urgency level (1-4)
            reported_source: How the incident was reported
            service_type: Type of service
            assigned_group: Group to assign the incident to
        Returns:
            dict: Created incident data or None if error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return None
            
        logger.info(f"Creating new incident: {summary}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk"
        
        # Build headers
        headers = {
            "Authorization": f"{self.token_type} {self.token}",
            "Content-Type": "application/json"
        }
        
        # Build incident data
        incident_data = {
            "values": {
                "Summary": summary,
                "Description": description,
                "Impact": impact,
                "Urgency": urgency,
                "Reported Source": reported_source,
                "Service Type": service_type
            }
        }
        
        if assigned_group:
            incident_data["values"]["Assigned Group"] = assigned_group
            
        # Make the request
        try:
            r = requests.post(url, headers=headers, json=incident_data, verify=self.ssl_verify)
            if r.status_code == 201:
                result = r.json()
                logger.info(f"Successfully created incident: {result.get('values', {}).get('Incident Number')}")
                return result
            else:
                logger.error(f"Create incident failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Create incident error: {str(e)}")
            return None

    def update_incident(self, incident_id, update_data):
        """
        Update an existing incident.
        Args:
            incident_id: The Incident Number to update
            update_data: Dictionary of fields to update
        Returns:
            bool: True on success, False on failure
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return False
            
        logger.info(f"Updating incident: {incident_id}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk"
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        url = f"{url}?q={quote(qualified_query)}"
        
        # Build headers
        headers = {
            "Authorization": f"{self.token_type} {self.token}",
            "Content-Type": "application/json"
        }
        
        # Build update data
        payload = {
            "values": update_data
        }
        
        # Make the request
        try:
            r = requests.put(url, headers=headers, json=payload, verify=self.ssl_verify)
            if r.status_code == 204:
                logger.info(f"Successfully updated incident: {incident_id}")
                return True
            else:
                logger.error(f"Update incident failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Update incident error: {str(e)}")
            return False
            
    def get_incident_history(self, incident_id):
        """
        Get the history of changes for a specific incident.
        Args:
            incident_id: The Incident Number
        Returns:
            list: History entries or empty list if none found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching history for incident: {incident_id}")
        
        # Build URL for history form
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk History"
        
        # Qualified query to filter by incident number
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Query parameters
        params = {
            "q": qualified_query,
            "fields": "History Date Time,Action,Description,Status,Changed By,Assigned Group"
        }
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully retrieved history for incident {incident_id} with {len(result.get('entries', []))} entries")
                return result.get("entries", [])
            else:
                logger.error(f"Get history failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return []
        except Exception as e:
            logger.error(f"Get history error: {str(e)}")
            return []
            
    def get_incident_attachments(self, incident_id):
        """
        Get attachments for a specific incident.
        Args:
            incident_id: The Incident Number
        Returns:
            list: Attachment metadata or empty list if none found/error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
            
        logger.info(f"Fetching attachments for incident: {incident_id}")
        
        # Build URL for attachments
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Attachment"
        
        # Qualified query to filter by incident number
        qualified_query = f"'Request ID'=\"{incident_id}\""
        
        # Headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Query parameters
        params = {
            "q": qualified_query,
            "fields": "Incident Number,Request ID,Document,Document Type,Document Size,Modified Date,Created By"
        }
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully retrieved attachments for incident {incident_id} with {len(result.get('entries', []))} attachments")
                return result.get("entries", [])
            else:
                logger.error(f"Get attachments failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return []
        except Exception as e:
            logger.error(f"Get attachments error: {str(e)}")
            return []
            
    def download_attachment(self, incident_id, attachment_id):
        """
        Download a specific attachment.
        Args:
            incident_id: The Incident Number
            attachment_id: The attachment ID
        Returns:
            tuple: (content, filename) or (None, None) if error
        """
        if not self.ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return None, None
            
        logger.info(f"Downloading attachment {attachment_id} for incident: {incident_id}")
        
        # Build URL for attachment download
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Attachment/{attachment_id}/content"
        
        # Headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, verify=self.ssl_verify)
            if r.status_code == 200:
                # Get filename from Content-Disposition header
                content_disp = r.headers.get('Content-Disposition', '')
                filename = re.findall('filename="(.+)"', content_disp)
                if filename:
                    filename = filename[0]
                else:
                    filename = f"attachment_{attachment_id}"
                    
                logger.info(f"Successfully downloaded attachment: {filename}")
                return r.content, filename
            else:
                logger.error(f"Download attachment failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return None, None
        except Exception as e:
            logger.error(f"Download attachment error: {str(e)}")
            return None, None
            
    def process_incident_for_rag(self, incident):
        """
        Process an incident into a format suitable for RAG indexing.
        Args:
            incident: Raw incident data from Remedy API
        Returns:
            dict: Processed incident with metadata and content
        """
        if not incident or "values" not in incident:
            return None
            
        values = incident.get("values", {})
        
        # Extract metadata
        metadata = {
            "incident_number": values.get("Incident Number"),
            "status": values.get("Status"),
            "priority": values.get("Priority"),
            "impact": values.get("Impact"),
            "assignee": values.get("Assignee"),
            "owner": values.get("Owner"),
            "owner_group": values.get("Owner Group"),
            "assigned_group": values.get("Assigned Group"),
            "submitter": values.get("Submitter"),
            "submit_date": values.get("Submit Date"),
            "summary": values.get("Summary")
        }
        
        # Build content for embedding
        content_parts = []
        if values.get("Summary"):
            content_parts.append(f"Summary: {values.get('Summary')}")
        if values.get("Description"):
            content_parts.append(f"Description: {values.get('Description')}")
        if values.get("Status"):
            content_parts.append(f"Status: {values.get('Status')}")
        if values.get("Priority"):
            content_parts.append(f"Priority: {values.get('Priority')}")
        if values.get("Impact"):
            content_parts.append(f"Impact: {values.get('Impact')}")
        if values.get("Assignee"):
            content_parts.append(f"Assigned to: {values.get('Assignee')}")
        if values.get("Owner Group"):
            content_parts.append(f"Owner Group: {values.get('Owner Group')}")
        
        # Combine content parts into a single text
        content = "\n".join(content_parts)
        
        return {
            "metadata": metadata,
            "content": content,
            "raw_data": values
        }

    def format_incidents_as_table(self, incidents):
        """
        Format a list of incidents as a table for display.
        
        Args:
            incidents: List of incident data
            
        Returns:
            str: Formatted table as string
        """
        if not incidents:
            return "No incidents found."
            
        data = []
        for inc in incidents:
            values = inc.get("values", {})
            data.append({
                "Incident Number": values.get("Incident Number", ""),
                "Summary": values.get("Summary", "")[:50] + ("..." if len(values.get("Summary", "")) > 50 else ""),
                "Status": values.get("Status", ""),
                "Priority": values.get("Priority", ""),
                "Assignee": values.get("Assignee", ""),
                "Submit Date": values.get("Submit Date", "")
            })
            
        df = pd.DataFrame(data)
        
        # Return tabulated string
        return tabulate(df, headers="keys", tablefmt="grid", showindex=False)
        
    def extract_table_from_description(self, description):
        """
        Extracts tables from incident descriptions.
        Supports simple text-based tables with | delimiters.
        
        Args:
            description: Text that might contain tables
            
        Returns:
            list: Extracted tables (if any)
        """
        if not description:
            return []
            
        tables = []
        table_lines = []
        in_table = False
        
        for line in description.split('\n'):
            # Check if line looks like a table row (contains | character)
            if '|' in line and line.count('|') > 1:
                if not in_table:
                    in_table = True
                table_lines.append(line)
            elif in_table:
                # End of table
                if table_lines:
                    tables.append('\n'.join(table_lines))
                    table_lines = []
                in_table = False
                
        # Add last table if there is one
        if table_lines:
            tables.append('\n'.join(table_lines))
            
        return tables
        
class GeminiAI:
    """Class for interacting with Gemini models via Vertex AI."""
    
    def __init__(self):
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.text_model = GenerativeModel(MODEL_NAME)
        self.vision_model = GenerativeModel(MULTIMODAL_MODEL)  # For handling images
        logger.info(f"Initialized Gemini AI with text model: {MODEL_NAME} and vision model: {MULTIMODAL_MODEL}")
        
    def generate_response(self, prompt, temperature=0.2, max_tokens=8192):
        """
        Generate a response from Gemini.
        Args:
            prompt: The prompt to send to Gemini
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum length of generated text
        Returns:
            str: The generated response
        """
        logger.info(f"Generating response for prompt (first 100 chars): {prompt[:100]}...")
        
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
            )
            
            # Generate response
            response = self.text_model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            if response.candidates and response.candidates[0].text:
                logger.info(f"Generated response of length: {len(response.candidates[0].text)}")
                return response.candidates[0].text
            else:
                logger.warning("No response generated")
                return "I couldn't generate a response. Please try a different query."
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
            
    def analyze_image(self, image_data, prompt, temperature=0.2):
        """
        Analyze an image with Gemini multimodal model.
        Args:
            image_data: Binary image data
            prompt: Text prompt describing what to analyze in the image
            temperature: Controls randomness (0.0-1.0)
        Returns:
            str: The analysis result
        """
        logger.info(f"Analyzing image with prompt: {prompt}")
        
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=2048,
            )
            
            # Create content parts with image and text
            from vertexai.generative_models import Part
            
            # Convert binary image data to appropriate format
            parts = [
                Part.from_data(image_data, mime_type="image/jpeg"),
                Part.from_text(prompt),
            ]
            
            # Generate response with multimodal model
            response = self.vision_model.generate_content(
                parts,
                generation_config=generation_config,
            )
            
            if response.candidates and response.candidates[0].text:
                logger.info(f"Generated image analysis of length: {len(response.candidates[0].text)}")
                return response.candidates[0].text
            else:
                logger.warning("No analysis generated for the image")
                return "I couldn't analyze the image. Please try a different image or prompt."
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"
            
class RemedyChatbot:
    """
    Chatbot that integrates BMC Remedy with Gemini for natural language interactions.
    """
    
    def __init__(self, server_url=DEFAULT_SERVER_URL, username=DEFAULT_USERNAME, password=DEFAULT_PASSWORD):
        """
        Initialize the chatbot with Remedy client and Gemini AI.
        Args:
            server_url: The Remedy server URL
            username: Username for Remedy
            password: Password for Remedy
        """
        self.remedy_client = RemedyClient(server_url, username, password, ssl_verify=False)
        self.gemini_ai = GeminiAI()
        self.nlp_processor = NLPProcessor()
        
        # Conversation history for context
        self.conversation_history = []
        self.max_history_length = 10
        
        # Initialize by logging in and loading cache
        logger.info("Initializing Remedy chatbot")
        self._initialize()
        
    def _initialize(self):
        """Initialize the chatbot by logging in and loading or building cache."""
        # First, try to log in
        status, _ = self.remedy_client.login()
        if status != 1:
            logger.error("Failed to initialize Remedy client - login failed")
            return False
            
        # Try to load cache from disk
        cache_loaded = self.remedy_client._load_cache_from_disk()
        
        # If cache not loaded or expired, fetch fresh data
        if not cache_loaded:
            logger.info("Cache not available or expired, fetching fresh data")
            self.remedy_client.fetch_and_cache_all_incidents(days_back=30)
        
        # Build NLP index from the cached incidents
        self.nlp_processor.build_index_from_incidents(self.remedy_client.incidents_cache)
        
        logger.info("Successfully initialized Remedy chatbot")
        return True
            
    def _get_system_prompt(self):
        """
        Get the system prompt that defines the chatbot's capabilities and behavior.
        Returns:
            str: The system prompt
        """
        return """
        You are RemedyBot, an advanced AI assistant designed to help users with BMC Remedy incident management. You have access to Remedy incident data through a specialized client that can retrieve various types of incident information.

        Your capabilities include:
        1. Finding specific incidents by ID (e.g., INC000001234567)
        2. Retrieving incidents submitted on a specific date (including today, yesterday, etc.)
        3. Finding incidents with a particular status (e.g., Open, Closed, Resolved)
        4. Retrieving incidents assigned to a specific person
        5. Searching for incidents containing specific text in their summary or description
        6. Finding incidents assigned to a specific support group
        7. Getting recent incidents from the past X days
        8. Analyzing incident history and attachments
        9. Understanding tables and structured data within incidents
        10. Extracting key information and insights from incident data

        You always provide helpful, accurate, and professional responses. Be concise but complete in your answers. When presenting multiple incidents, organize them in a clear table format when appropriate. Use bullet points for lists when needed.

        If you don't have the information needed to answer a question, explain what information you would need and how the user could provide it. If a question is ambiguous, ask for clarification to ensure you provide the most relevant information.

        Always aim to understand the user's intent. If they seem to be looking for troubleshooting help, provide guidance based on similar incidents if available. If they're asking about incident management processes, provide best practices and recommendations.

        Remember to cite incident numbers and other reference information when providing answers to make it easy for users to verify the information.
        """
        
    def process_query(self, query):
        """
        Process a natural language query about Remedy incidents using the cached data.
        Args:
            query: The user's query
        Returns:
            str: The response
        """
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
            
        # Check for special commands
        if query.lower() == "help":
            response = self._get_help_message()
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
            
        if query.lower() == "refresh cache":
            success = self.remedy_client.fetch_and_cache_all_incidents(days_back=30)
            if success:
                self.nlp_processor.build_index_from_incidents(self.remedy_client.incidents_cache)
                response = "Cache refreshed successfully! I now have the latest incident data."
            else:
                response = "Failed to refresh cache. Please check the logs for details."
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # First, check for direct incident ID references
        incident_id_pattern = r"(?:^|\s)(INC\d{12})(?:$|\s)"
        incident_id_match = re.search(incident_id_pattern, query)
        
        if incident_id_match:
            # Direct lookup by incident ID from cache
            incident_id = incident_id_match.group(1)
            logger.info(f"Direct lookup for incident ID: {incident_id}")
            incident = self.remedy_client.get_cached_incident(incident_id)
            
            if not incident:
                # If not in cache, try to get it directly (it might be too old or too new)
                incident = self.remedy_client.get_incident(incident_id)
                
            if incident:
                # Use Gemini to format the response
                prompt = f"""
                Here is the data for incident {incident_id}:
                {json.dumps(incident, indent=2)}
                
                The user asked: "{query}"
                
                Please provide a concise, well-formatted response that highlights the most important information from this incident. Include the incident status, priority, summary, and any other relevant details. If there are tables in the description, describe them clearly.
                """
                response = self.gemini_ai.generate_response(prompt)
            else:
                response = f"I couldn't find incident {incident_id} in the system. It may not exist, be too old to be in our cache, or you might not have permission to view it."
                
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
            
        # Determine the query type and parameters
        query_type, parameters = self._analyze_query(query)
        
        # Retrieve the requested information based on the query type
        retrieved_data = self._retrieve_cached_data(query_type, parameters)
        
        # Generate the response
        system_prompt = self._get_system_prompt()
        history_text = self._format_conversation_history()
        
        if retrieved_data:
            prompt = f"""
            {system_prompt}
            
            Conversation history:
            {history_text}
            
            Current query: {query}
            
            Here is the information retrieved from Remedy:
            {json.dumps(retrieved_data, indent=2, default=str)}
            
            Based on this information, please provide a helpful and concise response to the user's query. Format the response appropriately using markdown for readability. If there are multiple incidents, consider using a table format when appropriate. If you're showing a list of incidents, include the incident number, summary, status, and priority. If the response is about a specific incident, organize the most important information clearly.
            
            Add a link or reference to any relevant incidents by their ID so the user can look them up directly if needed.
            """
        else:
            # Try a free text search using NLP
            logger.info(f"No structured data found for query, trying NLP search: {query}")
            
            search_results = self.nlp_processor.search(query, self.remedy_client.incidents_cache)
            
            if search_results:
                prompt = f"""
                {system_prompt}
                
                Conversation history:
                {history_text}
                
                Current query: {query}
                
                I performed a semantic search for your query and found these potentially relevant incidents:
                {json.dumps(search_results, indent=2, default=str)}
                
                Please provide a helpful response summarizing these results in a clear, well-organized way. Format the response appropriately using markdown for readability. If there are multiple incidents, consider using a table format when appropriate. Make sure to include incident numbers, summaries, statuses, and priorities.
                """
            else:
                # No data retrieved, give a response explaining why
                prompt = f"""
                {system_prompt}
                
                Conversation history:
                {history_text}
                
                Current query: {query}
                
                I was unable to find any relevant information in our cached incident data. Please provide a helpful response explaining why this might be the case (such as no matching incidents, ambiguous query, etc.) and suggesting alternatives or asking for clarification. Be friendly and helpful.
                """
                
        # Generate the response
        response = self.gemini_ai.generate_response(prompt)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
        
        # Form a prompt for Gemini to generate the final response
    def _analyze_query(self, query):
        """
        Analyze a query to determine its type and extract parameters.
        Uses pattern matching and heuristics instead of LLM to avoid JSON parsing issues.
        
        Args:
            query: User query
            
        Returns:
            tuple: (query_type, parameters)
        """
        query_lower = query.lower()
        
        # Check for incident ID
        incident_id_match = re.search(r"(INC\d{12})", query)
        if incident_id_match:
            return "incident_id", {"incident_id": incident_id_match.group(1)}
            
        # Check for date-based queries
        if any(term in query_lower for term in ["today", "yesterday", "last week", "this week", "last month"]):
            # Extract date
            if "today" in query_lower:
                date_str = "today"
            elif "yesterday" in query_lower:
                date_str = "yesterday"
            elif "last week" in query_lower:
                date_str = "last week"
            elif "this week" in query_lower:
                date_str = "this week"
            elif "last month" in query_lower:
                date_str = "last month"
            else:
                date_str = "today"
                
            # Extract status if present
            status = None
            status_terms = ["open", "closed", "resolved", "in progress", "pending"]
            for term in status_terms:
                if term in query_lower:
                    status = term.capitalize()
                    break
                    
            return "date", {"date": date_str, "status": status}
            
        # Check for status-based queries
        status_terms = ["open", "closed", "resolved", "in progress", "pending"]
        for status in status_terms:
            if status in query_lower and ("incidents" in query_lower or "tickets" in query_lower):
                return "status", {"status": status.capitalize()}
                
        # Check for assignee-based queries
        if "assigned to" in query_lower or "working on" in query_lower:
            # Extract assignee
            assignee_match = re.search(r"(assigned to|working on by) (.*?)([\.\?]|$| and | or )", query_lower)
            if assignee_match:
                assignee = assignee_match.group(2).strip()
                return "assignee", {"assignee": assignee}
                
        # Check for support group queries
        if "support group" in query_lower or ("team" in query_lower and "incidents" in query_lower):
            # Extract support group
            group_match = re.search(r"(support group|team) (.*?)([\.\?]|$| and | or )", query_lower)
            if group_match:
                support_group = group_match.group(2).strip()
                return "support_group", {"support_group": support_group}
                
        # Check for recent incidents queries
        if "recent" in query_lower and ("incidents" in query_lower or "tickets" in query_lower):
            days = 7  # Default
            days_match = re.search(r"(last|past) (\d+) (days|weeks)", query_lower)
            if days_match:
                try:
                    days = int(days_match.group(2))
                    if days_match.group(3) == "weeks":
                        days = days * 7
                except ValueError:
                    pass
            return "recent", {"days": days}
            
        # Default to search
        # Clean query for better search results
        search_text = query
        common_words = ["show", "me", "list", "get", "find", "all", "incidents", "tickets"]
        for word in common_words:
            search_text = re.sub(r'\b' + word + r'\b', '', search_text, flags=re.IGNORECASE)
        search_text = search_text.strip()
        
        return "search", {"search_text": search_text}
        
    def _retrieve_cached_data(self, query_type, parameters):
        """
        Retrieve data from cache based on query type and parameters.
        
        Args:
            query_type: Type of query
            parameters: Query parameters
            
        Returns:
            dict: Retrieved data or None if retrieval failed
        """
        try:
            if query_type == "incident_id":
                incident_id = parameters.get("incident_id")
                if incident_id:
                    incident = self.remedy_client.get_cached_incident(incident_id)
                    return {
                        "type": "incident",
                        "data": incident
                    }
                    
            elif query_type == "date":
                date_str = parameters.get("date")
                status = parameters.get("status")
                if date_str:
                    # Parse date string
                    date_obj = self._parse_date_expression(date_str)
                    if date_obj:
                        incidents = self.remedy_client.get_cached_incidents_by_date(date_obj, status)
                        return {
                            "type": "incidents_by_date",
                            "data": incidents,
                            "parameters": parameters
                        }
                        
            elif query_type == "status":
                status = parameters.get("status")
                if status:
                    incidents = self.remedy_client.get_cached_incidents_by_status(status)
                    return {
                        "type": "incidents_by_status",
                        "data": incidents,
                        "parameters": parameters
                    }
                    
            elif query_type == "assignee":
                assignee = parameters.get("assignee")
                if assignee:
                    incidents = self.remedy_client.get_cached_incidents_by_assignee(assignee)
                    return {
                        "type": "incidents_by_assignee",
                        "data": incidents,
                        "parameters": parameters
                    }
                    
            elif query_type == "support_group":
                support_group = parameters.get("support_group")
                if support_group:
                    incidents = self.remedy_client.get_cached_incidents_by_support_group(support_group)
                    return {
                        "type": "incidents_by_support_group",
                        "data": incidents,
                        "parameters": parameters
                    }
                    
            elif query_type == "recent":
                days = parameters.get("days", 7)
                # Recent incidents are already in cache
                # Just filter by date
                today = datetime.now()
                start_date = today - timedelta(days=days)
                
                # Filter incidents by date
                incidents = []
                for incident in self.remedy_client.incidents_cache.values():
                    if "values" in incident and "Submit Date" in incident["values"]:
                        submit_date_str = incident["values"]["Submit Date"]
                        try:
                            # Parse date from string (format may vary)
                            submit_date = datetime.strptime(submit_date_str[:10], "%Y-%m-%d")
                            if submit_date >= start_date:
                                incidents.append(incident)
                        except ValueError:
                            # Skip incidents with unparseable dates
                            pass
                            
                return {
                    "type": "recent_incidents",
                    "data": incidents,
                    "parameters": parameters
                }
                
            elif query_type == "search":
                search_text = parameters.get("search_text")
                if search_text:
                    # Use NLP processor for semantic search
                    incidents = self.nlp_processor.search(search_text, self.remedy_client.incidents_cache)
                    return {
                        "type": "search_results",
                        "data": incidents,
                        "parameters": parameters
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached data: {str(e)}")
            return None
            
    def _parse_date_expression(self, date_expr):
        """
        Parse a date expression into a datetime object.
        
        Args:
            date_expr: Date expression
            
        Returns:
            datetime object or None
        """
        today = datetime.now()
        
        # Check for keywords
        if date_expr.lower() == 'today':
            return today
        elif date_expr.lower() == 'yesterday':
            return today - timedelta(days=1)
        elif date_expr.lower() == 'last week':
            return today - timedelta(days=7)
        elif date_expr.lower() == 'this week':
            # Return the beginning of the current week (Monday)
            return today - timedelta(days=today.weekday())
        elif date_expr.lower() == 'last month':
            # Approximate a month as 30 days
            return today - timedelta(days=30)
            
        # Try to parse as YYYY-MM-DD
        try:
            return datetime.strptime(date_expr, "%Y-%m-%d")
        except ValueError:
            # Try other common formats
            formats = [
                "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
                "%b %d %Y", "%d %b %Y", "%B %d %Y", "%d %B %Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_expr, fmt)
                except ValueError:
                    continue
                    
        return None
        
    def _format_conversation_history(self):
        """
        Format the conversation history for inclusion in prompts.
        Returns:
            str: Formatted conversation history
        """
        formatted = []
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n\n".join(formatted)
        
    def _retrieve_data(self, analysis):
        """
        Retrieve data from Remedy based on the query analysis.
        Args:
            analysis: The query analysis
        Returns:
            dict: Retrieved data or None if retrieval failed
        """
        query_type = analysis.get("query_type", "unknown")
        parameters = analysis.get("parameters", {})
        
        try:
            if query_type == "incident_id":
                incident_id = parameters.get("incident_id")
                if incident_id:
                    return {
                        "type": "incident",
                        "data": self.remedy_client.get_incident(incident_id)
                    }
                    
            elif query_type == "date":
                date_str = parameters.get("date")
                status = parameters.get("status")
                owner_group = parameters.get("owner_group")
                if date_str:
                    return {
                        "type": "incidents_by_date",
                        "data": self.remedy_client.get_incidents_by_date(date_str, status, owner_group),
                        "parameters": {"date": date_str, "status": status, "owner_group": owner_group}
                    }
                    
            elif query_type == "status":
                status = parameters.get("status")
                limit = parameters.get("limit", 100)
                if status:
                    return {
                        "type": "incidents_by_status",
                        "data": self.remedy_client.get_incidents_by_status(status, limit),
                        "parameters": {"status": status, "limit": limit}
                    }
                    
            elif query_type == "assignee":
                assignee = parameters.get("assignee")
                limit = parameters.get("limit", 100)
                if assignee:
                    return {
                        "type": "incidents_by_assignee",
                        "data": self.remedy_client.get_incidents_by_assignee(assignee, limit),
                        "parameters": {"assignee": assignee, "limit": limit}
                    }
                    
            elif query_type == "search":
                search_text = parameters.get("search_text")
                limit = parameters.get("limit", 50)
                if search_text:
                    return {
                        "type": "search_results",
                        "data": self.remedy_client.search_incidents(search_text, limit),
                        "parameters": {"search_text": search_text, "limit": limit}
                    }
                    
            elif query_type == "support_group":
                support_group = parameters.get("support_group")
                limit = parameters.get("limit", 100)
                if support_group:
                    return {
                        "type": "incidents_by_support_group",
                        "data": self.remedy_client.get_incidents_by_support_group(support_group, limit),
                        "parameters": {"support_group": support_group, "limit": limit}
                    }
                    
            elif query_type == "recent":
                days = parameters.get("days", 7)
                limit = parameters.get("limit", 100)
                return {
                    "type": "recent_incidents",
                    "data": self.remedy_client.get_recent_incidents(days, limit),
                    "parameters": {"days": days, "limit": limit}
                }
                
            elif query_type == "history":
                incident_id = parameters.get("incident_id")
                if incident_id:
                    # Get both the incident and its history
                    incident = self.remedy_client.get_incident(incident_id)
                    history = self.remedy_client.get_incident_history(incident_id)
                    
                    # Also check for attachments
                    attachments = self.remedy_client.get_incident_attachments(incident_id)
                    
                    return {
                        "type": "incident_history",
                        "data": {
                            "incident": incident,
                            "history": history,
                            "attachments": attachments
                        },
                        "parameters": {"incident_id": incident_id}
                    }
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            return None
            
        # Default case if no data was retrieved
        logger.warning(f"No data retrieved for query type: {query_type} with parameters: {parameters}")
        return None
        
    def _get_help_message(self):
        """
        Get help message describing the chatbot's capabilities.
        Returns:
            str: Help message
        """
        return """
        # RemedyBot Help
        
        I can help you with the following types of requests:
        
        ## Finding Specific Incidents
        - Find incident INC000001234567
        - Tell me about incident INC000001234567
        - What's the status of incident INC000001234567?
        
        ## Incidents by Date
        - Show me incidents from today
        - What incidents were submitted yesterday?
        - List incidents from 2023-05-15
        - Show closed incidents from last week
        
        ## Incidents by Status
        - Show me all open incidents
        - List closed incidents
        - How many incidents are currently in progress?
        
        ## Incidents by Assignee
        - Show incidents assigned to John Smith
        - What is Jane Doe working on?
        - How many tickets are assigned to me?
        
        ## Searching for Incidents
        - Find incidents about network outage
        - Search for incidents related to email problems
        - Look for incidents mentioning database errors
        
        ## Support Group Incidents
        - Show incidents for the Network Support team
        - What's the workload of the Database team?
        - List incidents assigned to Infrastructure Support
        
        ## Recent Incidents
        - Show me recent incidents
        - What incidents were created in the past 3 days?
        - List new incidents from this week
        
        ## Incident History and Analysis
        - Show history of incident INC000001234567
        - Who modified incident INC000001234567?
        - What attachments are available for incident INC000001234567?
        
        Feel free to ask me in your own words, and I'll try to understand what you need!
        """
        
    def process_image_query(self, image_data, query):
        """
        Process a query about an image related to a Remedy incident.
        Args:
            image_data: Binary image data
            query: The user's query about the image
        Returns:
            str: The response
        """
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": f"[IMAGE QUERY] {query}"})
        
        # Generate a prompt for image analysis
        prompt = f"""
        This image is from a BMC Remedy incident management system.
        
        {query}
        
        Please analyze the image and provide a detailed response. If the image contains text or tables,
        extract and summarize the most important information. If it's a screenshot of an error or system state,
        interpret what's happening and what it might mean in the context of IT service management.
        """
        
        # Use Gemini to analyze the image
        response = self.gemini_ai.analyze_image(image_data, prompt)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

def main():
    """Main function to run the chatbot interactively."""
    print("=" * 80)
    print("RemedyBot - BMC Remedy Chatbot powered by Google Gemini")
    print("=" * 80)
    
    # Initialize chatbot with the default credentials
    print("\nInitializing Remedy chatbot...")
    print("Connecting to Remedy server and loading cache...")
    chatbot = RemedyChatbot()
    
    # Check if cache is loaded
    if len(chatbot.remedy_client.incidents_cache) == 0:
        print("No cached incidents found. Fetching incidents from Remedy...")
        chatbot.remedy_client.fetch_and_cache_all_incidents(days_back=30)
        chatbot.nlp_processor.build_index_from_incidents(chatbot.remedy_client.incidents_cache)
    
    print(f"\nInitialization complete. Loaded {len(chatbot.remedy_client.incidents_cache)} incidents in cache.")
    print("Type 'help' for available commands, 'refresh cache' to update the cache, or 'exit' to quit.")
    print("You can now start chatting with RemedyBot!\n")
    
    while True:
        try:
            # Get user input
            query = input("\n> ")
            
            # Check for exit command
            if query.lower() in ('exit', 'quit', 'bye'):
                print("Thank you for using RemedyBot. Goodbye!")
                break
                
            # Process the query and display the response
            if query:
                print("\nProcessing your query...\n")
                response = chatbot.process_query(query)
                print(response)
                
        except KeyboardInterrupt:
            print("\nOperation interrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            logger.error(f"Error in main loop: {str(e)}")

if __name__ == "__main__":
    main()




















#!/usr/bin/env python3
"""
Advanced Jira-Gemini AI Chatbot
--------------------------------
This script integrates Jira with Google's Gemini AI to create a powerful chatbot
that can answer questions about Jira tickets, provide deep insights,
understand images and tables in documentation, and maintain context
through conversations.

Features:
- SSL verification disabled for enterprise environments
- Robust caching system to avoid repeated API calls
- Enhanced prompt engineering for high-quality responses
- Support for understanding images and tables in Jira tickets
- Provides source links in responses
- Multiple output formats (paragraphs, tables, bullet points)
- Conversational abilities with minimal prompting
"""

import logging
import os
import sys
import json
import re
import time
import datetime
import hashlib
import pickle
import base64
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import urllib3
from datetime import datetime, timedelta
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from urllib.parse import urlparse, urljoin
from PIL import Image
from io import BytesIO
import pandas as pd
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Disable SSL warnings for enterprise environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_gemini_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("JiraGeminiChatbot")

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
CACHE_TTL_DAYS = int(os.environ.get("CACHE_TTL_DAYS", "7"))  # Cache valid for 7 days by default
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://your-domain.atlassian.net")
JIRA_USERNAME = os.environ.get("JIRA_USERNAME", "your-username")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "your-api-token")

# Cache directory setup
CACHE_DIR = Path.home() / ".jira_gemini_cache"
CACHE_DIR.mkdir(exist_ok=True)


class JiraClient:
    """Enhanced Jira REST API client with caching support."""
    
    def __init__(self, base_url: str, username: str, api_token: str):
        """
        Initialize the Jira client.
        
        Args:
            base_url: Jira instance base URL
            username: Jira username (usually email)
            api_token: Jira API token
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.verify = False  # Disable SSL verification for enterprise environments
        self.cache_file = CACHE_DIR / "jira_data_cache.pickle"
        self.cache = self._load_cache()
        self.image_cache_dir = CACHE_DIR / "images"
        self.image_cache_dir.mkdir(exist_ok=True)
        
        # Define cache sections
        self.cache.setdefault("issues", {})
        self.cache.setdefault("projects", {})
        self.cache.setdefault("users", {})
        self.cache.setdefault("metadata", {"last_full_sync": None})

    def _load_cache(self) -> Dict:
        """Load the cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                # Check if cache structure is as expected
                if isinstance(cache, dict) and "metadata" in cache:
                    logger.info(f"Cache loaded, last sync: {cache['metadata'].get('last_full_sync')}")
                    return cache
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        # Return empty cache if loading failed or file doesn't exist
        return {"metadata": {"last_full_sync": None}}

    def _save_cache(self) -> None:
        """Save the cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid based on TTL."""
        last_sync = self.cache["metadata"].get("last_full_sync")
        if not last_sync:
            return False
        
        cutoff_date = datetime.now() - timedelta(days=CACHE_TTL_DAYS)
        return last_sync > cutoff_date
    
    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None, data: Dict = None) -> Dict:
        """
        Make a request to the Jira API with error handling.
        
        Args:
            endpoint: API endpoint (without base URL)
            method: HTTP method
            params: Query parameters
            data: Request data for POST/PUT
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                timeout=30
            )
            
            if response.status_code >= 400:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                try:
                    error_data = response.json()
                    if "errorMessages" in error_data:
                        error_msg = f"API Error: {', '.join(error_data['errorMessages'])}"
                except:
                    pass
                raise Exception(error_msg)
                
            try:
                return response.json()
            except:
                return {"status": "success", "statusCode": response.status_code}
                
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise Exception(f"Failed to connect to Jira API: {str(e)}")
    
    def get_all_projects(self, force_refresh: bool = False) -> List[Dict]:
        """Get all projects the user has access to."""
        if not force_refresh and "projects" in self.cache and self.cache["projects"]:
            logger.info("Using cached projects data")
            return list(self.cache["projects"].values())
        
        logger.info("Fetching all projects from Jira")
        results = self._make_request("/rest/api/3/project")
        
        # Update cache
        self.cache["projects"] = {project["key"]: project for project in results}
        self._save_cache()
        
        return results
    
    def search_issues(self, jql: str, max_results: int = 100, fields: List[str] = None, expand: List[str] = None) -> Dict:
        """
        Search for issues using JQL (Jira Query Language).
        
        Args:
            jql: JQL search string
            max_results: Maximum number of results to return
            fields: List of field names to include
            expand: List of fields to expand
            
        Returns:
            Search results containing issues
        """
        # Generate cache key based on the query
        cache_key = hashlib.md5(f"{jql}:{max_results}:{fields}:{expand}".encode()).hexdigest()
        
        if cache_key in self.cache.get("search_results", {}):
            cache_entry = self.cache["search_results"][cache_key]
            # Check if cached result is still valid
            if datetime.fromisoformat(cache_entry["timestamp"]) > datetime.now() - timedelta(days=1):
                logger.info(f"Using cached search results for JQL: {jql}")
                return cache_entry["data"]
        
        logger.info(f"Searching issues with JQL: {jql}")
        params = {
            "jql": jql,
            "maxResults": max_results
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        if expand:
            params["expand"] = ",".join(expand)
        
        result = self._make_request("/rest/api/3/search", params=params)
        
        # Cache the search results
        self.cache.setdefault("search_results", {})
        self.cache["search_results"][cache_key] = {
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
        self._save_cache()
        
        # Also cache individual issues for direct access
        if "issues" in result:
            for issue in result["issues"]:
                self.cache["issues"][issue["key"]] = {
                    "data": issue,
                    "timestamp": datetime.now().isoformat(),
                    "full_data": False  # Indicates this is not the complete issue data
                }
            self._save_cache()
        
        return result
    
    def get_issue(self, issue_key: str, force_refresh: bool = False) -> Dict:
        """
        Get detailed information about a specific issue.
        
        Args:
            issue_key: The issue key (e.g., PROJECT-123)
            force_refresh: Whether to force a refresh from the API
        
        Returns:
            Issue data
        """
        # Check if we have a cached version and it's recent
        if not force_refresh and issue_key in self.cache["issues"]:
            issue_cache = self.cache["issues"][issue_key]
            cache_date = datetime.fromisoformat(issue_cache["timestamp"])
            
            if issue_cache.get("full_data", False) and cache_date > datetime.now() - timedelta(days=1):
                logger.info(f"Using cached data for issue {issue_key}")
                return issue_cache["data"]
        
        logger.info(f"Fetching issue {issue_key} from Jira")
        params = {
            "expand": "renderedFields,names,schema,transitions,operations,editmeta,changelog,versions,attachments"
        }
        
        result = self._make_request(f"/rest/api/3/issue/{issue_key}", params=params)
        
        # Cache the full issue data
        self.cache["issues"][issue_key] = {
            "data": result,
            "timestamp": datetime.now().isoformat(),
            "full_data": True
        }
        self._save_cache()
        
        # Cache issue attachments
        if "fields" in result and "attachment" in result["fields"] and result["fields"]["attachment"]:
            self._cache_attachments(issue_key, result["fields"]["attachment"])
        
        return result
    
    def _cache_attachments(self, issue_key: str, attachments: List[Dict]) -> None:
        """Cache issue attachments, especially focusing on images."""
        issue_dir = self.image_cache_dir / issue_key
        issue_dir.mkdir(exist_ok=True)
        
        for attachment in attachments:
            if not attachment.get("mimeType", "").startswith("image/"):
                continue  # Only cache images for now
                
            attachment_id = attachment["id"]
            filename = attachment["filename"]
            content_url = attachment["content"]
            
            # Check if already cached
            file_path = issue_dir / f"{attachment_id}_{filename}"
            if file_path.exists():
                continue
                
            # Download attachment
            try:
                response = self.session.get(content_url, verify=False)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Cached image attachment: {filename} for issue {issue_key}")
            except Exception as e:
                logger.error(f"Failed to download attachment {filename}: {str(e)}")
    
    def get_issue_comments(self, issue_key: str) -> List[Dict]:
        """Get all comments for an issue."""
        issue_data = self.get_issue(issue_key)
        if "fields" in issue_data and "comment" in issue_data["fields"]:
            return issue_data["fields"]["comment"]["comments"]
        return []
    
    def get_issue_attachments(self, issue_key: str) -> List[Dict]:
        """Get all attachments for an issue."""
        issue_data = self.get_issue(issue_key)
        if "fields" in issue_data and "attachment" in issue_data["fields"]:
            return issue_data["fields"]["attachment"]
        return []

    def get_issue_images(self, issue_key: str) -> List[Dict]:
        """Get all image attachments for an issue with local paths."""
        attachments = self.get_issue_attachments(issue_key)
        images = []
        
        for attachment in attachments:
            if attachment.get("mimeType", "").startswith("image/"):
                attachment_id = attachment["id"]
                filename = attachment["filename"]
                local_path = self.image_cache_dir / issue_key / f"{attachment_id}_{filename}"
                
                if local_path.exists():
                    images.append({
                        "id": attachment_id,
                        "filename": filename,
                        "mimeType": attachment.get("mimeType"),
                        "local_path": str(local_path)
                    })
        
        return images
    
    def full_sync(self) -> None:
        """Perform a full synchronization of key Jira data."""
        logger.info("Starting full Jira data synchronization")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=Console()
        ) as progress:
            # Fetch projects
            task = progress.add_task("Fetching projects...", total=None)
            self.get_all_projects(force_refresh=True)
            progress.update(task, description="Projects fetched successfully")
            
            # Fetch recent issues
            task = progress.add_task("Fetching recent issues...", total=None)
            result = self.search_issues(
                "updated >= -30d ORDER BY updated DESC", 
                max_results=200,
                fields=["summary", "description", "status", "priority", "assignee", "reporter", "created", "updated"]
            )
            progress.update(task, description=f"Fetched {len(result.get('issues', []))} recent issues")
            
            # Get full details for recent issues
            if "issues" in result:
                task = progress.add_task("Fetching detailed issue data...", total=len(result["issues"]))
                for issue in result["issues"]:
                    issue_key = issue["key"]
                    self.get_issue(issue_key, force_refresh=True)
                    progress.update(task, advance=1, description=f"Fetched details for {issue_key}")
        
        # Update last sync timestamp
        self.cache["metadata"]["last_full_sync"] = datetime.now()
        self._save_cache()
        logger.info("Full synchronization completed")


class GeminiAIClient:
    """Client for interacting with Google Gemini AI models."""
    
    def __init__(self, project_id: str, location: str, model_name: str):
        """
        Initialize the Gemini AI client.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            model_name: Name of the Gemini model to use
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel(model_name)
            logger.info(f"Initialized Gemini AI model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini AI model: {str(e)}")
            raise

    def create_enhanced_system_prompt(self, jira_client: JiraClient) -> str:
        """
        Create a detailed system prompt with context about Jira and available data.
        
        Args:
            jira_client: The Jira client instance with cached data
            
        Returns:
            A comprehensive system prompt for the Gemini model
        """
        # Get statistics from cache for context
        project_count = len(jira_client.cache.get("projects", {}))
        issue_count = len(jira_client.cache.get("issues", {}))
        
        now = datetime.now()
        
        prompt = f"""
        You are JiraGPT, an exceptionally helpful and intelligent assistant specialized in answering questions about Jira tickets, projects, and documentation within the organization.

        Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}
        
        You have access to:
        - {project_count} Jira projects
        - {issue_count} Jira issues with their details, comments, and attachments
        
        CAPABILITIES:
        1. Answer questions about specific Jira tickets using their contents, comments, descriptions, and metadata.
        2. Analyze and explain technical information found in Jira tickets.
        3. Extract meaning from images, tables, and diagrams that appear in Jira tickets when they are provided to you.
        4. Summarize status of tickets, projects, or specific domains of work.
        5. Provide accurate, concise answers based on the available information.
        6. Always cite the sources (Jira ticket IDs) that you used to generate your answers.

        BEHAVIOR GUIDELINES:
        - Be helpful and concise in your responses. Focus on providing value.
        - When answering questions, prioritize information that directly answers the question.
        - When uncertain, be transparent about your limitations but try to provide helpful information.
        - For technical questions, provide detailed, technically accurate information.
        - Output should be well-formatted, with appropriate use of tables, bullet points, or paragraphs based on the content being presented.
        - IMPORTANT: Always include links to the relevant Jira tickets in your responses, formatted properly for easy access.
        
        RESPONSE FORMAT:
        - Present information in a clear, well-structured manner.
        - Use tables when presenting structured data with multiple attributes.
        - Use bullet points for lists or step-by-step instructions.
        - Use paragraphs for explanations, analyses, and summaries.
        - Always include relevant ticket IDs and links when referencing specific tickets.
        - At the end of detailed responses, provide a concise 1-2 sentence summary of the key point.
        
        OUTPUT STYLE:
        - Professional but conversational
        - Precise and accurate
        - Well-organized
        - Prioritizing clarity and relevance
        
        IMPORTANT: Your goal is to provide the most valuable, accurate information possible without unnecessary back-and-forth. If a question is ambiguous, provide the most comprehensive answer that addresses the likely intent rather than asking for clarification.
        """
        
        return prompt.strip()

    def generate_response(self, prompt: str, system_instructions: str = None, images: List[str] = None, 
                          temperature: float = 0.7, max_tokens: int = 8192) -> str:
        """
        Generate a response using the Gemini model.
        
        Args:
            prompt: The user's query
            system_instructions: System instructions for the model
            images: Paths to image files to include in the prompt
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum tokens in the response
            
        Returns:
            The generated response text
        """
        logger.info(f"Generating response with Gemini for prompt: {prompt[:100]}...")
        
        # Prepare generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=max_tokens,
        )
        
        try:
            # Handle multimodal input (text + images)
            if images and any(Path(img).exists() for img in images):
                valid_images = [img for img in images if Path(img).exists()]
                logger.info(f"Including {len(valid_images)} images in prompt")
                
                # For text-only content
                if system_instructions:
                    full_prompt = f"{system_instructions}\n\n{prompt}"
                else:
                    full_prompt = prompt
                
                # Process images
                image_parts = []
                for img_path in valid_images:
                    try:
                        from PIL import Image
                        # Open with PIL first to ensure proper format
                        img = Image.open(img_path)
                        # Convert to RGB if needed (handles PNG with alpha channel)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Save to BytesIO to get bytes
                        import io
                        buf = io.BytesIO()
                        img.save(buf, format='JPEG')
                        image_bytes = buf.getvalue()
                        # Add to parts
                        image_parts.append(image_bytes)
                    except Exception as e:
                        logger.error(f"Failed to process image {img_path}: {str(e)}")
                
                # Generate response with images
                if image_parts:
                    response = self.model.generate_content(
                        [full_prompt] + image_parts,
                        generation_config=generation_config
                    )
                else:
                    # Fall back to text-only if image processing failed
                    response = self.model.generate_content(
                        full_prompt,
                        generation_config=generation_config
                    )
            else:
                # Text-only input
                if system_instructions:
                    full_prompt = f"{system_instructions}\n\nUser query: {prompt}"
                else:
                    full_prompt = prompt
                
                # Generate response
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.text
            else:
                logger.warning("Empty response from Gemini")
                return "I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            return f"An error occurred while generating the response: {str(e)}"


class JiraGeminiChatbot:
    """Main chatbot class that integrates Jira with Gemini AI."""
    
    def __init__(self, jira_base_url: str, jira_username: str, jira_api_token: str,
                 project_id: str, location: str, model_name: str):
        """
        Initialize the chatbot.
        
        Args:
            jira_base_url: Base URL of the Jira instance
            jira_username: Jira username
            jira_api_token: Jira API token
            project_id: Google Cloud project ID
            location: Google Cloud region
            model_name: Gemini model name
        """
        self.jira_client = JiraClient(jira_base_url, jira_username, jira_api_token)
        self.gemini_client = GeminiAIClient(project_id, location, model_name)
        self.console = Console()
        self.conversation_history = []
        
    def initialize(self) -> None:
        """Initialize the chatbot, ensuring data is synced and ready."""
        # Check if cache needs refreshing
        if not self.jira_client._is_cache_valid():
            self.console.print(Panel("[bold yellow]Cache is outdated or missing. Performing initial data sync...[/]"))
            self.jira_client.full_sync()
            self.console.print(Panel("[bold green]Data sync completed![/]"))
        else:
            self.console.print(Panel("[bold green]Using cached Jira data[/]"))
    
    def find_relevant_tickets(self, query: str, max_results: int = 5) -> List[str]:
        """
        Find tickets relevant to the user's query.
        
        Args:
            query: User's query text
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant ticket keys
        """
        # Extract ticket keys from the query
        ticket_pattern = r'([A-Z]+-\d+)'
        explicit_tickets = re.findall(ticket_pattern, query)
        
        if explicit_tickets:
            # User explicitly mentioned ticket(s)
            return explicit_tickets[:max_results]
        
        # Extract key terms for search
        # Remove common words and punctuation
        stop_words = {"the", "a", "an", "and", "in", "on", "at", "to", "for", "with", "about", "is", "are", "what", "how", "when", "where", "who", "which"}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        if not keywords:
            return []  # No meaningful keywords found
        
        # Build JQL query for text search
        if len(keywords) > 3:
            # Use the most important keywords
            keywords = keywords[:3]
        
        jql = " AND ".join([f'text ~ "{keyword}"' for keyword in keywords])
        jql += " ORDER BY updated DESC"
        
        try:
            results = self.jira_client.search_issues(jql, max_results=max_results)
            if "issues" in results and results["issues"]:
                return [issue["key"] for issue in results["issues"]]
        except Exception as e:
            logger.error(f"Error searching for relevant tickets: {str(e)}")
            
        return []
    
    def get_ticket_context(self, ticket_key: str) -> Dict:
        """
        Get comprehensive context for a ticket.
        
        Args:
            ticket_key: The Jira ticket key
            
        Returns:
            Dictionary with ticket details
        """
        try:
            # Get full ticket data
            ticket = self.jira_client.get_issue(ticket_key)
            
            # Get comments
            comments = self.jira_client.get_issue_comments(ticket_key)
            
            # Get images
            images = self.jira_client.get_issue_images(ticket_key)
            
            # Extract key information
            context = {
                "key": ticket_key,
                "url": f"{self.jira_client.base_url}/browse/{ticket_key}",
                "summary": ticket["fields"].get("summary", ""),
                "status": ticket["fields"].get("status", {}).get("name", "Unknown"),
                "description": ticket["fields"].get("description", {}),
                "comments": comments,
                "created": ticket["fields"].get("created", ""),
                "updated": ticket["fields"].get("updated", ""),
                "assignee": ticket["fields"].get("assignee", {}).get("displayName", "Unassigned"),
                "reporter": ticket["fields"].get("reporter", {}).get("displayName", "Unknown"),
                "priority": ticket["fields"].get("priority", {}).get("name", "None"),
                "images": images,
                "link": f"{self.jira_client.base_url}/browse/{ticket_key}"
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context for ticket {ticket_key}: {str(e)}")
            return {"key": ticket_key, "error": str(e)}
    
    def prepare_ticket_context_for_prompt(self, contexts: List[Dict]) -> Tuple[str, List[str]]:
        """
        Prepare ticket contexts for inclusion in the prompt.
        
        Args:
            contexts: List of ticket context dictionaries
            
        Returns:
            Tuple of (context text, list of image paths)
        """
        context_text = []
        image_paths = []
        
        for ctx in contexts:
            if "error" in ctx:
                continue
                
            ticket_text = f"--- TICKET {ctx['key']} ---\n"
            ticket_text += f"Summary: {ctx['summary']}\n"
            ticket_text += f"Status: {ctx['status']}\n"
            ticket_text += f"Priority: {ctx['priority']}\n"
            ticket_text += f"Assignee: {ctx['assignee']}\n"
            ticket_text += f"Reporter: {ctx['reporter']}\n"
            ticket_text += f"Created: {ctx['created']}\n"
            ticket_text += f"Updated: {ctx['updated']}\n"
            ticket_text += f"Link: {ctx['link']}\n\n"
            
            # Add description
            if ctx.get("description"):
                desc = ctx["description"]
                if isinstance(desc, dict) and "content" in desc:
                    # Try to extract text from Atlassian Document Format
                    text_content = self._extract_text_from_adf(desc)
                    ticket_text += f"Description:\n{text_content}\n\n"
                elif isinstance(desc, str):
                    ticket_text += f"Description:\n{desc}\n\n"
            
            # Add comments (limited to first 5 for brevity)
            if ctx.get("comments"):
                ticket_text += "Comments:\n"
                for i, comment in enumerate(ctx["comments"][:5]):
                    author = comment.get("author", {}).get("displayName", "Unknown")
                    created = comment.get("created", "")
                    
                    body = comment.get("body", {})
                    if isinstance(body, dict) and "content" in body:
                        # Extract text from Atlassian Document Format
                        comment_text = self._extract_text_from_adf(body)
                    elif isinstance(body, str):
                        comment_text = body
                    else:
                        comment_text = str(body)
                    
                    ticket_text += f"[Comment by {author} on {created}]:\n{comment_text}\n\n"
            
            # Reference images
            if ctx.get("images"):
                ticket_text += f"Images: This ticket has {len(ctx['images'])} image attachments\n"
                
                # Add image paths for multimodal prompt
                for img in ctx["images"][:3]:  # Limit to first 3 images
                    if "local_path" in img and Path(img["local_path"]).exists():
                        image_paths.append(img["local_path"])
                        ticket_text += f"- Image: {img['filename']}\n"
            
            context_text.append(ticket_text)
        
        return "\n".join(context_text), image_paths
    
    def _extract_text_from_adf(self, adf_doc: Dict) -> str:
        """Extract plain text from Atlassian Document Format."""
        if not isinstance(adf_doc, dict) or "content" not in adf_doc:
            return str(adf_doc)
            
        extracted_text = []
        
        def extract_recursive(node):
            if isinstance(node, dict):
                if node.get("type") == "text" and "text" in node:
                    extracted_text.append(node["text"])
                elif "content" in node and isinstance(node["content"], list):
                    for child in node["content"]:
                        extract_recursive(child)
            elif isinstance(node, list):
                for item in node:
                    extract_recursive(item)
        
        extract_recursive(adf_doc)
        return " ".join(extracted_text)
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response.
        
        Args:
            query: The user's question or query
            
        Returns:
            The chatbot's response
        """
        try:
            # Find relevant tickets
            relevant_tickets = self.find_relevant_tickets(query)
            
            # Get context for each relevant ticket
            ticket_contexts = [self.get_ticket_context(ticket) for ticket in relevant_tickets]
            
            # Prepare context for the prompt
            context_text, image_paths = self.prepare_ticket_context_for_prompt(ticket_contexts)
            
            # Create the system prompt
            system_prompt = self.gemini_client.create_enhanced_system_prompt(self.jira_client)
            
            # Create the full prompt with context
            full_prompt = f"""
            User Query: {query}
            
            Relevant Jira Tickets:
            {context_text if context_text else "No directly relevant tickets found."}
            
            Please provide a complete, direct answer to the user's query based on the information above.
            Include links to relevant tickets and cite your sources.
            """
            
            # Generate the response
            response = self.gemini_client.generate_response(
                prompt=full_prompt,
                system_instructions=system_prompt,
                images=image_paths if image_paths else None
            )
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error processing your query: {str(e)}\n\nPlease try again with a different question or check the logs for more details."
    
    def format_response(self, response: str) -> str:
        """Format the response for display using Rich."""
        # Check if the response contains a Jira ticket reference
        response = re.sub(r'([A-Z]+-\d+)', 
                          lambda m: f"[link={self.jira_client.base_url}/browse/{m.group(1)}][bold cyan]{m.group(1)}[/bold cyan][/link]", 
                          response)
        
        # Return as Markdown for Rich to render
        return response
    
    def run_cli(self) -> None:
        """Run the chatbot in an interactive CLI loop."""
        self.initialize()
        
        self.console.print(Panel.fit("[bold green]Jira-Gemini Chatbot Initialized[/]"))
        self.console.print("Ask me anything about your Jira tickets! Type 'exit' to quit.\n")
        
        while True:
            query = self.console.input("[bold blue]You:[/] ")
            
            if query.lower() in ('exit', 'quit', 'bye'):
                self.console.print("\n[bold green]Goodbye![/]")
                break
                
            if query.lower() in ('refresh', 'sync'):
                self.console.print(Panel("[bold yellow]Refreshing Jira data...[/]"))
                self.jira_client.full_sync()
                self.console.print(Panel("[bold green]Data refresh completed![/]"))
                continue
                
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                response = self.process_query(query)
                progress.update(task, description="Response ready!")
            
            formatted_response = self.format_response(response)
            self.console.print("\n[bold green]Assistant:[/]")
            self.console.print(Markdown(formatted_response))
            self.console.print("\n" + "-" * 50 + "\n")


def main():
    """Main function to run the chatbot."""
    # Check for environment variables
    jira_base_url = os.environ.get("JIRA_BASE_URL")
    jira_username = os.environ.get("JIRA_USERNAME")
    jira_api_token = os.environ.get("JIRA_API_TOKEN")
    
    if not all([jira_base_url, jira_username, jira_api_token]):
        console = Console()
        console.print(Panel("[bold red]Missing Jira credentials![/]"))
        console.print("""
        Please set the following environment variables:
        - JIRA_BASE_URL: Your Jira instance URL
        - JIRA_USERNAME: Your Jira username (usually email)
        - JIRA_API_TOKEN: Your Jira API token
        
        For Google Cloud credentials, make sure you're authenticated with:
        - gcloud auth application-default login
        """)
        
        # Prompt for credentials if missing
        if not jira_base_url:
            jira_base_url = console.input("[bold yellow]Enter your Jira base URL:[/] ")
        if not jira_username:
            jira_username = console.input("[bold yellow]Enter your Jira username:[/] ")
        if not jira_api_token:
            jira_api_token = console.input("[bold yellow]Enter your Jira API token:[/] ", password=True)
    
    # Initialize and run the chatbot
    try:
        chatbot = JiraGeminiChatbot(
            jira_base_url=jira_base_url,
            jira_username=jira_username,
            jira_api_token=jira_api_token,
            project_id=PROJECT_ID,
            location=REGION,
            model_name=MODEL_NAME
        )
        
        chatbot.run_cli()
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error initializing chatbot:[/] {str(e)}")
        console.print("\nTroubleshooting tips:")
        console.print("1. Make sure your Google Cloud credentials are set up correctly")
        console.print("2. Verify your Jira credentials and API token")
        console.print("3. Check if you have the required permissions in Jira")
        console.print("4. Ensure you're authenticated with gcloud:")
        console.print("   gcloud auth application-default login")


if __name__ == "__main__":
    main()
