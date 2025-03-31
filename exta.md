def _retrieve_and_index_sources(self, sources):
    """
    Retrieve documents from sources and index them.
    
    Args:
        sources: List of source types to retrieve from
    """
    all_chunks = []
    
    # Use ThreadPoolExecutor for parallel document retrieval
    with ThreadPoolExecutor(max_workers=min(len(sources), 3)) as executor:
        future_to_source = {}
        
        # Start document retrieval for each source
        for source in sources:
            if source in self.document_sources:
                source_manager = self.document_sources[source]
                
                # Define a wrapper function to handle the call properly
                def get_docs_from_source(sm):
                    try:
                        return sm.get_documents()
                    except Exception as e:
                        logger.error(f"Error in get_documents for {source}: {str(e)}")
                        return []
                
                future = executor.submit(get_docs_from_source, source_manager)
                future_to_source[future] = source
        
        # Process results as they complete
        for future in future_to_source:
            source = future_to_source[future]
            try:
                documents = future.result()  # Now this returns the actual documents
                if documents:
                    # Process and chunk documents
                    chunks = self.process_documents(documents, source_type=source)
                    all_chunks.extend(chunks)
                    logger.info(f"Retrieved and processed {len(documents)} documents from {source}")
            except Exception as e:
                logger.error(f"Error retrieving documents from {source}: {e}")
    
    # Index all chunks
    if all_chunks:
        self.index_documents(all_chunks)
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(sources)} sources")






















"""
JIRA connector for the RAG system.
"""
import os
import time
import json
from utils.logger import get_logger
from utils.cache import cache_manager
from utils.content_parser import content_parser
from data_sources.jira.client import JIRAClient
import config

logger = get_logger(__name__)

class JIRAConnector:
    """
    Connector for JIRA data source.
    Handles retrieving and processing JIRA content for RAG.
    """
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=True):
        """
        Initialize the JIRA connector.
        
        Args:
            base_url: The base URL of the JIRA server
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.client = JIRAClient(base_url, username, api_token, ssl_verify)
        self.cache_dir = os.path.join(config.CACHE_DIR, 'jira')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("Initialized JIRA connector")
    
    def get_documents(self, projects=None, issue_types=None, max_issues=100, jql=None):
        """
        Get documents from JIRA.
        
        Args:
            projects: Optional list of project keys to limit the search
            issue_types: Optional list of issue types to limit the search
            max_issues: Maximum number of issues to retrieve
            jql: Optional JQL query to search for specific issues
            
        Returns:
            List of processed documents
        """
        # Build cache key
        cache_key = f"jira_documents:{json.dumps(projects)}:{json.dumps(issue_types)}:{max_issues}:{jql}"
        cached_docs = cache_manager.get(cache_key)
        if cached_docs is not None:
            logger.info(f"Using cached JIRA documents ({len(cached_docs)})")
            return cached_docs
        
        logger.info(f"Retrieving JIRA documents (projects={projects}, max_issues={max_issues})")
        
        # Test the connection first
        if not self.client.test_connection():
            logger.error("Failed to connect to JIRA. Check credentials and network.")
            return []
        
        try:
            # Build JQL query if not provided
            if not jql:
                jql_parts = []
                
                if projects:
                    project_clause = " OR ".join([f'project = "{p}"' for p in projects])
                    jql_parts.append(f"({project_clause})")
                
                if issue_types:
                    type_clause = " OR ".join([f'issuetype = "{t}"' for t in issue_types])
                    jql_parts.append(f"({type_clause})")
                
                # Combine all conditions with AND
                if jql_parts:
                    jql = " AND ".join(jql_parts)
                else:
                    jql = ""  # Empty query will get all issues
            
            # Get issues using JQL
            issues = self.client.search_issues(jql, max_results=max_issues)
            
            # Process each issue
            documents = []
            
            for i, issue in enumerate(issues):
                try:
                    # Extract issue key and fields
                    issue_key = issue.get('key')
                    if not issue_key:
                        continue
                    
                    # Extract fields
                    fields = issue.get('fields', {})
                    summary = fields.get('summary', 'No Summary')
                    description = fields.get('description', '')
                    issue_type = fields.get('issuetype', {}).get('name', 'Unknown Type')
                    status = fields.get('status', {}).get('name', 'Unknown Status')
                    priority = fields.get('priority', {}).get('name', 'Unknown Priority')
                    
                    # Parse description
                    parsed_description = content_parser.parse_jira_description(description)
                    
                    # Build URL for source reference
                    source_url = f"{self.client.base_url}/browse/{issue_key}"
                    
                    # Combine fields into text
                    content_parts = [
                        f"Summary: {summary}",
                        f"Description: {parsed_description}",
                        f"Type: {issue_type}",
                        f"Status: {status}",
                        f"Priority: {priority}"
                    ]
                    
                    # Include comments if available
                    comments = fields.get('comment', {}).get('comments', [])
                    if comments:
                        comment_texts = []
                        for comment in comments:
                            author = comment.get('author', {}).get('displayName', 'Unknown')
                            comment_body = comment.get('body', '')
                            parsed_comment = content_parser.parse_jira_description(comment_body)
                            comment_texts.append(f"Comment by {author}: {parsed_comment}")
                        
                        content_parts.append("Comments:\n" + "\n".join(comment_texts))
                    
                    # Create document
                    document = {
                        'text': "\n\n".join(content_parts),
                        'metadata': {
                            'id': issue_key,
                            'title': summary,
                            'source': f"JIRA: {issue_key}",
                            'source_link': source_url,
                            'source_type': 'jira',
                            'issue_type': issue_type,
                            'status': status,
                            'priority': priority
                        }
                    }
                    
                    documents.append(document)
                    
                    # Log progress for large retrievals
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(issues)} JIRA issues")
                        
                except Exception as e:
                    logger.error(f"Error processing JIRA issue {issue.get('key', 'unknown')}: {str(e)}")
            
            # Cache the results
            cache_manager.set(cache_key, documents, timeout=3600 * 4)  # Cache for 4 hours
            
            logger.info(f"Retrieved {len(documents)} documents from JIRA")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving JIRA documents: {str(e)}")
            return []
    
    def search_documents(self, query, max_results=20):
        """
        Search for documents in JIRA.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of processed documents
        """
        logger.info(f"Searching JIRA for: {query}")
        
        try:
            # Use text search in JQL
            jql = f'text ~ "{query}"'
            
            # Get issues matching the query
            issues = self.client.search_issues(jql, max_results=max_results)
            
            if not issues:
                logger.info("No JIRA search results found")
                return []
            
            # Process results similarly to get_documents
            documents = []
            
            for issue in issues:
                try:
                    # Extract issue key and fields
                    issue_key = issue.get('key')
                    if not issue_key:
                        continue
                    
                    # Extract fields
                    fields = issue.get('fields', {})
                    summary = fields.get('summary', 'No Summary')
                    description = fields.get('description', '')
                    issue_type = fields.get('issuetype', {}).get('name', 'Unknown Type')
                    status = fields.get('status', {}).get('name', 'Unknown Status')
                    priority = fields.get('priority', {}).get('name', 'Unknown Priority')
                    
                    # Parse description
                    parsed_description = content_parser.parse_jira_description(description)
                    
                    # Build URL for source reference
                    source_url = f"{self.client.base_url}/browse/{issue_key}"
                    
                    # Combine fields into text
                    content_parts = [
                        f"Summary: {summary}",
                        f"Description: {parsed_description}",
                        f"Type: {issue_type}",
                        f"Status: {status}",
                        f"Priority: {priority}"
                    ]
                    
                    # Create document
                    document = {
                        'text': "\n\n".join(content_parts),
                        'metadata': {
                            'id': issue_key,
                            'title': summary,
                            'source': f"JIRA: {issue_key}",
                            'source_link': source_url,
                            'source_type': 'jira',
                            'issue_type': issue_type,
                            'status': status,
                            'priority': priority
                        }
                    }
                    
                    documents.append(document)
                    
                except Exception as e:
                    logger.error(f"Error processing JIRA search result {issue.get('key', 'unknown')}: {str(e)}")
            
            logger.info(f"Retrieved {len(documents)} search results from JIRA")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching JIRA documents: {str(e)}")
            return []

# Initialize the JIRA connector with default configuration
jira_connector = JIRAConnector(
    base_url=config.JIRA_URL,
    username=config.JIRA_USERNAME,
    api_token=config.JIRA_TOKEN,
    ssl_verify=False
)


























"""
JIRA API client for Enterprise RAG System.
"""
import requests
import logging
import json

import config
from utils.logger import get_logger
from utils.cache import cache_manager

logger = get_logger(__name__)

# Disable insecure request warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class JIRAClient:
    """
    Client for JIRA REST API operations with comprehensive error handling.
    """
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=True):
        """
        Initialize the JIRA client with server and authentication details.
        
        Args:
            base_url: The base URL of the JIRA server
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.base_url = base_url or config.JIRA_URL
        self.username = username or config.JIRA_USERNAME
        self.api_token = api_token or config.JIRA_TOKEN
        self.ssl_verify = ssl_verify
        
        # Configure headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized JIRA client for {self.base_url}")
    
    def test_connection(self):
        """
        Test the connection to JIRA server.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Testing connection to JIRA...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/serverInfo",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                server_info = response.json()
                logger.info(f"Connection to JIRA successful! Server version: {server_info.get('version', 'Unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to JIRA. Check log for details.")
                if hasattr(response, 'text') and response.text:
                    logger.error(f"Status code: {response.status_code}")
                    logger.error(f"Response content: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_issue(self, issue_key, fields=None, expand=None):
        """
        Get a specific issue by its key.
        
        Args:
            issue_key: The Issue Key (e.g., DEMO-1)
            fields: Comma-separated list of field names to include
            expand: Comma-separated list of sections to expand
            
        Returns:
            dict: Issue data or None if not found/error
        """
        cache_key = f"jira_issue:{issue_key}:{fields}:{expand}"
        cached_issue = cache_manager.get(cache_key)
        if cached_issue:
            return cached_issue
        
        logger.info(f"Fetching issue: {issue_key}")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        
        # Build query parameters
        params = {}
        if fields:
            params["fields"] = fields
        if expand:
            params["expand"] = expand
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                params=params,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            issue = response.json()
            
            # Cache the result
            cache_manager.set(cache_key, issue, timeout=3600)  # 1 hour
            
            logger.info(f"Successfully retrieved issue: {issue_key}")
            return issue
        except requests.RequestException as e:
            logger.error(f"Failed to get issue {issue_key}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return None
    
    def search_issues(self, jql, max_results=50, start_at=0, fields=None, expand=None):
        """
        Search for issues using JQL.
        
        Args:
            jql: JQL query string
            max_results: Maximum number of results to return
            start_at: Index of the first result to return
            fields: List of fields to include in the response
            expand: List of sections to expand
            
        Returns:
            list: List of matching issues
        """
        cache_key = f"jira_search:{hash(jql)}:{max_results}:{start_at}:{fields}:{expand}"
        cached_issues = cache_manager.get(cache_key)
        if cached_issues:
            return cached_issues
        
        logger.info(f"Searching issues with JQL: {jql}")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/search"
        
        # Build request payload
        payload = {
            "jql": jql,
            "maxResults": min(max_results, 100),  # JIRA API limit is 100 per request
            "startAt": start_at
        }
        
        if fields:
            if isinstance(fields, list):
                payload["fields"] = fields
            else:
                payload["fields"] = [field.strip() for field in fields.split(',')]
        
        if expand:
            if isinstance(expand, list):
                payload["expand"] = expand
            else:
                payload["expand"] = [exp.strip() for exp in expand.split(',')]
        
        try:
            response = requests.post(
                url,
                auth=(self.username, self.api_token),
                json=payload,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            search_results = response.json()
            
            issues = search_results.get('issues', [])
            total = search_results.get('total', 0)
            
            # If more results are available and max_results is higher than what we got,
            # make additional requests
            current_count = len(issues)
            current_start = start_at + current_count
            
            while current_count < max_results and current_start < total:
                # Update startAt in payload
                payload["startAt"] = current_start
                
                # Make the next request
                next_response = requests.post(
                    url,
                    auth=(self.username, self.api_token),
                    json=payload,
                    headers=self.headers,
                    verify=self.ssl_verify
                )
                
                next_response.raise_for_status()
                next_results = next_response.json()
                
                # Add new issues to the list
                next_issues = next_results.get('issues', [])
                issues.extend(next_issues)
                
                # Update counters
                current_count += len(next_issues)
                current_start += len(next_issues)
                
                # Break if we got fewer issues than requested (end of results)
                if len(next_issues) < payload["maxResults"]:
                    break
            
            # Trim to max_results
            if len(issues) > max_results:
                issues = issues[:max_results]
            
            # Cache the result
            cache_manager.set(cache_key, issues, timeout=1800)  # 30 minutes
            
            logger.info(f"Search returned {len(issues)} issues")
            return issues
        except requests.RequestException as e:
            logger.error(f"Failed to search issues: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return []
    
    def get_projects(self):
        """
        Get all projects.
        
        Returns:
            list: List of projects
        """
        cache_key = "jira_projects"
        cached_projects = cache_manager.get(cache_key)
        if cached_projects:
            return cached_projects
        
        logger.info("Fetching projects")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/project"
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            projects = response.json()
            
            # Cache the result
            cache_manager.set(cache_key, projects, timeout=3600 * 24)  # 24 hours
            
            logger.info(f"Successfully retrieved {len(projects)} projects")
            return projects
        except requests.RequestException as e:
            logger.error(f"Failed to get projects: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return []
    
    def get_issue_types(self):
        """
        Get all issue types.
        
        Returns:
            list: List of issue types
        """
        cache_key = "jira_issue_types"
        cached_issue_types = cache_manager.get(cache_key)
        if cached_issue_types:
            return cached_issue_types
        
        logger.info("Fetching issue types")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/issuetype"
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            issue_types = response.json()
            
            # Cache the result
            cache_manager.set(cache_key, issue_types, timeout=3600 * 24)  # 24 hours
            
            logger.info(f"Successfully retrieved {len(issue_types)} issue types")
            return issue_types
        except requests.RequestException as e:
            logger.error(f"Failed to get issue types: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return []























