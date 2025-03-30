import json
import requests
import logging
import os
import sys
import urllib3
from datetime import datetime, timedelta
import time
import getpass
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("remedy_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RemedyClient")

class RemedyClient:
    """
    Client for BMC Remedy REST API operations with comprehensive error handling and advanced querying.
    """
    
    def __init__(self, server_url, username=None, password=None, ssl_verify=False):
        """
        Initialize the Remedy client with server and authentication details.
        
        Args:
            server_url: The base URL of the Remedy server (e.g., https://cmegroup-restapi.onbmc.com)
            username: Username for authentication (will prompt if None)
            password: Password for authentication (will prompt if None)
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.token_type = "AR-JWT"
        
        # Handle SSL verification
        if ssl_verify is False:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled.")
            self.ssl_verify = False
        else:
            self.ssl_verify = True
            
        logger.info(f"Initialized Remedy client for {self.server_url}")
    
    def login(self):
        """
        Log in to Remedy and get authentication token.
        
        Returns:
            tuple: (returnVal, token) where returnVal is 1 on success, -1 on failure
        """
        if not self.username:
            self.username = input("Enter Username: ")
        
        if not self.password:
            self.password = getpass.getpass(prompt="Enter Password: ")
        
        logger.info(f"Attempting to login as {self.username}")
        
        url = f"{self.server_url}/api/jwt/login"
        payload = {"username": self.username, "password": self.password}
        headers = {"content-type": "application/x-www-form-urlencoded"}
        
        try:
            r = requests.post(url, data=payload, headers=headers, verify=self.ssl_verify)
            
            if r.status_code == 200:
                self.token = r.text
                logger.info("Login successful")
                return 1, self.token
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                print(f"Failure...")
                print(f"Status Code: {r.status_code}")
                return -1, r.text
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return -1, str(e)
    
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
        if not self.token:
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
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            return result["entries"][0]
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None
    
    def get_incidents_by_date(self, date, status=None, owner_group=None):
        """
        Get all incidents submitted on a specific date.
        
        Args:
            date: The submission date in YYYY-MM-DD format
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents for date: {date}")
        
        # Parse the date and create date range (entire day)
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
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
                logger.info(f"Retrieved {len(result['entries'])} incidents for date {date}")
                return result["entries"]
            else:
                logger.warning(f"No incidents found for date {date} or error occurred")
                return []
                
        except ValueError:
            logger.error(f"Invalid date format: {date}. Use YYYY-MM-DD.")
            return []
    
    def get_incidents_by_status(self, status, limit=100):
        """
        Get incidents by their status.
        
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
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
        if not self.token:
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
        if not self.token:
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
    
    def create_incident(self, summary, description, impact="4-Minor/Localized", urgency="4-Low", 
                      reported_source="Direct Input", service_type="User Service Restoration", 
                      assigned_group=None):
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
        if not self.token:
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
        if not self.token:
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
        if not self.token:
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

# Example usage
def test_remedy_client():
    # Create the client
    server_url = "https://cmegroup-restapi.onbmc.com"
    client = RemedyClient(server_url, ssl_verify=False)
    
    # Login
    status, token = client.login()
    if status != 1:
        print("Failed to login. Exiting.")
        return
    
    print("Successfully logged in!")
    
    # Get a specific incident
    incident_id = input("Enter Incident Number: ")
    incident = client.get_incident(incident_id)
    
    if incident:
        # Process the incident for display
        processed = client.process_incident_for_rag(incident)
        print("\nIncident Details:")
        print(f"Incident Number: {processed['metadata']['incident_number']}")
        print(f"Summary: {processed['metadata']['summary']}")
        print(f"Status: {processed['metadata']['status']}")
        print(f"Owner: {processed['metadata']['owner']}")
        print(f"Description: {processed['raw_data'].get('Description', 'N/A')[:100]}...")
    else:
        print(f"Could not find incident {incident_id}")
    
    # Try a more complex query: closed incidents from a specific owner group
    print("\nFetching closed incidents owned by TOCC Support:")
    closed_incidents = client.query_form(
        "HPD:Help Desk",
        "'Status'=\"Closed\" AND 'Owner Group'=\"TOCC Support\"",
        ["Incident Number", "Summary", "Status", "Submit Date", "Owner Group"],
        limit=5
    )
    
    if closed_incidents and "entries" in closed_incidents:
        print(f"Found {len(closed_incidents['entries'])} incidents")
        
        for i, inc in enumerate(closed_incidents['entries'][:5], 1):
            values = inc.get('values', {})
            print(f"\n{i}. {values.get('Incident Number')} - {values.get('Summary')}")
            print(f"   Status: {values.get('Status')} | Submitted: {values.get('Submit Date')}")
    
    # Query by date example
    print("\nEnter a date to find incidents (YYYY-MM-DD):")
    date_query = input("> ")
    date_incidents = client.get_incidents_by_date(date_query)
    
    if date_incidents:
        print(f"Found {len(date_incidents)} incidents on {date_query}")
        
        for i, inc in enumerate(date_incidents[:5], 1):
            values = inc.get('values', {})
            print(f"\n{i}. {values.get('Incident Number')} - {values.get('Summary')}")
            print(f"   Status: {values.get('Status')} | Owner: {values.get('Owner')}")
    else:
        print(f"No incidents found for date {date_query}")
    
    # Clean up
    client.logout()
    print("\nLogged out successfully!")

if __name__ == "__main__":
    test_remedy_client()















import requests
import logging
import json
import os
import sys
import urllib3
from requests.auth import HTTPBasicAuth
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("JiraClient")

class JiraClient:
    """Client for Jira REST API operations with comprehensive error handling and SSL options."""
    
    def __init__(self, base_url, username, api_token, ssl_verify=True, ca_bundle=None):
        """
        Initialize the Jira client with authentication details and SSL options.
        
        Args:
            base_url: The base URL of the Jira instance (e.g., https://cmegroup-sandbox-461.atlassian.net)
            username: The email address for authentication
            api_token: The API token for authentication
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification - SECURITY RISK)
            ca_bundle: Path to a CA bundle file to use for verification (alternative to disabling verification)
        """
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, api_token)
        self.api_url = f"{self.base_url}/rest/api/3"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Handle SSL verification
        if ssl_verify is False:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled. This is not recommended for production use.")
            self.ssl_verify = False
        elif ca_bundle is not None:
            # Use custom CA bundle if provided
            if os.path.exists(ca_bundle):
                logger.info(f"Using custom CA bundle: {ca_bundle}")
                self.ssl_verify = ca_bundle
            else:
                logger.warning(f"CA bundle file not found: {ca_bundle}. Falling back to default verification.")
                self.ssl_verify = True
        else:
            # Default: verify certificates
            self.ssl_verify = True
            
        logger.info(f"Initialized Jira client for {self.base_url}")
        
    def test_connection(self):
        """Test the connection to Jira API."""
        try:
            logger.info("Testing connection to Jira...")
            # Try to get server info, which requires minimal permissions
            response = requests.get(
                f"{self.base_url}/rest/api/3/serverInfo",
                auth=self.auth,
                headers=self.headers,
                verify=self.ssl_verify
            )
            response.raise_for_status()
            server_info = response.json()
            logger.info(f"Connection successful! Server version: {server_info.get('version', 'Unknown')}")
            return True
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return False
            
    def get_issue(self, issue_key, fields=None, expand=None):
        """
        Get a specific issue by its key.
        
        Args:
            issue_key: The issue key (e.g., DEMO-1)
            fields: Comma-separated string of field names to include
            expand: Comma-separated string of sections to expand
        """
        params = {}
        if fields:
            params["fields"] = fields
        if expand:
            params["expand"] = expand
            
        try:
            logger.info(f"Fetching issue: {issue_key}")
            response = requests.get(
                f"{self.api_url}/issue/{issue_key}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            response.raise_for_status()
            issue = response.json()
            logger.info(f"Successfully retrieved issue: {issue_key}")
            return issue
        except requests.RequestException as e:
            logger.error(f"Failed to get issue {issue_key}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
            
    def search_issues(self, jql, start_at=0, max_results=50, fields=None, expand=None):
        """
        Search for issues using JQL (Jira Query Language).
        
        Args:
            jql: JQL search string
            start_at: Starting index for pagination
            max_results: Maximum number of results to return
            fields: Comma-separated string of field names to include
            expand: Comma-separated string of sections to expand
        """
        try:
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": max_results
            }
            
            if fields:
                params["fields"] = fields
            if expand:
                params["expand"] = expand
                
            logger.info(f"Searching issues with JQL: {jql}")
            response = requests.get(
                f"{self.api_url}/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            response.raise_for_status()
            search_results = response.json()
            total = search_results.get("total", 0)
            results_count = len(search_results.get("issues", []))
            logger.info(f"Search returned {results_count} issues (total: {total})")
            return search_results
        except requests.RequestException as e:
            logger.error(f"Failed to search issues: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_all_issues(self, jql, fields=None, expand=None, max_results=1000):
        """
        Get all issues matching a JQL query, handling pagination.
        
        Args:
            jql: JQL search string
            fields: Comma-separated string of field names to include
            expand: Comma-separated string of sections to expand
            max_results: Maximum total number of results to return
        """
        all_issues = []
        start_at = 0
        page_size = 100  # Jira recommends 100 for optimal performance
        
        logger.info(f"Retrieving all issues matching JQL: {jql}")
        
        while True:
            try:
                search_results = self.search_issues(
                    jql=jql, 
                    start_at=start_at, 
                    max_results=page_size,
                    fields=fields,
                    expand=expand
                )
                
                if not search_results or not search_results.get("issues"):
                    break
                    
                issues = search_results.get("issues", [])
                all_issues.extend(issues)
                logger.info(f"Retrieved {len(issues)} issues (total: {len(all_issues)})")
                
                # Check if we've reached the total or our max limit
                if len(all_issues) >= min(search_results.get("total", 0), max_results):
                    break
                    
                # Move to next page
                start_at += len(issues)
                
                # If no issues were returned, we're done
                if len(issues) == 0:
                    break
            except Exception as e:
                logger.error(f"Error retrieving all issues: {str(e)}")
                break
                
        logger.info(f"Retrieved a total of {len(all_issues)} issues")
        return all_issues
    
    def get_issue_types(self):
        """Get all issue types defined in the Jira instance."""
        try:
            logger.info("Fetching issue types...")
            response = requests.get(
                f"{self.api_url}/issuetype",
                auth=self.auth,
                headers=self.headers,
                verify=self.ssl_verify
            )
            response.raise_for_status()
            issue_types = response.json()
            logger.info(f"Successfully retrieved {len(issue_types)} issue types")
            return issue_types
        except requests.RequestException as e:
            logger.error(f"Failed to get issue types: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_projects(self):
        """Get all projects visible to the authenticated user."""
        try:
            logger.info("Fetching projects...")
            response = requests.get(
                f"{self.api_url}/project",
                auth=self.auth,
                headers=self.headers,
                verify=self.ssl_verify
            )
            response.raise_for_status()
            projects = response.json()
            logger.info(f"Successfully retrieved {len(projects)} projects")
            return projects
        except requests.RequestException as e:
            logger.error(f"Failed to get projects: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def create_issue(self, project_key, issue_type, summary, description, fields=None):
        """
        Create a new issue.
        
        Args:
            project_key: The project key
            issue_type: The issue type name or ID
            summary: The issue summary
            description: The issue description
            fields: Dictionary of additional fields to set
        """
        # Base issue data
        issue_data = {
            "fields": {
                "project": {
                    "key": project_key
                },
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": description
                                }
                            ]
                        }
                    ]
                },
                "issuetype": {
                    "name": issue_type
                }
            }
        }
        
        # Add additional fields if provided
        if fields:
            issue_data["fields"].update(fields)
            
        try:
            logger.info(f"Creating issue in project {project_key} of type {issue_type}")
            response = requests.post(
                f"{self.api_url}/issue",
                auth=self.auth,
                headers=self.headers,
                json=issue_data,
                verify=self.ssl_verify
            )
            response.raise_for_status()
            new_issue = response.json()
            logger.info(f"Successfully created issue: {new_issue.get('key')}")
            return new_issue
        except requests.RequestException as e:
            logger.error(f"Failed to create issue: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_issue_content(self, issue_key):
        """
        Get the content of an issue in a format suitable for RAG.
        
        Args:
            issue_key: The issue key (e.g., DEMO-1)
        """
        try:
            # Get issue with all relevant fields
            fields_to_include = "summary,description,issuetype,status,created,updated,assignee,reporter,priority,labels,components,fixVersions,resolution,comment"
            issue = self.get_issue(issue_key, fields=fields_to_include)
            
            if not issue:
                return None
                
            fields = issue.get("fields", {})
            
            # Extract key metadata
            metadata = {
                "key": issue.get("key"),
                "id": issue.get("id"),
                "type": fields.get("issuetype", {}).get("name"),
                "status": fields.get("status", {}).get("name"),
                "created": fields.get("created"),
                "updated": fields.get("updated"),
                "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
                "labels": fields.get("labels", []),
                "resolution": fields.get("resolution", {}).get("name") if fields.get("resolution") else None,
                "url": f"{self.base_url}/browse/{issue.get('key')}"
            }
            
            # Extract people
            if fields.get("assignee"):
                metadata["assignee"] = fields.get("assignee", {}).get("displayName")
                
            if fields.get("reporter"):
                metadata["reporter"] = fields.get("reporter", {}).get("displayName")
            
            # Extract content fields
            content_parts = []
            
            # Add summary
            summary = fields.get("summary", "")
            if summary:
                content_parts.append(f"Summary: {summary}")
            
            # Add description
            # Note: this is simplified and assumes the description is in plain text
            # In reality, Jira descriptions can be in various formats
            description = fields.get("description")
            if description:
                if isinstance(description, dict):
                    # Try to extract text from Atlassian Document Format
                    desc_text = self._extract_text_from_adf(description)
                    if desc_text:
                        content_parts.append(f"Description: {desc_text}")
                else:
                    content_parts.append(f"Description: {description}")
            
            # Add comments
            comments = fields.get("comment", {}).get("comments", [])
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "Unknown")
                created = comment.get("created", "")
                
                comment_body = comment.get("body")
                if isinstance(comment_body, dict):
                    # Extract text from Atlassian Document Format
                    comment_text = self._extract_text_from_adf(comment_body)
                    if comment_text:
                        content_parts.append(f"Comment by {author} on {created}: {comment_text}")
                else:
                    content_parts.append(f"Comment by {author} on {created}: {comment_body}")
            
            # Combine all content
            full_content = "\n\n".join(content_parts)
            
            return {
                "metadata": metadata,
                "content": full_content
            }
        except Exception as e:
            logger.error(f"Error processing issue content: {str(e)}")
            return None
    
    def _extract_text_from_adf(self, adf_doc):
        """
        Extract plain text from Atlassian Document Format (ADF).
        This is a simplified version that doesn't handle all ADF features.
        
        Args:
            adf_doc: The ADF document object
        """
        if not adf_doc or not isinstance(adf_doc, dict):
            return ""
            
        text_parts = []
        
        def extract_from_content(content_list):
            parts = []
            if not content_list or not isinstance(content_list, list):
                return parts
                
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                    
                # Extract text nodes
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                
                # Extract text from content recursively
                if "content" in item and isinstance(item["content"], list):
                    parts.extend(extract_from_content(item["content"]))
            
            return parts
        
        # Extract text from the main content array
        if "content" in adf_doc and isinstance(adf_doc["content"], list):
            text_parts = extract_from_content(adf_doc["content"])
        
        return " ".join(text_parts)

# Script execution
if __name__ == "__main__":
    # Use environment variables or command line args for better security
    jira_url = os.environ.get("JIRA_URL", "https://cmegroup-sandbox-461.atlassian.net")
    username = os.environ.get("JIRA_USERNAME", "firstname.lastname@cmegroup.com")
    api_token = os.environ.get("JIRA_API_TOKEN", "your-api-token-here")
    
    # SSL verification options
    # Option 1: Use default verification
    # ssl_verify = True
    
    # Option 2: Disable verification (NOT RECOMMENDED for production)
    ssl_verify = False
    
    # Option 3: Use custom CA bundle
    # ssl_verify = "/path/to/your/ca-bundle.pem"
    
    # Create the client with SSL verification settings
    client = JiraClient(jira_url, username, api_token, ssl_verify=ssl_verify)
    
    # Test the connection
    if client.test_connection():
        print("\n✅ Connection to Jira successful!")
    else:
        print("\n❌ Failed to connect to Jira. Check log for details.")
        sys.exit(1)
    
    # Get projects
    projects = client.get_projects()
    if projects:
        print(f"\n✅ Successfully retrieved {len(projects)} projects")
        if len(projects) > 0:
            print("Sample projects:")
            for project in projects[:3]:  # Show top 3
                print(f"  - {project.get('name')} (Key: {project.get('key')})")
    else:
        print("\n❌ Failed to retrieve projects")
    
    # Rest of the code remains the same...
    print("\nJira integration test completed.")















from atlassian import Confluence
import logging
import sys
import os
import json
from html.parser import HTMLParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ConfluenceClient")

class HTMLFilter(HTMLParser):
    """Parse HTML content and extract plain text for RAG processing."""
    text = ""
    def handle_data(self, data):
        self.text += data + " "
    
    def get_text(self):
        return self.text.strip()

class ConfluenceClient:
    """Client for Confluence API operations using the atlassian-python-api package."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://cmegroup.atlassian.net/wiki)
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url
        self.username = username
        self.api_token = api_token
        
        # Initialize the Confluence client
        logger.info(f"Initializing Confluence client for {base_url}")
        self.client = Confluence(
            url=base_url,
            username=username,
            password=api_token,
            verify_ssl=False,
            cloud=True  # Explicitly set for Confluence Cloud
        )
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            # Simple API call to test connection
            server_info = self.client.get_all_spaces(start=0, limit=1)
            logger.info("Connection successful!")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_all_spaces(self):
        """Get all spaces the user has access to."""
        try:
            logger.info("Retrieving all spaces...")
            spaces = self.client.get_all_spaces()
            logger.info(f"Successfully retrieved {len(spaces)} spaces")
            return spaces
        except Exception as e:
            logger.error(f"Failed to retrieve spaces: {str(e)}")
            return []
    
    def get_space_info(self, space_key):
        """Get information about a specific space."""
        try:
            logger.info(f"Retrieving information for space {space_key}...")
            space = self.client.get_space(space_key)
            logger.info(f"Successfully retrieved space information for {space_key}")
            return space
        except Exception as e:
            logger.error(f"Failed to retrieve space information for {space_key}: {str(e)}")
            return None
    
    def get_pages_in_space(self, space_key=None, limit=100):
        """
        Get all pages in a specific space or all spaces if space_key is None.
        
        Args:
            space_key: The key of the space to get pages from (None for all spaces)
            limit: Maximum number of pages to retrieve
        """
        try:
            logger.info(f"Retrieving pages from space {space_key or 'all spaces'}...")
            if space_key:
                pages = self.client.get_all_pages_from_space(space_key, limit=limit)
            else:
                # Get pages from all spaces
                pages = []
                spaces = self.get_all_spaces()
                for space in spaces:
                    space_key = space.get('key')
                    try:
                        space_pages = self.client.get_all_pages_from_space(space_key, limit=limit)
                        pages.extend(space_pages)
                        logger.info(f"Retrieved {len(space_pages)} pages from space {space_key}")
                    except Exception as e:
                        logger.error(f"Error retrieving pages from space {space_key}: {str(e)}")
            
            logger.info(f"Successfully retrieved {len(pages)} pages total")
            return pages
        except Exception as e:
            logger.error(f"Failed to retrieve pages: {str(e)}")
            return []
    
    def search_content(self, cql, limit=100):
        """
        Search for content using CQL (Confluence Query Language).
        
        Args:
            cql: The CQL query string
            limit: Maximum number of results to return
        """
        try:
            logger.info(f"Searching for content with CQL: {cql}")
            results = self.client.cql(cql, limit=limit)
            result_count = len(results.get('results', []))
            logger.info(f"Search returned {result_count} results")
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"results": []}
    
    def get_page_by_id(self, page_id, expand=None):
        """
        Get a specific page by its ID.
        
        Args:
            page_id: The ID of the page
            expand: Optional list of properties to expand
        """
        try:
            logger.info(f"Retrieving page with ID {page_id}")
            expand_str = ",".join(expand) if expand else "body.storage"
            page = self.client.get_page_by_id(page_id, expand=expand_str)
            logger.info(f"Successfully retrieved page: {page.get('title', 'Unknown')}")
            return page
        except Exception as e:
            logger.error(f"Failed to retrieve page with ID {page_id}: {str(e)}")
            return None
    
    def get_page_content(self, page_id):
        """
        Get the content of a page in a format suitable for RAG.
        
        Args:
            page_id: The ID of the page
        """
        try:
            logger.info(f"Retrieving content for page with ID {page_id}")
            page = self.get_page_by_id(page_id, expand=["body.storage", "version"])
            if not page:
                return None
            
            # Extract metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "version": page.get("version", {}).get("number"),
                "url": f"{self.base_url}/pages/viewpage.action?pageId={page.get('id')}"
            }
            
            # Extract content
            body = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Parse HTML to get plain text
            html_filter = HTMLFilter()
            html_filter.feed(body)
            plain_text = html_filter.get_text()
            
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": body
            }
        except Exception as e:
            logger.error(f"Failed to process page content for {page_id}: {str(e)}")
            return None
    
    def get_all_content_for_rag(self, space_key=None, limit_per_space=100):
        """
        Get all content from Confluence in a format suitable for RAG.
        
        Args:
            space_key: Optional space key to limit the retrieval
            limit_per_space: Maximum number of pages to retrieve per space
        """
        try:
            logger.info(f"Retrieving all content for RAG from {space_key or 'all spaces'}")
            all_pages = self.get_pages_in_space(space_key, limit=limit_per_space)
            
            processed_content = []
            for page in all_pages:
                try:
                    page_content = self.get_page_content(page.get('id'))
                    if page_content:
                        processed_content.append({
                            "id": page.get('id'),
                            "title": page.get('title'),
                            "space_key": page.get('space', {}).get('key'),
                            "url": page_content.get('metadata', {}).get('url'),
                            "content": page_content.get('content'),
                            "metadata": page_content.get('metadata')
                        })
                except Exception as e:
                    logger.error(f"Error processing page {page.get('id')}: {str(e)}")
            
            logger.info(f"Successfully retrieved and processed {len(processed_content)} pages for RAG")
            return processed_content
        except Exception as e:
            logger.error(f"Failed to retrieve content for RAG: {str(e)}")
            return []

# Script execution
if __name__ == "__main__":
    # Use environment variables or command line arguments for better security
    confluence_url = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net/wiki")
    username = os.environ.get("CONFLUENCE_USERNAME", "cli_api_user")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "your-api-token-here")
    
    # Create the client
    client = ConfluenceClient(confluence_url, username, api_token)
    
    # Test the connection
    if client.test_connection():
        print("\n✅ Connection to Confluence successful!")
    else:
        print("\n❌ Failed to connect to Confluence. Check log for details.")
        sys.exit(1)
    
    # Get all spaces
    spaces = client.get_all_spaces()
    if spaces:
        print(f"\n✅ Successfully retrieved {len(spaces)} spaces")
        if len(spaces) > 0:
            print("Sample spaces:")
            for space in spaces[:3]:  # Show top 3
                print(f"  - {space.get('name')} (Key: {space.get('key')})")
    else:
        print("\n❌ Failed to retrieve spaces or no spaces available")
    
    # Search for content
    search_results = client.search_content("type=page AND title~Test")
    results = search_results.get("results", [])
    if results:
        print(f"\n✅ Search returned {len(results)} results")
        for page in results[:3]:  # Show top 3
            print(f"  - {page.get('title')} (ID: {page.get('id')})")
    else:
        print("\n❌ Search returned no results")
    
    # Get pages from all spaces (limited to conserve resources)
    print("\nRetrieving sample pages from spaces...")
    for space in spaces[:2]:  # Only check first 2 spaces to avoid long runtime
        space_key = space.get('key')
        pages = client.get_pages_in_space(space_key, limit=5)  # Limit to 5 pages per space
        print(f"Space {space_key}: Retrieved {len(pages)} pages")
        
        # Get content of first page as example (if available)
        if pages:
            first_page = pages[0]
            page_id = first_page.get('id')
            page_content = client.get_page_content(page_id)
            if page_content:
                print(f"  Sample page: {first_page.get('title')}")
                content_preview = page_content.get('content', '')[:100].replace('\n', ' ').strip()
                print(f"  Content preview: {content_preview}...")
    
    print("\nConfluence integration test completed.")











import requests
import logging
import json
import os
import sys
from requests.auth import HTTPBasicAuth
import urllib3
from html.parser import HTMLParser

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ConfluenceClient")

class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data):
        self.text += data + " "

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://cmegroup.atlassian.net)
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, api_token)
        self.api_url = f"{self.base_url}/wiki/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ConfluenceClient/1.0 Python/Requests"
        }
        logger.info(f"Initialized Confluence client for {self.base_url}")
        
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.api_url}/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (test connection): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received during connection test")
                return True  # Still consider it a success if status code is OK
            
            try:
                response.json()
                logger.info("Connection successful!")
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Connection succeeded but received invalid JSON: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return False
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand (e.g., "body.storage,version,space")
        """
        try:
            params = {}
            if expand:
                params["expand"] = expand
                
            logger.info(f"Fetching content with ID: {content_id}, expand: {expand}")
            response = requests.get(
                f"{self.api_url}/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (content by ID): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning(f"Empty response received for content ID: {content_id}")
                return None
            
            try:
                content = response.json()
                logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
                return content
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for content ID {content_id}: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Failed to get content by ID: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
            
    def search_content(self, cql=None, title=None, content_type="page", 
                      expand=None, limit=10, start=0):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            content_type: Type of content to search for (default: page)
            expand: Properties to expand in results
            limit: Maximum number of results to return
            start: Starting index for pagination
        """
        params = {
            "limit": limit,
            "start": start
        }
        
        if cql:
            params["cql"] = cql
        else:
            # Build CQL if not provided
            query_parts = []
            
            if content_type:
                query_parts.append(f"type={content_type}")
                
            if title:
                # Escape special characters in title
                safe_title = title.replace('"', '\\"')
                query_parts.append(f'title~"{safe_title}"')
                
            if query_parts:
                params["cql"] = " AND ".join(query_parts)
        
        if expand:
            params["expand"] = expand
            
        try:
            logger.info(f"Searching for content with params: {params}")
            response = requests.get(
                f"{self.api_url}/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (search): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received for search query")
                return {"results": []}
            
            try:
                results = response.json()
                logger.info(f"Search returned {len(results.get('results', []))} results")
                return results
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for search: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return {"results": []}
                
        except requests.RequestException as e:
            logger.error(f"Failed to search content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return {"results": []}
    
    def get_page_content(self, page_id):
        """
        Get the content of a page in a suitable format for RAG.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
        """
        try:
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
                
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki/pages/viewpage.action?pageId={page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process the HTML content
            html_filter = HTMLFilter()
            html_filter.feed(content)
            plain_text = html_filter.text
            
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content
            }
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None

    def get_all_content(self, content_type="page", limit=100, expand=None):
        """
        Retrieve all content of specified type with pagination handling.
        
        Args:
            content_type: Type of content to retrieve (default: page)
            limit: Maximum number of results per request
            expand: Properties to expand in results
        """
        all_content = []
        start = 0
        
        logger.info(f"Retrieving all {content_type} content")
        
        while True:
            try:
                params = {
                    "type": content_type,
                    "limit": limit,
                    "start": start
                }
                
                if expand:
                    params["expand"] = expand
                
                response = requests.get(
                    f"{self.api_url}/content",
                    auth=self.auth,
                    headers=self.headers,
                    params=params,
                    verify=False
                )
                response.raise_for_status()
                
                # Print raw response for debugging
                raw_content = response.text
                logger.info(f"Raw response content (all content): {raw_content[:500]}...")
                
                # Handle empty response
                if not raw_content.strip():
                    logger.warning(f"Empty response received when retrieving all {content_type} content")
                    break
                
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON when retrieving content: {str(e)}")
                    logger.error(f"Response content: {raw_content}")
                    break
                
                results = data.get("results", [])
                if not results:
                    break
                    
                all_content.extend(results)
                logger.info(f"Retrieved {len(results)} {content_type}s (total: {len(all_content)})")
                
                # Check if there are more pages
                if len(results) < limit:
                    break
                    
                start += limit
            except requests.RequestException as e:
                logger.error(f"Error retrieving content: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Status code: {e.response.status_code}")
                    try:
                        error_details = e.response.json()
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    except:
                        logger.error(f"Response content: {e.response.text}")
                break
                
        logger.info(f"Retrieved a total of {len(all_content)} {content_type}s")
        return all_content
    
    def get_spaces(self, limit=25, start=0):
        """
        Get all spaces the user has access to.
        
        Args:
            limit: Maximum number of results per request
            start: Starting index for pagination
        """
        try:
            params = {
                "limit": limit,
                "start": start
            }
            
            logger.info("Fetching spaces...")
            response = requests.get(
                f"{self.api_url}/space",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (spaces): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received when fetching spaces")
                return {"results": []}
            
            try:
                spaces = response.json()
                logger.info(f"Successfully retrieved {len(spaces.get('results', []))} spaces")
                return spaces
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for spaces: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return {"results": []}
                
        except requests.RequestException as e:
            logger.error(f"Failed to get spaces: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return {"results": []}
    
    def get_all_spaces(self):
        """
        Retrieve all spaces with pagination handling.
        """
        all_spaces = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
        logger.info("Retrieving all spaces")
        
        while True:
            spaces = self.get_spaces(limit=limit, start=start)
            if not spaces:
                break
                
            results = spaces.get("results", [])
            if not results:
                break
                
            all_spaces.extend(results)
            logger.info(f"Retrieved {len(results)} spaces (total: {len(all_spaces)})")
            
            # Check if there are more spaces
            if len(results) < limit:
                break
                
            # Check the '_links' for a 'next' link
            links = spaces.get("_links", {})
            if not links.get("next"):
                break
                
            start += limit
                
        logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
        return all_spaces

    def try_basic_api_call(self):
        """
        Try a different API endpoint to check connectivity and permissions.
        """
        try:
            logger.info("Trying basic API call to /content/search...")
            response = requests.get(
                f"{self.api_url}/content/search",
                auth=self.auth,
                headers=self.headers,
                params={"cql": "type=page", "limit": 1},
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response from basic API call: {raw_content[:500]}...")
            
            return {
                "status_code": response.status_code,
                "content_length": len(raw_content),
                "is_json": self._is_valid_json(raw_content)
            }
        except requests.RequestException as e:
            logger.error(f"Basic API call failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                return {
                    "status_code": e.response.status_code,
                    "error": str(e),
                    "content": e.response.text[:500] if hasattr(e.response, 'text') else None
                }
            return {"error": str(e)}

    def _is_valid_json(self, text):
        """Check if a string is valid JSON."""
        if not text or not text.strip():
            return False
            
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

# Script execution
if __name__ == "__main__":
    # Use environment variables or command line arguments for better security
    confluence_url = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
    username = os.environ.get("CONFLUENCE_USERNAME", "cli_api_user")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "your-api-token-here")
    
    # Create the client
    client = ConfluenceClient(confluence_url, username, api_token)
    
    # Test the connection
    if client.test_connection():
        print("\n✅ Connection to Confluence successful!")
    else:
        print("\n❌ Failed to connect to Confluence. Check log for details.")
        
        # Try a basic API call to diagnose issues
        print("\nTrying a basic API call to diagnose issues...")
        result = client.try_basic_api_call()
        print(f"Basic API call result: {result}")
        
        # Continue with the script anyway to see if other calls work
    
    # Get all spaces (if access is available)
    try:
        spaces = client.get_all_spaces()
        if spaces:
            print(f"\n✅ Successfully retrieved {len(spaces)} spaces")
            if len(spaces) > 0:
                print("Sample spaces:")
                for space in spaces[:3]:  # Show top 3
                    print(f"  - {space.get('name')} (Key: {space.get('key')})")
        else:
            print("\n❌ Failed to retrieve spaces or no spaces available")
    except Exception as e:
        print(f"\n❌ Error retrieving spaces: {str(e)}")
    
    # Search for content
    search_title = "Test"
    search_results = client.search_content(title=search_title)
    if search_results and search_results.get("results"):
        print(f"\n✅ Search for '{search_title}' returned {len(search_results.get('results'))} results")
        for page in search_results.get("results")[:3]:  # Show top 3
            print(f"  - {page.get('title')} (ID: {page.get('id')})")
    else:
        print(f"\n❌ Search for '{search_title}' failed or returned no results")
    
    # Get all content
    try:
        all_pages = client.get_all_content(content_type="page", limit=25)
        if all_pages:
            print(f"\n✅ Retrieved {len(all_pages)} pages from Confluence")
            if len(all_pages) > 0:
                print("Sample pages:")
                for page in all_pages[:3]:  # Show top 3
                    print(f"  - {page.get('title')} (ID: {page.get('id')})")
        else:
            print("\n❌ Failed to retrieve pages or no pages available")
    except Exception as e:
        print(f"\n❌ Error retrieving pages: {str(e)}")
    
    print("\nConfluence integration test completed.")










import requests
import logging
import json
import os
import sys
from requests.auth import HTTPBasicAuth
from urllib.parse import quote
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ConfluenceClient")

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url, username, api_token, space_key=None):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://cmegroup.atlassian.net/wiki)
            username: The username for authentication
            api_token: The API token for authentication
            space_key: The default space key to use for operations
        """
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, api_token)
        self.space_key = space_key
        self.api_url = f"{self.base_url}/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized Confluence client for {self.base_url}")
        
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.api_url}/space",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1}
            )
            response.raise_for_status()
            logger.info("Connection successful!")
            return True
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return False
            
    def get_space_info(self, space_key=None):
        """Get information about a specific space."""
        space_key = space_key or self.space_key
        if not space_key:
            logger.error("No space key provided.")
            return None
            
        try:
            logger.info(f"Fetching information for space: {space_key}")
            response = requests.get(
                f"{self.api_url}/space/{space_key}",
                auth=self.auth,
                headers=self.headers
            )
            response.raise_for_status()
            space_data = response.json()
            logger.info(f"Successfully retrieved space data for {space_key}")
            return space_data
        except requests.RequestException as e:
            logger.error(f"Failed to get space info: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand (e.g., "body.storage,version,space")
        """
        try:
            params = {}
            if expand:
                params["expand"] = expand
                
            logger.info(f"Fetching content with ID: {content_id}, expand: {expand}")
            response = requests.get(
                f"{self.api_url}/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            content = response.json()
            logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
            return content
        except requests.RequestException as e:
            logger.error(f"Failed to get content by ID: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
            
    def search_content(self, cql=None, title=None, space_key=None, content_type="page", 
                      expand=None, limit=10, start=0):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            space_key: Space key to limit search
            content_type: Type of content to search for (default: page)
            expand: Properties to expand in results
            limit: Maximum number of results to return
            start: Starting index for pagination
        """
        space_key = space_key or self.space_key
        params = {
            "limit": limit,
            "start": start
        }
        
        if cql:
            params["cql"] = cql
        else:
            # Build CQL if not provided
            query_parts = []
            
            if content_type:
                query_parts.append(f"type={content_type}")
                
            if space_key:
                query_parts.append(f"space={space_key}")
                
            if title:
                # Escape special characters in title
                safe_title = title.replace('"', '\\"')
                query_parts.append(f'title~"{safe_title}"')
                
            if query_parts:
                params["cql"] = " AND ".join(query_parts)
        
        if expand:
            params["expand"] = expand
            
        try:
            logger.info(f"Searching for content with params: {params}")
            response = requests.get(
                f"{self.api_url}/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            results = response.json()
            logger.info(f"Search returned {len(results.get('results', []))} results")
            return results
        except requests.RequestException as e:
            logger.error(f"Failed to search content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def create_page(self, title, body_content, space_key=None, parent_id=None, content_type="page"):
        """
        Create a new page in Confluence.
        
        Args:
            title: The title of the page
            body_content: The HTML content of the page
            space_key: The space key where the page will be created
            parent_id: Optional parent page ID
            content_type: Type of content to create (default: page)
        """
        space_key = space_key or self.space_key
        if not space_key:
            logger.error("No space key provided for page creation.")
            return None
            
        data = {
            "type": content_type,
            "title": title,
            "space": {"key": space_key},
            "body": {
                "storage": {
                    "value": body_content,
                    "representation": "storage"
                }
            }
        }
        
        if parent_id:
            data["ancestors"] = [{"id": parent_id}]
            
        try:
            logger.info(f"Creating page '{title}' in space {space_key}")
            response = requests.post(
                f"{self.api_url}/content",
                auth=self.auth,
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            new_page = response.json()
            logger.info(f"Successfully created page with ID: {new_page.get('id')}")
            return new_page
        except requests.RequestException as e:
            logger.error(f"Failed to create page: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None

    def get_page_content(self, page_id):
        """
        Get the content of a page in a suitable format for RAG.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
        """
        try:
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
                
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/pages/viewpage.action?pageId={page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process the HTML content (you might want to use BeautifulSoup or similar for better processing)
            # This is a simple version that just strips HTML tags
            from html.parser import HTMLParser

            class HTMLFilter(HTMLParser):
                text = ""
                def handle_data(self, data):
                    self.text += data + " "
            
            html_filter = HTMLFilter()
            html_filter.feed(content)
            plain_text = html_filter.text
            
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content
            }
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None

    def get_all_pages_in_space(self, space_key=None, limit=100):
        """
        Retrieve all pages in a space with pagination handling.
        
        Args:
            space_key: The space key to get pages from
            limit: Maximum number of results per request
        """
        space_key = space_key or self.space_key
        if not space_key:
            logger.error("No space key provided.")
            return []
            
        all_pages = []
        start = 0
        
        logger.info(f"Retrieving all pages from space {space_key}")
        
        while True:
            try:
                params = {
                    "spaceKey": space_key,
                    "limit": limit,
                    "start": start,
                    "expand": "history.lastUpdated,version"
                }
                
                response = requests.get(
                    f"{self.api_url}/content",
                    auth=self.auth,
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    break
                    
                all_pages.extend(results)
                logger.info(f"Retrieved {len(results)} pages (total: {len(all_pages)})")
                
                # Check if there are more pages
                if len(results) < limit:
                    break
                    
                start += limit
            except requests.RequestException as e:
                logger.error(f"Error retrieving pages: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Status code: {e.response.status_code}")
                    try:
                        error_details = e.response.json()
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    except:
                        logger.error(f"Response content: {e.response.text}")
                break
                
        logger.info(f"Retrieved a total of {len(all_pages)} pages from space {space_key}")
        return all_pages

# Script execution
if __name__ == "__main__":
    # Use environment variables or command line arguments for better security
    confluence_url = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net/wiki")
    username = os.environ.get("CONFLUENCE_USERNAME", "cli_api_user")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "your-api-token-here")
    space_key = os.environ.get("CONFLUENCE_SPACE_KEY", "TST")
    
    # Create the client
    client = ConfluenceClient(confluence_url, username, api_token, space_key)
    
    # Test the connection
    if client.test_connection():
        print("\n✅ Connection to Confluence successful!")
    else:
        print("\n❌ Failed to connect to Confluence. Check log for details.")
        sys.exit(1)
    
    # Get space info
    space_info = client.get_space_info()
    if space_info:
        print(f"\n✅ Successfully retrieved space info for {space_key}")
        print(f"Space name: {space_info.get('name')}")
        print(f"Space description: {space_info.get('description', {}).get('plain', {}).get('value', 'No description')}")
    else:
        print(f"\n❌ Failed to get space info for {space_key}")
    
    # Search for content
    search_title = "Test Page"
    search_results = client.search_content(title=search_title)
    if search_results and search_results.get("results"):
        print(f"\n✅ Search for '{search_title}' returned {len(search_results.get('results'))} results")
        for page in search_results.get("results")[:3]:  # Show top 3
            print(f"  - {page.get('title')} (ID: {page.get('id')})")
    else:
        print(f"\n❌ Search for '{search_title}' failed or returned no results")
    
    # Example of getting page content
    if search_results and search_results.get("results"):
        first_page_id = search_results.get("results")[0].get("id")
        page_content = client.get_page_content(first_page_id)
        if page_content:
            print(f"\n✅ Successfully retrieved content for page {first_page_id}")
            print(f"Title: {page_content.get('metadata', {}).get('title')}")
            content_preview = page_content.get('content', '')[:100].replace('\n', ' ').strip()
            print(f"Content preview: {content_preview}...")
        else:
            print(f"\n❌ Failed to retrieve content for page {first_page_id}")
    
    print("\nConfluence integration test completed.")










    import requests
import logging
import json
import os
import sys
from requests.auth import HTTPBasicAuth
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("JiraClient")

class JiraClient:
    """Client for Jira REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Jira client with authentication details.
        
        Args:
            base_url: The base URL of the Jira instance (e.g., https://cmegroup-sandbox-461.atlassian.net)
            username: The email address for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, api_token)
        self.api_url = f"{self.base_url}/rest/api/3"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized Jira client for {self.base_url}")
        
    def test_connection(self):
        """Test the connection to Jira API."""
        try:
            logger.info("Testing connection to Jira...")
            # Try to get server info, which requires minimal permissions
            response = requests.get(
                f"{self.base_url}/rest/api/3/serverInfo",
                auth=self.auth,
                headers=self.headers
            )
            response.raise_for_status()
            server_info = response.json()
            logger.info(f"Connection successful! Server version: {server_info.get('version', 'Unknown')}")
            return True
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return False
            
    def get_issue(self, issue_key, fields=None, expand=None):
        """
        Get a specific issue by its key.
        
        Args:
            issue_key: The issue key (e.g., DEMO-1)
            fields: Comma-separated string of field names to include
            expand: Comma-separated string of sections to expand
        """
        params = {}
        if fields:
            params["fields"] = fields
        if expand:
            params["expand"] = expand
            
        try:
            logger.info(f"Fetching issue: {issue_key}")
            response = requests.get(
                f"{self.api_url}/issue/{issue_key}",
                auth=self.auth,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            issue = response.json()
            logger.info(f"Successfully retrieved issue: {issue_key}")
            return issue
        except requests.RequestException as e:
            logger.error(f"Failed to get issue {issue_key}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
            
    def search_issues(self, jql, start_at=0, max_results=50, fields=None, expand=None):
        """
        Search for issues using JQL (Jira Query Language).
        
        Args:
            jql: JQL search string
            start_at: Starting index for pagination
            max_results: Maximum number of results to return
            fields: Comma-separated string of field names to include
            expand: Comma-separated string of sections to expand
        """
        try:
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": max_results
            }
            
            if fields:
                params["fields"] = fields
            if expand:
                params["expand"] = expand
                
            logger.info(f"Searching issues with JQL: {jql}")
            response = requests.get(
                f"{self.api_url}/search",
                auth=self.auth,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            search_results = response.json()
            total = search_results.get("total", 0)
            results_count = len(search_results.get("issues", []))
            logger.info(f"Search returned {results_count} issues (total: {total})")
            return search_results
        except requests.RequestException as e:
            logger.error(f"Failed to search issues: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_all_issues(self, jql, fields=None, expand=None, max_results=1000):
        """
        Get all issues matching a JQL query, handling pagination.
        
        Args:
            jql: JQL search string
            fields: Comma-separated string of field names to include
            expand: Comma-separated string of sections to expand
            max_results: Maximum total number of results to return
        """
        all_issues = []
        start_at = 0
        page_size = 100  # Jira recommends 100 for optimal performance
        
        logger.info(f"Retrieving all issues matching JQL: {jql}")
        
        while True:
            try:
                search_results = self.search_issues(
                    jql=jql, 
                    start_at=start_at, 
                    max_results=page_size,
                    fields=fields,
                    expand=expand
                )
                
                if not search_results or not search_results.get("issues"):
                    break
                    
                issues = search_results.get("issues", [])
                all_issues.extend(issues)
                logger.info(f"Retrieved {len(issues)} issues (total: {len(all_issues)})")
                
                # Check if we've reached the total or our max limit
                if len(all_issues) >= min(search_results.get("total", 0), max_results):
                    break
                    
                # Move to next page
                start_at += len(issues)
                
                # If no issues were returned, we're done
                if len(issues) == 0:
                    break
            except Exception as e:
                logger.error(f"Error retrieving all issues: {str(e)}")
                break
                
        logger.info(f"Retrieved a total of {len(all_issues)} issues")
        return all_issues
    
    def get_issue_types(self):
        """Get all issue types defined in the Jira instance."""
        try:
            logger.info("Fetching issue types...")
            response = requests.get(
                f"{self.api_url}/issuetype",
                auth=self.auth,
                headers=self.headers
            )
            response.raise_for_status()
            issue_types = response.json()
            logger.info(f"Successfully retrieved {len(issue_types)} issue types")
            return issue_types
        except requests.RequestException as e:
            logger.error(f"Failed to get issue types: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_projects(self):
        """Get all projects visible to the authenticated user."""
        try:
            logger.info("Fetching projects...")
            response = requests.get(
                f"{self.api_url}/project",
                auth=self.auth,
                headers=self.headers
            )
            response.raise_for_status()
            projects = response.json()
            logger.info(f"Successfully retrieved {len(projects)} projects")
            return projects
        except requests.RequestException as e:
            logger.error(f"Failed to get projects: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def create_issue(self, project_key, issue_type, summary, description, fields=None):
        """
        Create a new issue.
        
        Args:
            project_key: The project key
            issue_type: The issue type name or ID
            summary: The issue summary
            description: The issue description
            fields: Dictionary of additional fields to set
        """
        # Base issue data
        issue_data = {
            "fields": {
                "project": {
                    "key": project_key
                },
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": description
                                }
                            ]
                        }
                    ]
                },
                "issuetype": {
                    "name": issue_type
                }
            }
        }
        
        # Add additional fields if provided
        if fields:
            issue_data["fields"].update(fields)
            
        try:
            logger.info(f"Creating issue in project {project_key} of type {issue_type}")
            response = requests.post(
                f"{self.api_url}/issue",
                auth=self.auth,
                headers=self.headers,
                json=issue_data
            )
            response.raise_for_status()
            new_issue = response.json()
            logger.info(f"Successfully created issue: {new_issue.get('key')}")
            return new_issue
        except requests.RequestException as e:
            logger.error(f"Failed to create issue: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_issue_content(self, issue_key):
        """
        Get the content of an issue in a format suitable for RAG.
        
        Args:
            issue_key: The issue key (e.g., DEMO-1)
        """
        try:
            # Get issue with all relevant fields
            fields_to_include = "summary,description,issuetype,status,created,updated,assignee,reporter,priority,labels,components,fixVersions,resolution,comment"
            issue = self.get_issue(issue_key, fields=fields_to_include)
            
            if not issue:
                return None
                
            fields = issue.get("fields", {})
            
            # Extract key metadata
            metadata = {
                "key": issue.get("key"),
                "id": issue.get("id"),
                "type": fields.get("issuetype", {}).get("name"),
                "status": fields.get("status", {}).get("name"),
                "created": fields.get("created"),
                "updated": fields.get("updated"),
                "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
                "labels": fields.get("labels", []),
                "resolution": fields.get("resolution", {}).get("name") if fields.get("resolution") else None,
                "url": f"{self.base_url}/browse/{issue.get('key')}"
            }
            
            # Extract people
            if fields.get("assignee"):
                metadata["assignee"] = fields.get("assignee", {}).get("displayName")
                
            if fields.get("reporter"):
                metadata["reporter"] = fields.get("reporter", {}).get("displayName")
            
            # Extract content fields
            content_parts = []
            
            # Add summary
            summary = fields.get("summary", "")
            if summary:
                content_parts.append(f"Summary: {summary}")
            
            # Add description
            # Note: this is simplified and assumes the description is in plain text
            # In reality, Jira descriptions can be in various formats
            description = fields.get("description")
            if description:
                if isinstance(description, dict):
                    # Try to extract text from Atlassian Document Format
                    desc_text = self._extract_text_from_adf(description)
                    if desc_text:
                        content_parts.append(f"Description: {desc_text}")
                else:
                    content_parts.append(f"Description: {description}")
            
            # Add comments
            comments = fields.get("comment", {}).get("comments", [])
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "Unknown")
                created = comment.get("created", "")
                
                comment_body = comment.get("body")
                if isinstance(comment_body, dict):
                    # Extract text from Atlassian Document Format
                    comment_text = self._extract_text_from_adf(comment_body)
                    if comment_text:
                        content_parts.append(f"Comment by {author} on {created}: {comment_text}")
                else:
                    content_parts.append(f"Comment by {author} on {created}: {comment_body}")
            
            # Combine all content
            full_content = "\n\n".join(content_parts)
            
            return {
                "metadata": metadata,
                "content": full_content
            }
        except Exception as e:
            logger.error(f"Error processing issue content: {str(e)}")
            return None
    
    def _extract_text_from_adf(self, adf_doc):
        """
        Extract plain text from Atlassian Document Format (ADF).
        This is a simplified version that doesn't handle all ADF features.
        
        Args:
            adf_doc: The ADF document object
        """
        if not adf_doc or not isinstance(adf_doc, dict):
            return ""
            
        text_parts = []
        
        def extract_from_content(content_list):
            parts = []
            if not content_list or not isinstance(content_list, list):
                return parts
                
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                    
                # Extract text nodes
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                
                # Extract text from content recursively
                if "content" in item and isinstance(item["content"], list):
                    parts.extend(extract_from_content(item["content"]))
            
            return parts
        
        # Extract text from the main content array
        if "content" in adf_doc and isinstance(adf_doc["content"], list):
            text_parts = extract_from_content(adf_doc["content"])
        
        return " ".join(text_parts)

# Script execution
if __name__ == "__main__":
    # Use environment variables or command line args for better security
    jira_url = os.environ.get("JIRA_URL", "https://cmegroup-sandbox-461.atlassian.net")
    username = os.environ.get("JIRA_USERNAME", "firstname.lastname@cmegroup.com")
    api_token = os.environ.get("JIRA_API_TOKEN", "your-api-token-here")
    
    # Create the client
    client = JiraClient(jira_url, username, api_token)
    
    # Test the connection
    if client.test_connection():
        print("\n✅ Connection to Jira successful!")
    else:
        print("\n❌ Failed to connect to Jira. Check log for details.")
        sys.exit(1)
    
    # Get projects
    projects = client.get_projects()
    if projects:
        print(f"\n✅ Successfully retrieved {len(projects)} projects")
        if len(projects) > 0:
            print("Sample projects:")
            for project in projects[:3]:  # Show top 3
                print(f"  - {project.get('name')} (Key: {project.get('key')})")
    else:
        print("\n❌ Failed to retrieve projects")
    
    # Get issue types
    issue_types = client.get_issue_types()
    if issue_types:
        print(f"\n✅ Successfully retrieved {len(issue_types)} issue types")
        if len(issue_types) > 0:
            print("Sample issue types:")
            for issue_type in issue_types[:5]:  # Show top 5
                print(f"  - {issue_type.get('name')}")
    else:
        print("\n❌ Failed to retrieve issue types")
    
    # Search for issues
    # Using a simple JQL query as an example
    jql = "project = DEMO ORDER BY created DESC"
    search_results = client.search_issues(jql, max_results=10)
    if search_results and "issues" in search_results:
        issues = search_results.get("issues", [])
        print(f"\n✅ Search returned {len(issues)} issues (total: {search_results.get('total', 0)})")
        if len(issues) > 0:
            print("Sample issues:")
            for issue in issues[:3]:  # Show top 3
                print(f"  - {issue.get('key')}: {issue.get('fields', {}).get('summary', 'No summary')}")
                
            # Get content for the first issue as an example
            first_issue_key = issues[0].get("key")
            issue_content = client.get_issue_content(first_issue_key)
            if issue_content:
                print(f"\n✅ Successfully retrieved content for issue {first_issue_key}")
                print(f"Summary: {issue_content.get('metadata', {}).get('summary', 'No summary')}")
                content_preview = issue_content.get('content', '')[:100].replace('\n', ' ').strip()
                print(f"Content preview: {content_preview}...")
            else:
                print(f"\n❌ Failed to retrieve content for issue {first_issue_key}")
    else:
        print(f"\n❌ Search failed or returned no results")
    
    print("\nJira integration test completed.")













import requests
import logging
import json
import os
import sys
from requests.auth import HTTPBasicAuth
import urllib3
from html.parser import HTMLParser

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ConfluenceClient")

class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data):
        self.text += data + " "

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://cmegroup.atlassian.net)
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, api_token)
        self.api_url = f"{self.base_url}/wiki/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized Confluence client for {self.base_url}")
        
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.api_url}/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=False
            )
            response.raise_for_status()
            logger.info("Connection successful!")
            return True
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return False
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand (e.g., "body.storage,version,space")
        """
        try:
            params = {}
            if expand:
                params["expand"] = expand
                
            logger.info(f"Fetching content with ID: {content_id}, expand: {expand}")
            response = requests.get(
                f"{self.api_url}/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            content = response.json()
            logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
            return content
        except requests.RequestException as e:
            logger.error(f"Failed to get content by ID: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
            
    def search_content(self, cql=None, title=None, content_type="page", 
                      expand=None, limit=10, start=0):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            content_type: Type of content to search for (default: page)
            expand: Properties to expand in results
            limit: Maximum number of results to return
            start: Starting index for pagination
        """
        params = {
            "limit": limit,
            "start": start
        }
        
        if cql:
            params["cql"] = cql
        else:
            # Build CQL if not provided
            query_parts = []
            
            if content_type:
                query_parts.append(f"type={content_type}")
                
            if title:
                # Escape special characters in title
                safe_title = title.replace('"', '\\"')
                query_parts.append(f'title~"{safe_title}"')
                
            if query_parts:
                params["cql"] = " AND ".join(query_parts)
        
        if expand:
            params["expand"] = expand
            
        try:
            logger.info(f"Searching for content with params: {params}")
            response = requests.get(
                f"{self.api_url}/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            results = response.json()
            logger.info(f"Search returned {len(results.get('results', []))} results")
            return results
        except requests.RequestException as e:
            logger.error(f"Failed to search content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_page_content(self, page_id):
        """
        Get the content of a page in a suitable format for RAG.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
        """
        try:
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
                
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki/pages/viewpage.action?pageId={page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process the HTML content
            html_filter = HTMLFilter()
            html_filter.feed(content)
            plain_text = html_filter.text
            
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content
            }
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None

    def get_all_content(self, content_type="page", limit=100, expand=None):
        """
        Retrieve all content of specified type with pagination handling.
        
        Args:
            content_type: Type of content to retrieve (default: page)
            limit: Maximum number of results per request
            expand: Properties to expand in results
        """
        all_content = []
        start = 0
        
        logger.info(f"Retrieving all {content_type} content")
        
        while True:
            try:
                params = {
                    "type": content_type,
                    "limit": limit,
                    "start": start
                }
                
                if expand:
                    params["expand"] = expand
                
                response = requests.get(
                    f"{self.api_url}/content",
                    auth=self.auth,
                    headers=self.headers,
                    params=params,
                    verify=False
                )
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    break
                    
                all_content.extend(results)
                logger.info(f"Retrieved {len(results)} {content_type}s (total: {len(all_content)})")
                
                # Check if there are more pages
                if len(results) < limit:
                    break
                    
                start += limit
            except requests.RequestException as e:
                logger.error(f"Error retrieving content: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Status code: {e.response.status_code}")
                    try:
                        error_details = e.response.json()
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    except:
                        logger.error(f"Response content: {e.response.text}")
                break
                
        logger.info(f"Retrieved a total of {len(all_content)} {content_type}s")
        return all_content
    
    def get_spaces(self, limit=25, start=0):
        """
        Get all spaces the user has access to.
        
        Args:
            limit: Maximum number of results per request
            start: Starting index for pagination
        """
        try:
            params = {
                "limit": limit,
                "start": start
            }
            
            logger.info("Fetching spaces...")
            response = requests.get(
                f"{self.api_url}/space",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            spaces = response.json()
            logger.info(f"Successfully retrieved {len(spaces.get('results', []))} spaces")
            return spaces
        except requests.RequestException as e:
            logger.error(f"Failed to get spaces: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_all_spaces(self):
        """
        Retrieve all spaces with pagination handling.
        """
        all_spaces = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
        logger.info("Retrieving all spaces")
        
        while True:
            spaces = self.get_spaces(limit=limit, start=start)
            if not spaces or not spaces.get("results"):
                break
                
            results = spaces.get("results", [])
            all_spaces.extend(results)
            logger.info(f"Retrieved {len(results)} spaces (total: {len(all_spaces)})")
            
            # Check if there are more spaces
            if len(results) < limit:
                break
                
            # Check the '_links' for a 'next' link
            links = spaces.get("_links", {})
            if not links.get("next"):
                break
                
            start += limit
                
        logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
        return all_spaces

# Script execution
if __name__ == "__main__":
    # Use environment variables or command line arguments for better security
    confluence_url = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
    username = os.environ.get("CONFLUENCE_USERNAME", "cli_api_user")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "your-api-token-here")
    
    # Create the client
    client = ConfluenceClient(confluence_url, username, api_token)
    
    # Test the connection
    if client.test_connection():
        print("\n✅ Connection to Confluence successful!")
    else:
        print("\n❌ Failed to connect to Confluence. Check log for details.")
        sys.exit(1)
    
    # Get all spaces (if access is available)
    try:
        spaces = client.get_all_spaces()
        if spaces:
            print(f"\n✅ Successfully retrieved {len(spaces)} spaces")
            if len(spaces) > 0:
                print("Sample spaces:")
                for space in spaces[:3]:  # Show top 3
                    print(f"  - {space.get('name')} (Key: {space.get('key')})")
        else:
            print("\n❌ Failed to retrieve spaces or no spaces available")
    except Exception as e:
        print(f"\n❌ Error retrieving spaces: {str(e)}")
    
    # Search for content
    search_title = "Test"
    search_results = client.search_content(title=search_title)
    if search_results and search_results.get("results"):
        print(f"\n✅ Search for '{search_title}' returned {len(search_results.get('results'))} results")
        for page in search_results.get("results")[:3]:  # Show top 3
            print(f"  - {page.get('title')} (ID: {page.get('id')})")
    else:
        print(f"\n❌ Search for '{search_title}' failed or returned no results")
    
    # Get all content
    try:
        all_pages = client.get_all_content(content_type="page", limit=25)
        if all_pages:
            print(f"\n✅ Retrieved {len(all_pages)} pages from Confluence")
            if len(all_pages) > 0:
                print("Sample pages:")
                for page in all_pages[:3]:  # Show top 3
                    print(f"  - {page.get('title')} (ID: {page.get('id')})")
        else:
            print("\n❌ Failed to retrieve pages or no pages available")
    except Exception as e:
        print(f"\n❌ Error retrieving pages: {str(e)}")
    
    print("\nConfluence integration test completed.")
