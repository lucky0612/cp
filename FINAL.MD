import os
import json
import logging
import sys
from datetime import datetime
import hashlib # For creating cache keys for JQL

import requests
from requests.auth import HTTPBasicAuth
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, HarmCategory, HarmBlockThreshold

# --- Configuration ---
# Jira Configuration (Update these with your details or use environment variables)
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://your-jira-instance.atlassian.net")
JIRA_USERNAME = os.environ.get("JIRA_USERNAME", "your-email@example.com")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "your_api_token")

# Vertex AI Configuration
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "your-gcp-project-id")
VERTEX_REGION = os.environ.get("VERTEX_REGION", "us-central1")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash-001") # Using flash for speed

# Cache Configuration
CACHE_FILE_PATH = "jira_data_cache.json"
CACHE_EXPIRY_SECONDS = 3600  # 1 hour, set to 0 to always use cache if available, or None to disable expiry

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_gemini_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("JiraGeminiChatbot")

# --- JIRA Client Class (Modified from provided PDF) ---
class JiraClient:
    def __init__(self, base_url, username, api_token, verify_ssl=False, ca_bundle=None, cache_file=CACHE_FILE_PATH, cache_expiry=CACHE_EXPIRY_SECONDS): # [cite: 1]
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.cache_file = cache_file
        self.cache_expiry = cache_expiry
        self.cache = self._load_cache()

        if verify_ssl is False: # [cite: 1]
            logger.warning("SSL certificate verification is disabled. This is not recommended for production use.") # [cite: 1]
            self.ssl_verify = False # [cite: 1]
        elif isinstance(verify_ssl, str): # ca_bundle path [cite: 1]
            if not os.path.exists(verify_ssl): # [cite: 1]
                logger.warning(f"Custom CA bundle specified but not found: {verify_ssl}. Falling back to default verification.") # [cite: 1]
                self.ssl_verify = True # Default: certifi certificates [cite: 1]
            else:
                logger.info(f"Using custom CA bundle: {verify_ssl}") # [cite: 1]
                self.ssl_verify = verify_ssl # [cite: 1]
        else:
            self.ssl_verify = True # Default: westycertificates (actually uses certifi by default with True) [cite: 1]
            logger.info("Using default SSL certificate verification.")


    def _load_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Check cache expiry
                    if self.cache_expiry is not None and self.cache_expiry > 0:
                        last_updated_str = cache_data.get("metadata", {}).get("last_updated")
                        if last_updated_str:
                            last_updated_dt = datetime.fromisoformat(last_updated_str)
                            if (datetime.now() - last_updated_dt).total_seconds() > self.cache_expiry:
                                logger.info(f"Cache expired. Clearing old cache.")
                                return {"issues": {}, "projects": None, "issue_types": None, "jql_results": {}, "metadata": {}}
                    logger.info(f"Loaded data from cache: {self.cache_file}")
                    return cache_data
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from cache file: {self.cache_file}. Starting with an empty cache.")
            except Exception as e:
                logger.error(f"Error loading cache: {e}. Starting with an empty cache.")
        return {"issues": {}, "projects": None, "issue_types": None, "jql_results": {}, "metadata": {}}

    def _save_cache(self):
        if self.cache_file:
            try:
                self.cache["metadata"]["last_updated"] = datetime.now().isoformat()
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                logger.debug(f"Saved data to cache: {self.cache_file}")
            except Exception as e:
                logger.error(f"Error saving cache: {e}")

    def test_connection(self): # [cite: 1]
        """Test the connection to Jira and authentication."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/myself", # A lightweight endpoint
                headers=self.headers,
                auth=self.auth,
                verify=self.ssl_verify
            )
            response.raise_for_status()
            logger.info("Successfully connected to Jira.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira connection test failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            return False

    def get_issue(self, issue_key, expand=None): # [cite: 3]
        """Retrieve a specific issue by its key."""
        cache_item_key = f"issue_{issue_key}_{expand}"
        if cache_item_key in self.cache["issues"]:
            logger.info(f"Returning cached issue: {issue_key}")
            return self.cache["issues"][cache_item_key]

        logger.info(f"Fetching issue: {issue_key}") # [cite: 3]
        params = {}
        if expand:
            params["expand"] = expand # [cite: 3]

        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/issue/{issue_key}",
                headers=self.headers,
                auth=self.auth,
                params=params,
                verify=self.ssl_verify
            )
            response.raise_for_status() # [cite: 3]
            issue_data = response.json() # [cite: 3]
            self.cache["issues"][cache_item_key] = issue_data
            self._save_cache()
            logger.info(f"Successfully retrieved issue: {issue_key}") # [cite: 3]
            return issue_data
        except requests.exceptions.RequestException as e: # [cite: 3]
            logger.error(f"Failed to get issue {issue_key}: {e}") # [cite: 3]
            if hasattr(e, 'response') and e.response is not None: # [cite: 3]
                logger.error(f"Status code: {e.response.status_code}") # [cite: 3]
                try:
                    error_details = e.response.json() # [cite: 3]
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}") # [cite: 3]
                except json.JSONDecodeError:
                    logger.error(f"Response content: {e.response.text}") # [cite: 3]
            return None

    def search_issues_by_jql(self, jql_query, start_at=0, max_results=50, fields=None, expand=None): # [cite: 4]
        """Search issues using JQL."""
        logger.info(f"Searching issues with JQL: {jql_query}") # [cite: 4]
        params = {
            "jql": jql_query,
            "startAt": start_at, # [cite: 4]
            "maxResults": max_results
        }
        if fields:
            params["fields"] = fields # [cite: 4]
        if expand:
            params["expand"] = expand # [cite: 4]

        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/search", # [cite: 4]
                headers=self.headers,
                auth=self.auth,
                params=params,
                verify=self.ssl_verify
            )
            response.raise_for_status() # [cite: 4]
            search_results = response.json() # [cite: 4]
            
            # Cache individual issues from search results
            if "issues" in search_results:
                for issue_data in search_results["issues"]:
                    issue_key = issue_data.get("key")
                    if issue_key:
                        # A simplified cache key for issues from search; specific 'get_issue' might have more expand options
                        cache_item_key = f"issue_{issue_key}_search" 
                        self.cache["issues"][cache_item_key] = issue_data
                self._save_cache()

            logger.info(f"Search returned {len(search_results.get('issues', []))} issues (total: {search_results.get('total', 0)})") # [cite: 4]
            return search_results
        except requests.exceptions.RequestException as e: # [cite: 4]
            logger.error(f"Failed to search issues with JQL '{jql_query}': {e}") # [cite: 4]
            if hasattr(e, 'response') and e.response is not None: # [cite: 4]
                logger.error(f"Status code: {e.response.status_code}") # [cite: 4]
            # Further error detail logging as in get_issue
            return None

    def get_all_issues(self, jql_query="ORDER BY created DESC", fields=None, expand="renderedFields", max_results_total=None): # [cite: 7]
        """Retrieve all issues matching a JQL query, handling pagination."""
        jql_hash = hashlib.md5(jql_query.encode()).hexdigest()
        cache_key = f"jql_{jql_hash}_{fields}_{expand}_{max_results_total}"

        if cache_key in self.cache["jql_results"]:
            logger.info(f"Returning all issues for JQL '{jql_query}' from cache.")
            # Still update individual issue caches from this master list for consistency
            cached_result = self.cache["jql_results"][cache_key]
            for issue_data in cached_result:
                 if "key" in issue_data:
                    self.cache["issues"][f"issue_{issue_data['key']}_all"] = issue_data # Simplified key
            self._save_cache()
            return cached_result

        all_issues = []
        start_at = 0
        page_size = 50  # Standard page size

        logger.info(f"Fetching all issues for JQL: {jql_query}")
        while True:
            logger.info(f"Fetching issues page: startAt={start_at}, pageSize={page_size}")
            search_results = self.search_issues_by_jql(jql_query, start_at, page_size, fields, expand)

            if not search_results or not search_results.get("issues"): # [cite: 7]
                logger.info("No more issues found or error in search.")
                break
            
            current_issues = search_results["issues"]
            all_issues.extend(current_issues)

            # Cache individual issues
            for issue_data in current_issues:
                if "key" in issue_data:
                    self.cache["issues"][f"issue_{issue_data['key']}_all"] = issue_data # Simplified key

            if max_results_total is not None and len(all_issues) >= max_results_total:
                logger.info(f"Reached max_results_total of {max_results_total}. Returning {len(all_issues)} issues.")
                all_issues = all_issues[:max_results_total]
                break
            
            total_available = search_results.get("total", 0)
            if start_at + len(current_issues) >= total_available: # Check if we've fetched all available issues [cite: 7]
                logger.info("Fetched all available issues based on search total.")
                break
            
            start_at += len(current_issues) # [cite: 7]
            if not current_issues: # Safety break if a page returns empty for some reason
                logger.info("Current page returned no issues, stopping.")
                break
        
        logger.info(f"Retrieved a total of {len(all_issues)} issues for JQL: {jql_query}") # [cite: 7]
        self.cache["jql_results"][cache_key] = all_issues
        self._save_cache()
        return all_issues

    def get_issue_types(self): # [cite: 7]
        """Get all issue types defined in the Jira instance."""
        if self.cache["issue_types"] is not None:
            logger.info("Returning cached issue types.")
            return self.cache["issue_types"]

        logger.info("Fetching issue types...") # [cite: 7]
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/issuetype", # [cite: 8]
                headers=self.headers,
                auth=self.auth,
                verify=self.ssl_verify
            )
            response.raise_for_status()
            issue_types = response.json() # [cite: 8]
            self.cache["issue_types"] = issue_types
            self._save_cache()
            logger.info(f"Successfully retrieved {len(issue_types)} issue types.") # [cite: 8]
            return issue_types
        except requests.exceptions.RequestException as e: # [cite: 8]
            logger.error(f"Failed to get issue types: {e}") # [cite: 8]
            # Further error detail logging as in get_issue
            return []

    def get_projects(self): # [cite: 8]
        """Get all projects visible to the authenticated user."""
        if self.cache["projects"] is not None:
            logger.info("Returning cached projects.")
            return self.cache["projects"]

        logger.info("Fetching projects...") # [cite: 8]
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/project",
                headers=self.headers,
                auth=self.auth,
                verify=self.ssl_verify
            )
            response.raise_for_status() # [cite: 8]
            projects = response.json() # [cite: 8]
            self.cache["projects"] = projects
            self._save_cache()
            logger.info(f"Successfully retrieved {len(projects)} projects.") # [cite: 8]
            return projects
        except requests.exceptions.RequestException as e: # [cite: 8]
            logger.error(f"Failed to get projects: {e}") # [cite: 9]
            # Further error detail logging as in get_issue
            return []

    def _extract_text_from_adf_node(self, node): # Based on [cite: 13]
        """Recursively extract text from a single ADF node."""
        text_parts = []
        if not isinstance(node, dict):
            return ""

        node_type = node.get("type")
        if node_type == "text":
            text_parts.append(node.get("text", ""))

        if "content" in node and isinstance(node["content"], list):
            for child_node in node["content"]:
                text_parts.append(self._extract_text_from_adf_node(child_node))
        
        # Basic handling for table content - extract text from cells
        if node_type in ["table", "tableRow", "tableHeader", "tableCell"]:
             if "content" in node and isinstance(node["content"], list):
                for content_node in node["content"]: # Iterate over rows or cells
                    text_parts.append(self._extract_text_from_adf_node(content_node))
        
        # Add more ADF node type handling here as needed (e.g., lists, blockquotes)
        if node_type == "listItem":
            text_parts.insert(0, "- ") # Add bullet for list items
        if node_type == "paragraph" and "".join(text_parts).strip():
             text_parts.append("\n")


        return "".join(text_parts)

    def extract_text_from_adf(self, adf_doc): # [cite: 13]
        """Extract plain text from Atlassian Document Format (ADF)."""
        if not isinstance(adf_doc, dict) or "content" not in adf_doc: # [cite: 13]
            # If it's not a standard ADF doc, or already plain text, return as is.
            if isinstance(adf_doc, str):
                return adf_doc
            return ""
        
        full_text_parts = []
        if isinstance(adf_doc.get("content"), list): # [cite: 13]
            for node in adf_doc["content"]:
                full_text_parts.append(self._extract_text_from_adf_node(node))
        
        return "".join(full_text_parts).strip()


    def get_issue_content_for_llm(self, issue_key_or_data, fields_to_include=None): # [cite: 11]
        """Prepare issue content in a string format suitable for an LLM prompt."""
        if isinstance(issue_key_or_data, str):
            issue_data = self.get_issue(issue_key_or_data, expand="renderedFields,changelog,comment")
        else:
            issue_data = issue_key_or_data # Assumes already fetched issue data object

        if not issue_data:
            return f"Could not retrieve data for issue: {issue_key_or_data if isinstance(issue_key_or_data, str) else 'provided data object'}."

        content_parts = []
        key = issue_data.get("key")
        fields = issue_data.get("fields", {})
        rendered_fields = issue_data.get("renderedFields", {}) # For fields like description, comments with ADF

        content_parts.append(f"Issue Key: {key}")
        content_parts.append(f"Link: {self.base_url}/browse/{key}")

        default_fields = [ # [cite: 11]
            "summary", "status", "assignee", "reporter", "priority", 
            "issuetype", "created", "updated", "labels", "components",
            "description", "comment" # "comment" is handled separately via renderedFields or direct comment API
        ]
        if fields_to_include is None:
            fields_to_include = default_fields

        for field_name in fields_to_include:
            value = None
            raw_value = fields.get(field_name)

            if field_name == "description":
                # Use renderedFields for ADF description, fallback to raw if not present
                value_adf = rendered_fields.get("description") if rendered_fields else None
                if value_adf: # ADF format
                    value = self.extract_text_from_adf(value_adf) # [cite: 12]
                elif isinstance(raw_value, str): # Plain text description
                    value = raw_value
                elif isinstance(raw_value, dict): # Potentially already an ADF doc if renderedFields not used
                     value = self.extract_text_from_adf(raw_value)


            elif field_name == "comment":
                # Comments are complex. renderedFields might have them, or a separate comments object.
                comments_obj = rendered_fields.get("comment") or fields.get("comment")
                if comments_obj and "comments" in comments_obj:
                    comment_texts = []
                    for comment in comments_obj["comments"]:
                        author = comment.get("author", {}).get("displayName", "Unknown author") # [cite: 12]
                        created_date = comment.get("created", "") # [cite: 12]
                        body_adf = comment.get("body") # body can be ADF [cite: 12]
                        
                        comment_text = ""
                        if isinstance(body_adf, str): # Plain text comment
                            comment_text = body_adf
                        elif isinstance(body_adf, dict): # ADF comment
                            comment_text = self.extract_text_from_adf(body_adf) # [cite: 13]
                        
                        if comment_text.strip():
                             comment_texts.append(f"Comment by {author} on {created_date}:\n{comment_text.strip()}")
                    if comment_texts:
                        value = "\n---\n".join(comment_texts)
                else: # Fallback if comments are not in expected structure
                    value = "No comments found or comments are not in expected format."


            elif raw_value:
                if isinstance(raw_value, dict):
                    # Handle complex fields like status, assignee, reporter, priority, issuetype
                    if "name" in raw_value:
                        value = raw_value["name"]
                    elif "displayName" in raw_value: # for assignee, reporter
                        value = raw_value["displayName"]
                    elif "value" in raw_value: # for some custom fields
                        value = raw_value["value"]
                    else: # fallback to string representation
                        value = str(raw_value)
                elif isinstance(raw_value, list):
                    # Handle list fields like labels, components
                    if all(isinstance(item, str) for item in raw_value):
                        value = ", ".join(raw_value)
                    else: # List of objects (e.g., components)
                        value_parts = []
                        for item in raw_value:
                            if isinstance(item, dict) and "name" in item:
                                value_parts.append(item["name"])
                            else:
                                value_parts.append(str(item))
                        value = ", ".join(value_parts)
                else: # Simple field
                    value = str(raw_value)
            
            if value is not None and str(value).strip(): # Only add if there's content
                 content_parts.append(f"{field_name.replace('_', ' ').capitalize()}: {str(value).strip()}")


        return "\n".join(content_parts)


# --- Gemini Chatbot Class ---
class JiraGeminiChatbot:
    def __init__(self, jira_client, project_id, region, model_name=GEMINI_MODEL_NAME): # [cite: 15, 17]
        self.jira_client = jira_client
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        
        try:
            vertexai.init(project=self.project_id, location=self.region) # [cite: 17]
            self.model = GenerativeModel(self.model_name) # [cite: 17]
            logger.info(f"Vertex AI initialized. Using model: {self.model_name}") # [cite: 17]
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI or load model: {e}")
            raise

        # Pre-load all Jira issues into a simple text corpus for context
        # For very large instances, consider more selective loading or on-demand fetching
        logger.info("Pre-loading Jira issues for context. This may take a while for large instances...")
        # You can customize the JQL here, e.g., to get recently updated tickets, or from specific projects
        all_raw_issues = self.jira_client.get_all_issues(jql_query="ORDER BY updated DESC", max_results_total=200) # Limiting for performance
        
        self.jira_context_corpus = []
        if all_raw_issues:
            for issue_data in all_raw_issues:
                content = self.jira_client.get_issue_content_for_llm(issue_data)
                if content:
                    self.jira_context_corpus.append(content)
            logger.info(f"Loaded {len(self.jira_context_corpus)} issues into context corpus.")
        else:
            logger.warning("No Jira issues loaded into the context corpus.")


    def _get_relevant_jira_context(self, user_query, max_len=1500000): # Max context window for Gemini 1.5 Flash can be up to 2M tokens
        """
        A simple relevance filter. For production, consider semantic search or embeddings.
        This implementation will concatenate all pre-loaded issues.
        Be mindful of the model's context window limits.
        """
        if not self.jira_context_corpus:
            return "No Jira data available in the local corpus."

        # Simple concatenation for now. Could be improved with keyword matching etc.
        full_context = "\n\n---\n\n".join(self.jira_context_corpus)
        
        # Truncate if too long (very basic truncation)
        if len(full_context) > max_len:
            logger.warning(f"Combined Jira context is too long ({len(full_context)} chars), truncating to {max_len} chars.")
            full_context = full_context[:max_len] 
        return full_context


    def ask(self, user_query):
        logger.info(f"User query: {user_query}")

        # Attempt to get relevant Jira context
        # This is a simplified approach; more advanced would involve RAG (Retrieval Augmented Generation)
        # or letting Gemini itself formulate JQL if it has tool-use capabilities enabled.
        jira_context_str = self._get_relevant_jira_context(user_query)

        # Safety settings - adjust as needed
        safety_settings = { # [cite: 26, 33]
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        # Configure generation parameters [cite: 17, 18, 20, 21, 28, 33]
        generation_config = GenerationConfig(
            temperature=0.7, # [cite: 17, 18, 28]
            top_p=0.95, # [cite: 17, 18, 28]
            max_output_tokens=8192, # [cite: 17, 21, 28]
        )
        
        # Construct the prompt
        # Note: Direct image data is not passed here. The model relies on text descriptions from Jira.
        prompt_template = f"""
You are an advanced AI assistant integrated with a Jira instance. Your name is "JiraHelperBot".
Your primary goal is to answer questions based on the provided Jira ticket information and your general knowledge.
Be professional, friendly, and accurate.

**Instructions for Responding:**
1.  **Analyze the Question:** Understand if the user is asking about specific Jira tickets, general Jira concepts, or something outside of Jira.
2.  **Use Provided Jira Data:** If the question relates to Jira tickets, prioritize information from the "JIRA DATA CONTEXT" section below. This data contains extracted text from various Jira issues, including summaries, descriptions (which may include tables), statuses, assignees, comments, etc.
3.  **Cite Sources:** When using information from a specific Jira ticket, ALWAYS mention the Jira issue key (e.g., "According to ticket XYZ-123...") and provide its direct link: {self.jira_client.base_url}/browse/{{ISSUE_KEY}}.
4.  **Handle Ambiguity:** If the user's question is vague or could refer to multiple tickets, ask clarifying questions to narrow down their request. For example, if they ask "What's the progress on the UI task?", you could ask "Could you specify which UI task you're referring to by its ID or a keyword from its summary?"
5.  **Formatting:** Present answers clearly. Use paragraphs for explanations, bullet points for lists, and try to format information like tables if the query implies it (e.g., "Can you list all open bugs and their assignees?").
6.  **Image/Table Mentions:** The Jira data is text-based. If it mentions images (e.g., by filename or alt-text within a description), you can acknowledge their mention if relevant but state that you cannot directly display or analyze the visual content of images. You CAN interpret textual data from tables if it's extracted into the context.
7.  **Irrelevant Questions:** If the question is unrelated to Jira or general knowledge you can confidently answer, politely inform the user. For example: "I am JiraHelperBot, designed to assist with Jira and provide general information. I'm unable to help with [topic of irrelevant question]. How else can I assist you with Jira?"
8.  **No Data:** If the "JIRA DATA CONTEXT" is empty or doesn't contain relevant information for a Jira-specific query, state that you couldn't find the information in the loaded Jira data and suggest they refine their query or check Jira directly.
9.  **Be Concise but Thorough:** Provide complete answers without being overly verbose.
10. **Instant Replies:** Try to process and respond quickly.

--- JIRA DATA CONTEXT START ---
{jira_context_str}
--- JIRA DATA CONTEXT END ---

User Question: "{user_query}"

Your Answer:
        """
        
        logger.info(f"Sending prompt to Gemini (length: {len(prompt_template)} chars). First 500 chars of context: {jira_context_str[:500]}")

        try:
            response_stream = self.model.generate_content(
                [prompt_template],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True # [cite: 17, 21]
            )
            
            full_response_text = ""
            for chunk in response_stream: # [cite: 17, 21]
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                     part_text = chunk.candidates[0].content.parts[0].text
                     print(part_text, end="", flush=True) # Stream to console
                     full_response_text += part_text
            print() # Newline after streaming

            logger.info(f"Gemini response length: {len(full_response_text)} characters")
            return full_response_text

        except Exception as e: # [cite: 17, 21]
            logger.error(f"Error generating response from Gemini: {str(e)}")
            return f"Error: Could not get a response from the AI model. Details: {str(e)}"


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Jira Gemini Chatbot application...")

    # Validate essential configurations
    if not JIRA_BASE_URL or "your-jira-instance" in JIRA_BASE_URL or \
       not JIRA_USERNAME or "your-email" in JIRA_USERNAME or \
       not JIRA_API_TOKEN or "your_api_token" in JIRA_API_TOKEN:
        logger.error("Jira credentials (JIRA_BASE_URL, JIRA_USERNAME, JIRA_API_TOKEN) are not properly configured.")
        logger.error("Please set them as environment variables or directly in the script.")
        sys.exit(1)

    if not VERTEX_PROJECT_ID or "your-gcp-project" in VERTEX_PROJECT_ID:
        logger.error("Vertex AI Project ID (VERTEX_PROJECT_ID) is not properly configured.")
        sys.exit(1)
    
    # Initialize Jira Client
    try:
        jira_client = JiraClient(JIRA_BASE_URL, JIRA_USERNAME, JIRA_API_TOKEN, verify_ssl=False) # SSL verify set to False [cite: 1]
        if not jira_client.test_connection():
            logger.error("Exiting due to Jira connection failure.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize JiraClient: {e}")
        sys.exit(1)

    # Initialize Chatbot
    try:
        chatbot = JiraGeminiChatbot(jira_client, VERTEX_PROJECT_ID, VERTEX_REGION, GEMINI_MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to initialize JiraGeminiChatbot: {e}")
        sys.exit(1)

    print("\n==== Jira Gemini Chatbot ====")
    print("Ask me anything about your Jira projects or general topics!")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue

            print("\nJiraHelperBot: ", end="")
            response = chatbot.ask(user_input)
            # Response is already streamed to console, so just a newline if needed.
            # if response: print(response) 

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}")
            print("An error occurred. Please check the logs.")

    logger.info("Jira Gemini Chatbot application finished.")


















#!/usr/bin/env python3
import requests
import logging
import os
import sys
import json
import re
import time
import hashlib
import pickle
import threading
from datetime import datetime, timedelta
from urllib.parse import quote
from collections import Counter, defaultdict
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_gemini.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ConfluenceGemini")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
CACHE_DIR = os.environ.get("CACHE_DIR", "./confluence_cache")
CACHE_EXPIRY_DAYS = int(os.environ.get("CACHE_EXPIRY_DAYS", "7"))
MAX_RESULTS_PER_QUERY = int(os.environ.get("MAX_RESULTS_PER_QUERY", "50"))

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

class HTMLFilter:
    def __init__(self):
        self.text = ""
    
    def feed(self, data):
        self.text = re.sub(r'<[^>]*>', ' ', data)
        self.text = re.sub(r'\s+', ' ', self.text).strip()

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://company.atlassian.net)
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=False  # Setting SSL verify to False as requested
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (first 500 chars): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received during connection test")
                return True  # Still consider it a success if status code is OK
            
            try:
                response.json()
                logger.info("Connection successful!")
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Response content: {raw_content}")
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Connection successful but received invalid JSON: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return False
        except Exception as e:
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
            
            logger.info(f"Fetching content with ID: {content_id}")
            response = self.session.get(
                f"{self.base_url}/rest/api/content/{content_id}",
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
                logger.warning("Empty response received when retrieving content")
                return None
            
            try:
                content = response.json()
                logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
                return content
            except json.JSONDecodeError as e:
                logger.error(f"Response content: {raw_content}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for content ID {content_id}: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return None
        except Exception as e:
            logger.error(f"Connection successful but received invalid JSON: {str(e)}")
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
        Get the content of a page in a suitable format for NLP.
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
                "url": f"{self.base_url}/wiki/spaces/{page.get('_expandable', {}).get('space', '').split('/')[-1]}/pages/{page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process HTML content to plain text
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
        limit = 25  # Confluence API commonly uses 25 as default
        
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
                
                response = self.session.get(
                    f"{self.base_url}/rest/api/content",
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
                    logger.warning("Empty response received when retrieving all content")
                    break
                
                try:
                    data = response.json()
                    results = data.get("results", [])
                    if not results:
                        break
                    
                    all_content.extend(results)
                    logger.info(f"Retrieved {len(results)} {content_type} content (total: {len(all_content)})")
                    
                    # Check if there are more pages
                    if len(results) < limit:
                        break
                    
                    start += limit
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {content_type}: {str(e)}")
                    logger.error(f"Response content: {raw_content}")
                    break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON when retrieving content: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                break
            except Exception as e:
                logger.error(f"Error retrieving content: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_details = e.response.json()
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    except:
                        logger.error(f"Response content: {e.response.text}")
                break
        
        logger.info(f"Retrieved a total of {len(all_content)} {content_type}")
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
            response = self.session.get(
                f"{self.base_url}/rest/api/space",
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
                logger.error(f"Response content: {raw_content}")
                return {"results": []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for spaces: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return {"results": []}
        except Exception as e:
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
        """Retrieve all spaces with pagination handling."""
        all_spaces = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
        logger.info("Retrieving all spaces")
        
        while True:
            spaces = self.get_spaces(limit=limit, start=start)
            results = spaces.get("results", [])
            if not results:
                break
            
            all_spaces.extend(results)
            logger.info(f"Retrieved {len(results)} spaces (total: {len(all_spaces)})")
            
            # Check if there are more spaces
            if len(results) < limit:
                break
            
            start += limit
        
        logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
        return all_spaces
    
    def search_content(self, cql=None, title=None, content_type="page", expand=None, limit=10, start=0):
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
        try:
            params = {}
            
            # Build CQL if not provided
            query_parts = []
            if content_type:
                query_parts.append(f"type={content_type}")
            if title:
                # Fix for special characters in title
                safe_title = title.replace('"', '\\"')
                query_parts.append(f'title~"{safe_title}"')
            
            if query_parts:
                params["cql"] = " AND ".join(query_parts)
            
            # Override with explicit CQL if provided
            if cql:
                params["cql"] = cql
            
            if expand:
                params["expand"] = expand
            
            params["limit"] = limit
            params["start"] = start
            
            logger.info(f"Searching for content with params: {params}")
            response = self.session.get(
                f"{self.base_url}/rest/api/content/search",
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
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for search: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return {"results": []}
        except Exception as e:
            logger.error(f"Failed to search content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return {"results": []}

class TextProcessor:
    """Class for advanced text processing without relying on external NLP libraries."""
    
    def __init__(self):
        """Initialize text processing components."""
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
            'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
            'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
        }
        
        # Add domain-specific stop words if needed
        self.domain_stop_words = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'doing'}
        self.stop_words.update(self.domain_stop_words)
        
        logger.info("Initialized text processor")
    
    def tokenize_text(self, text):
        """Simple tokenization function to split text into words."""
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split by whitespace and filter out empty strings
        tokens = [token for token in text.split() if token]
        
        return tokens
    
    def sentence_tokenize(self, text):
        """Simple sentence tokenization."""
        if not text or not isinstance(text, str):
            return []
        
        # Split by common sentence delimiters
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def simple_lemmatize(self, word):
        """Very simple lemmatization for common English words."""
        # This is a simplified version and won't handle all cases
        if word.endswith('ing'):
            return word[:-3]  # running -> run
        elif word.endswith('ed') and len(word) > 3:
            return word[:-2]  # walked -> walk
        elif word.endswith('s') and not word.endswith('ss'):
            return word[:-1]  # cats -> cat
        elif word.endswith('ies'):
            return word[:-3] + 'y'  # cities -> city
        elif word.endswith('es'):
            return word[:-2]  # boxes -> box
        return word
    
    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing, removing stop words, and simple lemmatization.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing original, tokens, filtered_tokens, lemmatized_tokens, and segments
        """
        if not text or not isinstance(text, str):
            return {
                "original": "",
                "tokens": [],
                "filtered_tokens": [],
                "lemmatized_tokens": [],
                "segments": []
            }
        
        # Segment text into sentences
        segments = self.sentence_tokenize(text)
        
        # Tokenize
        tokens = self.tokenize_text(text)
        
        # Remove stop words and non-alphanumeric tokens
        filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Lemmatize tokens
        lemmatized_tokens = [self.simple_lemmatize(token) for token in filtered_tokens]
        
        return {
            "original": text,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "segments": segments
        }
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract the most important keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        processed = self.preprocess_text(text)
        word_freq = Counter(processed["lemmatized_tokens"])
        return [word for word, freq in word_freq.most_common(top_n)]
    
    def segment_text(self, text, max_segment_length=500):
        """
        Segment text into smaller chunks while respecting sentence boundaries.
        
        Args:
            text: Input text to segment
            max_segment_length: Maximum character length for each segment
            
        Returns:
            List of text segments
        """
        sentences = self.sentence_tokenize(text)
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= max_segment_length:
                current_segment += " " + sentence if current_segment else sentence
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def preprocess_question(self, question):
        """
        Preprocess a question to extract key components and intent.
        
        Args:
            question: User question
            
        Returns:
            Dict with processed question information
        """
        processed = self.preprocess_text(question)
        
        # Detect question type based on first word
        question_words = {"what", "who", "where", "when", "why", "how", "which", "can", "do", "is", "are", "will"}
        first_word = processed["tokens"][0].lower() if processed["tokens"] else ""
        
        question_type = "unknown"
        if first_word in question_words:
            question_type = first_word
        
        # Extract keywords
        keywords = self.extract_keywords(question, top_n=5)
        
        # Detect if it's a multi-part question by looking for specific patterns
        multi_part_patterns = [
            r'\d+\s*\.\s+',  # Numbered list (1. 2. etc)
            r'first.*?second',
            r'part\s+\d+',
            r'multiple questions',
            r'several questions',
            r'and also',
            r';',
            r'\?.*?\?'  # Multiple question marks
        ]
        
        is_multi_part = any(re.search(pattern, question, re.IGNORECASE) for pattern in multi_part_patterns)
        
        return {
            "processed": processed,
            "type": question_type,
            "keywords": keywords,
            "is_multi_part": is_multi_part
        }
    
    def split_multi_part_question(self, question):
        """
        Split a multi-part question into individual questions.
        
        Args:
            question: Multi-part question
            
        Returns:
            List of individual questions
        """
        # Method 1: Split by question marks followed by space or sentence start
        parts = re.split(r'\?\s+', question)
        
        # Make sure each part ends with a question mark
        for i, part in enumerate(parts):
            if i < len(parts) - 1 and not part.endswith('?'):
                parts[i] = part + '?'
        
        # Filter out empty parts and strip whitespace
        parts = [part.strip() for part in parts if part.strip()]
        
        # If splitting by question marks didn't work well, try numbering patterns
        if len(parts) <= 1:
            # Match numbered patterns like "1. First question 2. Second question"
            numbered_parts = re.split(r'\d+\s*\.\s+', question)
            
            # Remove empty parts and strip whitespace
            numbered_parts = [part.strip() for part in numbered_parts if part.strip()]
            
            if len(numbered_parts) > 1:
                parts = numbered_parts
        
        # If we still don't have multiple parts, try other delimiters
        if len(parts) <= 1:
            other_delimiters = [';', 'and also,', 'additionally,', 'moreover,', 'furthermore,']
            for delimiter in other_delimiters:
                if delimiter in question.lower():
                    parts = question.split(delimiter)
                    parts = [part.strip() for part in parts if part.strip()]
                    break
        
        # If still no success, default to the original question
        if len(parts) <= 1:
            return [question]
        
        return parts

class ConfluenceCache:
    """Handles caching of Confluence content to avoid repeated API calls."""
    
    def __init__(self, cache_dir=CACHE_DIR, expiry_days=CACHE_EXPIRY_DAYS):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            expiry_days: Number of days after which cache should be considered stale
        """
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        self.memory_cache = {}
        self.content_index = defaultdict(list)
        self.metadata = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing index if available
        self.load_index()
        
        logger.info(f"Initialized Confluence cache in {self.cache_dir} with {expiry_days} day expiry")
    
    def get_cache_path(self, key):
        """Get the file path for a cache item."""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pickle")
    
    def is_cache_valid(self, key):
        """Check if a cache item exists and is still valid (not expired)."""
        cache_path = self.get_cache_path(key)
        
        if key in self.memory_cache:
            return True
        
        if os.path.exists(cache_path):
            # Check if the cache file is recent enough
            modification_time = os.path.getmtime(cache_path)
            modification_date = datetime.fromtimestamp(modification_time)
            expiry_date = datetime.now() - timedelta(days=self.expiry_days)
            
            return modification_date > expiry_date
        
        return False
    
    def get(self, key):
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.is_cache_valid(key):
            return None
        
        # Try memory cache first
        if key in self.memory_cache:
            logger.info(f"Cache hit (memory): {key}")
            return self.memory_cache[key]
        
        # Then try file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
                # Store in memory cache for faster future access
                self.memory_cache[key] = value
                logger.info(f"Cache hit (file): {key}")
                return value
        except Exception as e:
            logger.error(f"Error reading cache for {key}: {str(e)}")
            return None
    
    def set(self, key, value):
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Store in memory cache
        self.memory_cache[key] = value
        
        # Store in file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.info(f"Cached: {key}")
        except Exception as e:
            logger.error(f"Error writing cache for {key}: {str(e)}")
    
    def clear(self, key=None):
        """
        Clear cache items.
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key:
            # Clear specific key
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            cache_path = self.get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            logger.info(f"Cleared cache for {key}")
        else:
            # Clear all cache
            self.memory_cache = {}
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pickle'):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("Cleared all cache")
    
    def build_index(self, content_list):
        """
        Build a searchable index from content.
        
        Args:
            content_list: List of content items to index
        """
        text_processor = TextProcessor()
        
        for content in content_list:
            if not content or not isinstance(content, dict):
                continue
            
            content_id = content.get("metadata", {}).get("id")
            if not content_id:
                continue
            
            # Store metadata
            self.metadata[content_id] = content.get("metadata", {})
            
            # Process text
            text = content.get("content", "")
            processed = text_processor.preprocess_text(text)
            
            # Index by lemmatized tokens
            for token in processed["lemmatized_tokens"]:
                self.content_index[token].append(content_id)
        
        logger.info(f"Built index with {len(self.content_index)} terms and {len(self.metadata)} documents")
        
        # Save the index
        self.save_index()
    
    def search_index(self, query, max_results=10):
        """
        Search the index for relevant content.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of content IDs sorted by relevance
        """
        if not self.content_index:
            logger.warning("Index is empty, search returned no results")
            return []
        
        text_processor = TextProcessor()
        processed_query = text_processor.preprocess_text(query)
        
        # Get scores for each document
        scores = defaultdict(int)
        
        # Score based on lemmatized tokens
        for token in processed_query["lemmatized_tokens"]:
            for content_id in self.content_index.get(token, []):
                scores[content_id] += 1
        
        # Sort by score (descending)
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top results
        top_results = [content_id for content_id, score in sorted_results[:max_results]]
        
        logger.info(f"Search for '{query}' returned {len(top_results)} results")
        
        return top_results
    
    def save_index(self):
        """Save the index to disk."""
        try:
            index_path = os.path.join(self.cache_dir, "index.pickle")
            metadata_path = os.path.join(self.cache_dir, "metadata.pickle")
            
            with open(index_path, 'wb') as f:
                pickle.dump(dict(self.content_index), f)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Saved index with {len(self.content_index)} terms and {len(self.metadata)} documents")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self):
        """Load the index from disk."""
        try:
            index_path = os.path.join(self.cache_dir, "index.pickle")
            metadata_path = os.path.join(self.cache_dir, "metadata.pickle")
            
            # Check if both files exist and are non-empty
            if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.getsize(index_path) > 0 and os.path.getsize(metadata_path) > 0:
                with open(index_path, 'rb') as f:
                    self.content_index = defaultdict(list, pickle.load(f))
                
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded index with {len(self.content_index)} terms and {len(self.metadata)} documents")
                
                # Validate the loaded data structure
                if not isinstance(self.content_index, defaultdict) and not isinstance(self.content_index, dict):
                    logger.warning("Invalid cache structure detected, rebuilding...")
                    self.content_index = defaultdict(list)
                    self.metadata = {}
                    return False
                
                # Check for at least some content
                if len(self.content_index) > 0 and len(self.metadata) > 0:
                    return True
                else:
                    logger.warning("Empty index or metadata detected, will rebuild")
                    return False
            else:
                logger.info("No existing index found or files are empty")
                return False
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # Initialize empty index and metadata
            self.content_index = defaultdict(list)
            self.metadata = {}
            return False

class GeminiClient:
    """Wrapper for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini client with model {MODEL_NAME}")
    
    def generate_response(self, prompt, system_prompt=None, temperature=0.7, max_tokens=8192):
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating response from Gemini with temperature {temperature}")
            logger.info(f"System prompt length: {len(system_prompt) if system_prompt else 0}")
            logger.info(f"User prompt length: {len(prompt)}")
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
            )
            
            # Build the full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate response with streaming
            response_text = ""
            for chunk in self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            ):
                if chunk.candidates and chunk.candidates[0].text:
                    response_text += chunk.candidates[0].text
                    print(".", end="", flush=True)  # Show progress
            
            print()  # New line after progress dots
            
            logger.info(f"Response length: {len(response_text)} characters")
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def build_system_prompt(self, context_docs=None):
        """
        Build a comprehensive system prompt for Gemini.
        
        Args:
            context_docs: List of documents to include as context
            
        Returns:
            System prompt string
        """
        system_prompt = """You are a highly knowledgeable and professional AI assistant that specializes in providing accurate information from a company's Confluence knowledge base. Your responses should be:

1. PRECISE AND ACCURATE: Always base your answers strictly on the provided Confluence documents. If the information isn't in the provided context, clearly state that you don't have that specific information.

2. PROFESSIONAL BUT FRIENDLY: Maintain a professional tone that would be appropriate in a corporate environment, while still being approachable and helpful.

3. WELL-STRUCTURED: For complex responses, use appropriate formatting with headings, bullet points, or numbered lists to improve readability.

4. CONTEXTUALLY AWARE: If you need clarification to provide a better answer, politely ask follow-up questions to better understand the user's needs.

5. SOURCE-TRANSPARENT: Always include references to the specific Confluence pages you used to answer the question. Include the exact page titles and URLs at the end of your response.

When responding to technical questions:
- Be precise with technical terminology
- Include code snippets when relevant
- Explain complex concepts clearly

When data or tables are mentioned in the context:
- Present numerical data clearly, using tables if appropriate
- Explain what the data means in business terms

When images are referenced in the context:
- Clearly describe what information the image contains
- Explain the relevance of the image to the question

If the question seems ambiguous or could have multiple interpretations:
- Consider the most likely interpretation based on the available context
- If necessary, provide answers to multiple interpretations
- Politely ask for clarification

If you don't have enough information to fully answer the question:
- Clearly state what you do know based on the provided context
- Explain what additional information would be needed
- Suggest which Confluence spaces might contain the relevant information

FORMAT YOUR RESPONSES:
1. Start with a direct answer to the question
2. Follow with supporting details, explanations, or elaborations
3. For complex topics, use appropriate headings and structured formatting
4. End with source references listing the Confluence pages you used

Remember that you are assisting with company-internal information. Your responses should be helpful for employees trying to find and understand information in their company's knowledge base.
"""
        
        # Add context documents if provided
        if context_docs and len(context_docs) > 0:
            context_text = "\n\n### CONTEXT DOCUMENTS ###\n\n"
            
            for i, doc in enumerate(context_docs):
                metadata = doc.get("metadata", {})
                title = metadata.get("title", "Untitled Document")
                url = metadata.get("url", "")
                content = doc.get("content", "")
                
                context_text += f"[DOCUMENT {i+1}]: {title}\n"
                context_text += f"URL: {url}\n"
                context_text += f"CONTENT: {content}\n\n"
            
            system_prompt += context_text
        
        return system_prompt

class ConfluenceGeminiBot:
    """Main class that integrates Confluence content with Gemini for answering questions."""
    
    def __init__(self, confluence_url, username, api_token, cache_dir=CACHE_DIR):
        """
        Initialize the bot with Confluence credentials.
        
        Args:
            confluence_url: Base URL of Confluence instance
            username: Confluence username
            api_token: Confluence API token
            cache_dir: Directory to store cache
        """
        self.confluence = ConfluenceClient(confluence_url, username, api_token)
        self.cache = ConfluenceCache(cache_dir)
        self.text_processor = TextProcessor()
        self.gemini = GeminiClient()
        
        # Test connection
        self.confluence.test_connection()
        
        logger.info("Initialized ConfluenceGeminiBot")
    
    def index_all_content(self, force_refresh=False):
        """
        Index all Confluence content for faster searching.
        
        Args:
            force_refresh: Force refreshing content even if cached
        
        Returns:
            Boolean indicating if we're using existing cache
        """
        # Check if we already have an index
        if not force_refresh and self.cache.content_index and self.cache.metadata:
            logger.info(f"Using existing index with {len(self.cache.content_index)} terms and {len(self.cache.metadata)} documents")
            return True  # Return True to indicate we're using existing cache
        
        logger.info("Indexing all Confluence content, this may take a while...")
        
        # Get all content
        all_content = self.confluence.get_all_content(expand="body.storage")
        logger.info(f"Retrieved {len(all_content)} content items")
        
        # Process and cache each content item
        processed_content = []
        
        for content in all_content:
            content_id = content.get("id")
            
            # Skip if already cached and not forcing refresh
            if not force_refresh and self.cache.is_cache_valid(f"content_{content_id}"):
                cached_content = self.cache.get(f"content_{content_id}")
                if cached_content:
                    processed_content.append(cached_content)
                    continue
            
            # Process content
            processed = self.confluence.get_page_content(content_id)
            if processed:
                self.cache.set(f"content_{content_id}", processed)
                processed_content.append(processed)
        
        # Build index
        self.cache.build_index(processed_content)
        
        logger.info(f"Indexed {len(processed_content)} content items")
        return False  # Return False to indicate we built a new cache
    
    def search_confluence(self, query, max_results=MAX_RESULTS_PER_QUERY):
        """
        Search Confluence for content related to the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of content items matching the query
        """
        logger.info(f"Searching Confluence for: {query}")
        
        # First try the cached index search
        if self.cache.content_index and self.cache.metadata:
            content_ids = self.cache.search_index(query, max_results=max_results)
            
            if content_ids:
                # Get full content for each ID
                results = []
                for content_id in content_ids:
                    cached_content = self.cache.get(f"content_{content_id}")
                    if cached_content:
                        results.append(cached_content)
                
                logger.info(f"Found {len(results)} results from index search")
                return results
        
        # If index search failed or returned no results, try direct Confluence search
        logger.info("Falling back to direct Confluence search API")
        
        # First, try an exact match search
        search_results = self.confluence.search_content(
            cql=f'text ~ "{query}"',  # Exact phrase match
            content_type="page",
            limit=max_results
        )
        
        # If no results, try a more relaxed search
        if not search_results.get("results"):
            # Use keywords for broader search
            keywords = self.text_processor.extract_keywords(query, top_n=3)
            keyword_query = " OR ".join(keywords)
            
            search_results = self.confluence.search_content(
                cql=f'text ~ "({keyword_query})"',
                content_type="page",
                limit=max_results
            )
        
        # Process search results
        results = []
        for result in search_results.get("results", []):
            content_id = result.get("id")
            
            # Try to get from cache first
            cached_content = self.cache.get(f"content_{content_id}")
            if cached_content:
                results.append(cached_content)
                continue
            
            # If not in cache, get content and cache it
            processed = self.confluence.get_page_content(content_id)
            if processed:
                self.cache.set(f"content_{content_id}", processed)
                results.append(processed)
        
        logger.info(f"Found {len(results)} results from direct Confluence search")
        return results
    
    def answer_question(self, question, temperature=0.7):
        """
        Answer a question using Confluence content and Gemini.
        
        Args:
            question: User's question
            temperature: Temperature for Gemini response generation
            
        Returns:
            Answer from Gemini
        """
        logger.info(f"Answering question: {question}")
        
        # Process the question
        processed_question = self.text_processor.preprocess_question(question)
        
        # Handle multi-part questions
        if processed_question["is_multi_part"]:
            logger.info("Detected multi-part question, splitting into parts")
            question_parts = self.text_processor.split_multi_part_question(question)
            
            if len(question_parts) > 1:
                logger.info(f"Split into {len(question_parts)} parts: {question_parts}")
                
                # Process each part separately
                responses = []
                for i, part in enumerate(question_parts):
                    logger.info(f"Processing question part {i+1}: {part}")
                    part_response = self._process_single_question(part, temperature)
                    responses.append(f"Part {i+1}: {part}\n\n{part_response}")
                
                # Combine responses
                combined_response = "\n\n" + "-" * 50 + "\n\n".join(responses)
                return combined_response
        
        # For single questions or if splitting failed
        return self._process_single_question(question, temperature)
    
    def _process_single_question(self, question, temperature=0.7):
        """
        Process a single question.
        
        Args:
            question: Question to answer
            temperature: Temperature for response generation
            
        Returns:
            Answer to the question
        """
        # Extract keywords for better search
        keywords = self.text_processor.extract_keywords(question)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Search for relevant content
        search_query = " ".join(keywords[:3]) if keywords else question
        relevant_docs = self.search_confluence(search_query)
        
        if not relevant_docs:
            logger.warning(f"No relevant documents found for question: {question}")
            return "I couldn't find any information in the Confluence knowledge base that answers your question. Could you rephrase your question or provide more details?"
        
        # Prepare context for Gemini
        logger.info(f"Found {len(relevant_docs)} relevant documents for the question")
        
        # Build system prompt with context
        system_prompt = self.gemini.build_system_prompt(context_docs=relevant_docs)
        
        # Generate response
        response = self.gemini.generate_response(
            prompt=question,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        return response
    
    def interactive_session(self):
        """Start an interactive Q&A session in the console."""
        print("\n=== Confluence Gemini Bot ===")
        print("Type 'exit' or 'quit' to end the session.")
        
        while True:
            try:
                question = input("\nYour question: ")
                
                if question.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                # Process the question
                print("\nThinking...")
                answer = self.answer_question(question)
                
                print("\nAnswer:")
                print("=" * 80)
                print(answer)
                print("=" * 80)
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {str(e)}")
                print(f"An error occurred: {str(e)}")

def main(force_rebuild=False):
    """Main function to run the application."""
    # Configuration
    confluence_url = os.environ.get("CONFLUENCE_URL", "https://your-company.atlassian.net")
    username = os.environ.get("CONFLUENCE_USERNAME", "your.email@company.com")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "your-api-token")
    
    # Initialize the bot
    try:
        bot = ConfluenceGeminiBot(confluence_url, username, api_token)
        
        if force_rebuild:
            print("Forcing cache rebuild as requested...")
            # This will run in the main thread to show progress
            bot.index_all_content(force_refresh=True)
        else:
            # Check if we have a valid cache first
            cache_exists = bot.cache.content_index and bot.cache.metadata
            if cache_exists:
                logger.info("Found existing cache, using it without reindexing")
                print("Using existing Confluence content cache...")
            else:
                # Only index in background if no cache exists
                print("No cache found. Building content index in background...")
                threading.Thread(target=bot.index_all_content).start()
        
        # Start interactive session
        bot.interactive_session()
    except Exception as e:
        logger.error(f"Error initializing bot: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Confluence Gemini Bot")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild the content cache")
    args = parser.parse_args()
    
    # Call main with force_rebuild parameter
    main(force_rebuild=args.rebuild_cache)
















