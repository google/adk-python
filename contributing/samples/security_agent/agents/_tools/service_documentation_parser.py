#!/usr/bin/env python3
"""
Intelligent Service Documentation Parser
Dynamically learns about new GCP services by parsing documentation URLs
"""

import os
import re
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import hashlib
import sqlite3
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceDocumentationParser:
    """
    Intelligently parses GCP documentation to learn about new services
    """

    # Known documentation patterns
    DOC_PATTERNS = {
        'cloud.google.com': {
            'service_name': r'<h1[^>]*>([^<]+)</h1>',
            'api_endpoint': r'([a-z\-]+)\.googleapis\.com',
            'resource_types': r'/([a-z]+)/v\d+/([a-z]+)',
            'permissions': r'([a-z]+\.[a-z]+\.[a-z]+)',
            'quotas': r'quota[s]?[:\s]+([^<\n]+)',
            'pricing': r'pricing[:\s]+([^<\n]+)',
            'regions': r'(us-[a-z]+\d+|europe-[a-z]+\d+|asia-[a-z]+\d+)',
        },
        'github.com': {
            'api_version': r'/v(\d+)/',
            'methods': r'(GET|POST|PUT|DELETE|PATCH)\s+/[^\s]+',
            'parameters': r'--([a-z\-]+)',
            'examples': r'```([^`]+)```',
        }
    }

    # Service capability indicators
    CAPABILITY_KEYWORDS = {
        'security': ['encryption', 'iam', 'audit', 'compliance', 'firewall', 'vpc', 'private'],
        'compute': ['instance', 'vm', 'cpu', 'memory', 'gpu', 'container', 'kubernetes'],
        'storage': ['bucket', 'object', 'file', 'disk', 'volume', 'backup', 'archive'],
        'database': ['sql', 'nosql', 'query', 'table', 'index', 'transaction', 'replica'],
        'networking': ['load balancer', 'cdn', 'dns', 'vpn', 'interconnect', 'peering'],
        'ai_ml': ['machine learning', 'ai', 'model', 'prediction', 'training', 'inference'],
        'analytics': ['bigquery', 'dataflow', 'dataproc', 'pipeline', 'streaming', 'batch'],
        'serverless': ['function', 'cloud run', 'app engine', 'serverless', 'event-driven'],
        'monitoring': ['logging', 'metrics', 'trace', 'debug', 'alert', 'dashboard'],
        'integration': ['pub/sub', 'eventarc', 'workflow', 'composer', 'api gateway'],
    }

    def __init__(self, cache_dir: str = "cache/service_docs"):
        """Initialize the documentation parser"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize cache database
        self.cache_db = os.path.join(cache_dir, "parsed_services.db")
        self._init_cache_db()

        # Initialize BigQuery client for storing learned services
        try:
            self.bq_client = bigquery.Client()
            self.project_id = self.bq_client.project
        except Exception as e:
            logger.warning(f"BigQuery client not available: {e}")
            self.bq_client = None

    def _init_cache_db(self):
        """Initialize the cache database for parsed services"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parsed_services (
                url TEXT PRIMARY KEY,
                service_name TEXT,
                api_endpoint TEXT,
                resource_types TEXT,
                capabilities TEXT,
                permissions TEXT,
                regions TEXT,
                parsed_data TEXT,
                parse_date TIMESTAMP,
                doc_hash TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS service_methods (
                service_name TEXT,
                method_name TEXT,
                http_method TEXT,
                endpoint TEXT,
                parameters TEXT,
                description TEXT,
                example TEXT,
                PRIMARY KEY (service_name, method_name)
            )
        ''')

        conn.commit()
        conn.close()

    def parse_documentation_url(
        self,
        url: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Parse a documentation URL to learn about a GCP service

        Args:
            url: Documentation URL (e.g., https://cloud.google.com/new-service/docs)
            force_refresh: Force re-parsing even if cached

        Returns:
            Parsed service information
        """
        # Check cache first
        if not force_refresh:
            cached = self._get_cached_service(url)
            if cached:
                logger.info(f"Using cached data for {url}")
                return cached

        logger.info(f"Parsing documentation from: {url}")

        try:
            # Fetch the documentation page
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract service information
            service_info = self._extract_service_info(soup, url)

            # Identify capabilities
            capabilities = self._identify_capabilities(response.text)
            service_info['capabilities'] = capabilities

            # Extract API information
            api_info = self._extract_api_info(soup, response.text)
            service_info.update(api_info)

            # Extract resource types
            resource_types = self._extract_resource_types(soup, response.text)
            service_info['resource_types'] = resource_types

            # Extract permissions and IAM roles
            permissions = self._extract_permissions(soup, response.text)
            service_info['permissions'] = permissions

            # Extract region availability
            regions = self._extract_regions(response.text)
            service_info['regions'] = regions

            # Extract code examples
            examples = self._extract_code_examples(soup)
            service_info['examples'] = examples

            # Extract related links for deeper parsing
            related_links = self._extract_related_links(soup, url)
            service_info['related_links'] = related_links

            # Cache the parsed data
            self._cache_service(url, service_info)

            # Store in BigQuery if available
            if self.bq_client:
                self._store_in_bigquery(service_info)

            return service_info

        except Exception as e:
            logger.error(f"Error parsing documentation: {e}")
            return {
                'error': str(e),
                'url': url,
                'status': 'failed'
            }

    def _extract_service_info(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract basic service information from the page"""
        info = {
            'url': url,
            'parsed_at': datetime.now().isoformat()
        }

        # Try to extract service name from title or h1
        title = soup.find('title')
        if title:
            info['page_title'] = title.text.strip()
            # Extract service name from title (e.g., "Cloud New Service | Google Cloud")
            match = re.search(r'^([^|]+)', title.text)
            if match:
                info['service_name'] = match.group(1).strip()

        # Look for h1 as alternative
        h1 = soup.find('h1')
        if h1 and 'service_name' not in info:
            info['service_name'] = h1.text.strip()

        # Extract description from meta tags or first paragraph
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            info['description'] = meta_desc.get('content', '')
        else:
            # Try first paragraph
            first_p = soup.find('p')
            if first_p:
                info['description'] = first_p.text.strip()[:500]

        # Extract version information
        version_pattern = r'v\d+(\.\d+)*|beta|alpha|preview'
        version_match = re.search(version_pattern, url, re.IGNORECASE)
        if version_match:
            info['version'] = version_match.group(0)

        return info

    def _identify_capabilities(self, content: str) -> List[str]:
        """Identify service capabilities based on keywords"""
        content_lower = content.lower()
        identified_capabilities = []

        for capability, keywords in self.CAPABILITY_KEYWORDS.items():
            # Count keyword occurrences
            keyword_count = sum(1 for keyword in keywords if keyword in content_lower)

            # If multiple keywords found, likely has this capability
            if keyword_count >= 2:
                identified_capabilities.append(capability)

        return identified_capabilities

    def _extract_api_info(self, soup: BeautifulSoup, content: str) -> Dict[str, Any]:
        """Extract API endpoint and version information"""
        api_info = {}

        # Look for API endpoint patterns
        api_pattern = r'([a-z][a-z0-9\-]*\.googleapis\.com)'
        api_matches = re.findall(api_pattern, content)
        if api_matches:
            api_info['api_endpoint'] = api_matches[0]

            # Extract service key from API endpoint
            service_key = api_matches[0].replace('.googleapis.com', '').replace('-', '')
            api_info['service_key'] = service_key

        # Look for API version
        version_pattern = r'/v(\d+[\.\d]*)\b'
        version_matches = re.findall(version_pattern, content)
        if version_matches:
            api_info['api_version'] = f"v{version_matches[0]}"

        # Look for REST API methods
        methods = []
        method_pattern = r'(GET|POST|PUT|DELETE|PATCH)\s+(/[a-zA-Z0-9/\{\}]+)'
        method_matches = re.findall(method_pattern, content)

        for http_method, endpoint in method_matches[:10]:  # Limit to first 10
            methods.append({
                'method': http_method,
                'endpoint': endpoint
            })

        if methods:
            api_info['api_methods'] = methods

        return api_info

    def _extract_resource_types(self, soup: BeautifulSoup, content: str) -> List[str]:
        """Extract resource types mentioned in the documentation"""
        resource_types = set()

        # Common resource type patterns
        patterns = [
            r'resource[s]?\s+type[s]?[:\s]+([a-zA-Z\.\s,]+)',
            r'create[s]?\s+([a-z]+)',
            r'manage[s]?\s+([a-z]+)',
            r'/([a-z]+)/v\d+',
            r'type:\s*([a-zA-Z\.]+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Clean and validate resource type
                resource = match.strip().lower()
                if len(resource) > 2 and len(resource) < 50 and ' ' not in resource:
                    resource_types.add(resource)

        # Look for resource types in code blocks
        code_blocks = soup.find_all(['code', 'pre'])
        for block in code_blocks:
            text = block.text
            # Look for resource definitions
            if 'resource' in text.lower() or 'type' in text.lower():
                words = re.findall(r'\b[a-z][a-z\.]+\b', text.lower())
                for word in words:
                    if '.' in word and len(word) > 5:
                        resource_types.add(word)

        return list(resource_types)[:20]  # Limit to 20 resource types

    def _extract_permissions(self, soup: BeautifulSoup, content: str) -> List[str]:
        """Extract IAM permissions and roles"""
        permissions = set()

        # IAM permission pattern (e.g., compute.instances.create)
        perm_pattern = r'([a-z]+\.[a-z]+\.[a-z]+)'
        perm_matches = re.findall(perm_pattern, content)

        for perm in perm_matches:
            # Validate it looks like a permission
            parts = perm.split('.')
            if len(parts) == 3 and all(len(p) > 1 for p in parts):
                permissions.add(perm)

        # Look for role patterns
        role_pattern = r'roles/([a-zA-Z\.]+)'
        role_matches = re.findall(role_pattern, content)
        for role in role_matches:
            permissions.add(f"role:{role}")

        return list(permissions)[:30]  # Limit to 30 permissions

    def _extract_regions(self, content: str) -> List[str]:
        """Extract region availability"""
        regions = set()

        # GCP region patterns
        region_patterns = [
            r'(us-[a-z]+\d+)',
            r'(europe-[a-z]+\d+)',
            r'(asia-[a-z]+\d+)',
            r'(australia-[a-z]+\d+)',
            r'(northamerica-[a-z]+\d+)',
            r'(southamerica-[a-z]+\d+)',
        ]

        for pattern in region_patterns:
            matches = re.findall(pattern, content)
            regions.update(matches)

        return list(regions)

    def _extract_code_examples(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract code examples from the documentation"""
        examples = []

        # Look for code blocks
        code_blocks = soup.find_all('pre')
        for block in code_blocks[:5]:  # Limit to 5 examples
            code = block.find('code')
            if code:
                # Try to identify the language
                language = 'unknown'
                if code.get('class'):
                    classes = code.get('class')
                    for cls in classes:
                        if 'language-' in cls:
                            language = cls.replace('language-', '')
                            break

                example = {
                    'code': code.text.strip(),
                    'language': language
                }

                # Try to identify what the example does
                lines = code.text.strip().split('\n')
                if lines:
                    # Look for comments or command
                    for line in lines[:3]:
                        if '#' in line or '//' in line or '/*' in line:
                            example['description'] = line.strip('#//* \t')
                            break

                examples.append(example)

        return examples

    def _extract_related_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract related documentation links for deeper parsing"""
        links = []
        base_domain = urlparse(base_url).netloc

        # Look for navigation links and related docs
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Make absolute URL
            if href.startswith('/'):
                href = urljoin(base_url, href)

            # Filter for documentation links
            if base_domain in href and any(keyword in href.lower() for keyword in
                ['docs', 'reference', 'api', 'guide', 'tutorial', 'quickstart']):

                # Avoid duplicates and anchors
                if '#' not in href and href not in links and href != base_url:
                    links.append(href)

        return links[:10]  # Limit to 10 related links

    def _get_cached_service(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached service data if available"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT parsed_data, parse_date
            FROM parsed_services
            WHERE url = ?
        ''', (url,))

        result = cursor.fetchone()
        conn.close()

        if result:
            parsed_data, parse_date = result
            # Check if cache is still valid (24 hours)
            parse_datetime = datetime.fromisoformat(parse_date)
            if datetime.now() - parse_datetime < timedelta(hours=24):
                return json.loads(parsed_data)

        return None

    def _cache_service(self, url: str, service_info: Dict[str, Any]):
        """Cache parsed service data"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Calculate document hash for change detection
        doc_hash = hashlib.md5(json.dumps(service_info, sort_keys=True).encode()).hexdigest()

        cursor.execute('''
            INSERT OR REPLACE INTO parsed_services
            (url, service_name, api_endpoint, resource_types, capabilities,
             permissions, regions, parsed_data, parse_date, doc_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            url,
            service_info.get('service_name', ''),
            service_info.get('api_endpoint', ''),
            json.dumps(service_info.get('resource_types', [])),
            json.dumps(service_info.get('capabilities', [])),
            json.dumps(service_info.get('permissions', [])),
            json.dumps(service_info.get('regions', [])),
            json.dumps(service_info),
            datetime.now().isoformat(),
            doc_hash
        ))

        conn.commit()
        conn.close()

    def _store_in_bigquery(self, service_info: Dict[str, Any]):
        """Store learned service in BigQuery for persistence"""
        try:
            dataset_id = "learned_services"
            table_id = "discovered_services"

            # Create dataset if it doesn't exist
            dataset_ref = self.bq_client.dataset(dataset_id)
            try:
                self.bq_client.get_dataset(dataset_ref)
            except:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset = self.bq_client.create_dataset(dataset, exists_ok=True)

            # Define schema
            schema = [
                bigquery.SchemaField("service_name", "STRING"),
                bigquery.SchemaField("api_endpoint", "STRING"),
                bigquery.SchemaField("description", "STRING"),
                bigquery.SchemaField("capabilities", "STRING", mode="REPEATED"),
                bigquery.SchemaField("resource_types", "STRING", mode="REPEATED"),
                bigquery.SchemaField("permissions", "STRING", mode="REPEATED"),
                bigquery.SchemaField("regions", "STRING", mode="REPEATED"),
                bigquery.SchemaField("documentation_url", "STRING"),
                bigquery.SchemaField("discovered_at", "TIMESTAMP"),
                bigquery.SchemaField("raw_data", "JSON"),
            ]

            # Create table if it doesn't exist
            table_ref = dataset_ref.table(table_id)
            try:
                table = self.bq_client.get_table(table_ref)
            except:
                table = bigquery.Table(table_ref, schema=schema)
                table = self.bq_client.create_table(table, exists_ok=True)

            # Insert the service data
            rows = [{
                "service_name": service_info.get('service_name', 'Unknown'),
                "api_endpoint": service_info.get('api_endpoint', ''),
                "description": service_info.get('description', ''),
                "capabilities": service_info.get('capabilities', []),
                "resource_types": service_info.get('resource_types', []),
                "permissions": service_info.get('permissions', []),
                "regions": service_info.get('regions', []),
                "documentation_url": service_info.get('url', ''),
                "discovered_at": datetime.now().isoformat(),
                "raw_data": service_info,
            }]

            errors = self.bq_client.insert_rows_json(table_ref, rows)

            if errors:
                logger.error(f"Error storing in BigQuery: {errors}")
            else:
                logger.info(f"Successfully stored {service_info.get('service_name')} in BigQuery")

        except Exception as e:
            logger.error(f"Failed to store in BigQuery: {e}")

    def parse_release_notes(self, url: str) -> List[Dict[str, Any]]:
        """Parse GCP release notes to find new services"""
        logger.info(f"Parsing release notes from: {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            new_services = []

            # Look for release note entries
            entries = soup.find_all(['article', 'div'], class_=re.compile('release|announcement|update'))

            for entry in entries:
                # Look for "new" or "launched" keywords
                text = entry.text.lower()
                if any(keyword in text for keyword in ['new service', 'now available', 'launched', 'introducing']):

                    # Extract service name and link
                    title = entry.find(['h2', 'h3', 'h4'])
                    if title:
                        service_name = title.text.strip()

                        # Look for documentation link
                        doc_link = entry.find('a', href=re.compile('/docs/'))
                        if doc_link:
                            new_service = {
                                'name': service_name,
                                'announcement_url': url,
                                'documentation_url': urljoin(url, doc_link['href']),
                                'found_date': datetime.now().isoformat()
                            }
                            new_services.append(new_service)

            return new_services

        except Exception as e:
            logger.error(f"Error parsing release notes: {e}")
            return []

    def learn_from_github_api(self, repo_url: str) -> Dict[str, Any]:
        """Learn about a service from its GitHub API documentation"""
        logger.info(f"Learning from GitHub: {repo_url}")

        try:
            # Parse GitHub raw content URL
            if 'github.com' in repo_url and '/blob/' in repo_url:
                # Convert to raw URL
                repo_url = repo_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

            response = requests.get(repo_url, timeout=30)
            response.raise_for_status()

            content = response.text
            service_info = {'source': 'github', 'url': repo_url}

            # Parse API specification (OpenAPI/Swagger)
            if repo_url.endswith('.json'):
                api_spec = json.loads(content)
                service_info.update(self._parse_openapi_spec(api_spec))

            # Parse proto files
            elif repo_url.endswith('.proto'):
                service_info.update(self._parse_proto_file(content))

            # Parse markdown documentation
            elif repo_url.endswith('.md'):
                service_info.update(self._parse_markdown_api_doc(content))

            return service_info

        except Exception as e:
            logger.error(f"Error learning from GitHub: {e}")
            return {'error': str(e)}

    def _parse_openapi_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAPI/Swagger specification"""
        info = {
            'api_type': 'openapi',
            'title': spec.get('info', {}).get('title', ''),
            'version': spec.get('info', {}).get('version', ''),
            'description': spec.get('info', {}).get('description', ''),
        }

        # Extract endpoints
        endpoints = []
        if 'paths' in spec:
            for path, methods in spec['paths'].items():
                for method, details in methods.items():
                    if method in ['get', 'post', 'put', 'delete', 'patch']:
                        endpoints.append({
                            'method': method.upper(),
                            'path': path,
                            'summary': details.get('summary', ''),
                            'operationId': details.get('operationId', '')
                        })

        info['endpoints'] = endpoints

        # Extract schemas/models
        if 'components' in spec and 'schemas' in spec['components']:
            info['models'] = list(spec['components']['schemas'].keys())

        return info

    def _parse_proto_file(self, content: str) -> Dict[str, Any]:
        """Parse protobuf file for service definition"""
        info = {'api_type': 'grpc'}

        # Extract service name
        service_match = re.search(r'service\s+(\w+)\s*\{', content)
        if service_match:
            info['service_name'] = service_match.group(1)

        # Extract RPC methods
        methods = []
        rpc_pattern = r'rpc\s+(\w+)\s*\(([^)]+)\)\s+returns\s+\(([^)]+)\)'
        for match in re.finditer(rpc_pattern, content):
            methods.append({
                'name': match.group(1),
                'request': match.group(2).strip(),
                'response': match.group(3).strip()
            })

        info['methods'] = methods

        # Extract messages
        message_pattern = r'message\s+(\w+)\s*\{'
        messages = re.findall(message_pattern, content)
        info['messages'] = messages

        return info

    def _parse_markdown_api_doc(self, content: str) -> Dict[str, Any]:
        """Parse markdown API documentation"""
        info = {'api_type': 'markdown'}

        # Extract title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            info['title'] = title_match.group(1)

        # Extract API endpoints from markdown
        endpoints = []
        endpoint_pattern = r'`(GET|POST|PUT|DELETE|PATCH)\s+([^`]+)`'
        for match in re.finditer(endpoint_pattern, content):
            endpoints.append({
                'method': match.group(1),
                'path': match.group(2)
            })

        info['endpoints'] = endpoints

        # Extract code examples
        code_blocks = re.findall(r'```[^`]+```', content)
        info['examples_count'] = len(code_blocks)

        return info


# Tool functions for ADK agent integration
def parse_service_documentation(url: str) -> str:
    """
    Parse GCP service documentation from a URL to learn about new services.

    Args:
        url: Documentation URL (e.g., https://cloud.google.com/new-service/docs)

    Returns:
        Formatted analysis of the service
    """
    parser = ServiceDocumentationParser()
    result = parser.parse_documentation_url(url)

    if 'error' in result:
        return f"❌ Failed to parse documentation: {result['error']}"

    output = f"📚 Parsed Service Documentation\n"
    output += "=" * 50 + "\n\n"

    output += f"Service: {result.get('service_name', 'Unknown')}\n"
    output += f"API Endpoint: {result.get('api_endpoint', 'Not found')}\n"
    output += f"Description: {result.get('description', 'No description')[:200]}...\n\n"

    if result.get('capabilities'):
        output += f"Capabilities: {', '.join(result['capabilities'])}\n"

    if result.get('resource_types'):
        output += f"Resource Types: {', '.join(result['resource_types'][:5])}\n"

    if result.get('regions'):
        output += f"Available Regions: {', '.join(result['regions'][:5])}\n"

    if result.get('permissions'):
        output += f"IAM Permissions: {len(result['permissions'])} found\n"

    if result.get('examples'):
        output += f"Code Examples: {len(result['examples'])} found\n"

    if result.get('related_links'):
        output += f"\nRelated Documentation:\n"
        for link in result['related_links'][:3]:
            output += f"  • {link}\n"

    return output


def discover_new_services(release_notes_url: Optional[str] = None) -> str:
    """
    Discover newly released GCP services from release notes or documentation.

    Args:
        release_notes_url: Optional URL to GCP release notes

    Returns:
        List of discovered new services
    """
    parser = ServiceDocumentationParser()

    # Default to GCP release notes if no URL provided
    if not release_notes_url:
        release_notes_url = "https://cloud.google.com/release-notes"

    new_services = parser.parse_release_notes(release_notes_url)

    output = f"🆕 New GCP Services Discovery\n"
    output += "=" * 50 + "\n\n"

    if new_services:
        output += f"Found {len(new_services)} new services:\n\n"

        for i, service in enumerate(new_services, 1):
            output += f"{i}. {service['name']}\n"
            output += f"   Documentation: {service['documentation_url']}\n"
            output += f"   Discovered: {service['found_date']}\n\n"

        output += "\nUse parse_service_documentation() with the documentation URL to learn more about each service."
    else:
        output += "No new services found in recent release notes.\n"
        output += "Try providing a specific release notes URL or documentation page."

    return output


def learn_service_from_api_spec(api_spec_url: str) -> str:
    """
    Learn about a service from its API specification (OpenAPI, Proto, etc).

    Args:
        api_spec_url: URL to API specification file

    Returns:
        Parsed API information
    """
    parser = ServiceDocumentationParser()

    if 'github' in api_spec_url:
        result = parser.learn_from_github_api(api_spec_url)
    else:
        # Try to parse as regular documentation
        result = parser.parse_documentation_url(api_spec_url)

    if 'error' in result:
        return f"❌ Failed to learn from API spec: {result['error']}"

    output = f"🔧 API Specification Analysis\n"
    output += "=" * 50 + "\n\n"

    output += f"API Type: {result.get('api_type', 'Unknown')}\n"
    output += f"Title: {result.get('title', result.get('service_name', 'Unknown'))}\n"
    output += f"Version: {result.get('version', result.get('api_version', 'Unknown'))}\n\n"

    if result.get('endpoints'):
        output += f"Endpoints ({len(result['endpoints'])} found):\n"
        for endpoint in result['endpoints'][:5]:
            output += f"  • {endpoint.get('method', '')} {endpoint.get('path', endpoint.get('name', ''))}\n"

    if result.get('methods'):
        output += f"\nRPC Methods ({len(result['methods'])} found):\n"
        for method in result['methods'][:5]:
            output += f"  • {method['name']}: {method['request']} → {method['response']}\n"

    if result.get('models'):
        output += f"\nData Models: {', '.join(result['models'][:10])}\n"

    if result.get('messages'):
        output += f"\nProtobuf Messages: {', '.join(result['messages'][:10])}\n"

    return output


def register_custom_service(
    service_name: str,
    api_endpoint: str,
    documentation_url: str,
    resource_types: List[str] = None,
    capabilities: List[str] = None
) -> str:
    """
    Manually register a custom GCP service that the agent should know about.

    Args:
        service_name: Name of the service
        api_endpoint: API endpoint (e.g., myservice.googleapis.com)
        documentation_url: URL to service documentation
        resource_types: List of resource types the service manages
        capabilities: List of service capabilities (security, compute, etc.)

    Returns:
        Registration status
    """
    parser = ServiceDocumentationParser()

    # Create service info
    service_info = {
        'service_name': service_name,
        'api_endpoint': api_endpoint,
        'url': documentation_url,
        'resource_types': resource_types or [],
        'capabilities': capabilities or [],
        'registered_at': datetime.now().isoformat(),
        'source': 'manual_registration'
    }

    # Cache the service
    parser._cache_service(documentation_url, service_info)

    # Store in BigQuery if available
    if parser.bq_client:
        parser._store_in_bigquery(service_info)

    output = f"✅ Service Registered Successfully\n"
    output += "=" * 50 + "\n\n"
    output += f"Service: {service_name}\n"
    output += f"API: {api_endpoint}\n"
    output += f"Documentation: {documentation_url}\n"

    if resource_types:
        output += f"Resource Types: {', '.join(resource_types)}\n"

    if capabilities:
        output += f"Capabilities: {', '.join(capabilities)}\n"

    output += f"\nThe service is now registered and available for analysis."

    return output