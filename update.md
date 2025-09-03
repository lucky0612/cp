#!/usr/bin/env python3
"""
Multi-Instance Database Connector
Supports OSCAR, Copper, Star, and EDBCore instances
Using Google Cloud SQL with IAM authentication
"""

import logging
import argparse
import sys
from typing import List, Dict, Any, Optional
from google.cloud.sql.connector import Connector
from sqlalchemy import create_engine, text
from google.auth import default
from google.auth.transport.requests import Request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database instance configurations
DB_INSTANCES = {
    'oscar': {
        'project': 'prj-dv-oscar-8302',
        'region': 'us-central1',
        'instance': 'csql-dv-usc1-1316-oscar-0004-m',
        'database': 'pgdb',
        'user': 'takehiya.vijay@caegroup.com',
        'schema': 'dv01csrs',
        'description': 'OSCAR Database Instance'
    },
    'star': {
        'project': 'prj-dv-star-4002',
        'region': 'us-central1', 
        'instance': 'csql-dv-usc1-485-star-0001-m',
        'database': 'pgdb',
        'user': 'takehiya.vijay@caegroup.com',
        'schema': 'dv01csrs',
        'description': 'Star Database Instance'
    },
    'edbcore': {
        'project': 'prj-dv-edbcore-6694',
        'region': 'us-central1',
        'instance': 'csql-dv-usc1-173-edbcore-0004-m', 
        'database': 'pgdb',
        'user': 'takehiya.vijay@caegroup.com',
        'schema': 'dv01csrs',
        'description': 'EDBCore Database Instance'
    },
    'copper': {
        'project': 'prj-dv-copper-5609',
        'region': 'us-central1',
        'instance': 'csql-dv-usc1-747-refdata-0001-m',
        'database': 'pgdb', 
        'user': 'takehiya.vijay@caegroup.com',
        'schema': 'dv01csrs',
        'description': 'Copper Database Instance'
    }
}

class DatabaseConnector:
    def __init__(self, instance_name: str):
        """Initialize database connector for specified instance"""
        
        if instance_name not in DB_INSTANCES:
            raise ValueError(f"Unknown instance: {instance_name}. Available: {list(DB_INSTANCES.keys())}")
        
        self.instance_name = instance_name
        self.config = DB_INSTANCES[instance_name]
        
        self.project = self.config['project']
        self.region = self.config['region']
        self.instance = self.config['instance']
        self.database = self.config['database']
        self.user = self.config['user']
        self.schema = self.config['schema']
        
        self.instance_connection_name = f"{self.project}:{self.region}:{self.instance}"
        self.connector = None
        self.engine = None
        
        logger.info(f"Initializing {self.config['description']} connector for {self.instance_connection_name}")
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            logger.info("Initializing Cloud SQL Connector...")
            
            # Test authentication
            credentials, project = default()
            if hasattr(credentials, 'refresh'):
                credentials.refresh(Request())
            
            logger.info("Authentication successful")
            
            # Initialize connector
            self.connector = Connector()
            
            # Create connection function
            def getconn():
                logger.debug("Creating database connection...")
                conn = self.connector.connect(
                    self.instance_connection_name,
                    "pg8000",
                    user=self.user,
                    db=self.database,
                    enable_iam_auth=True,
                    ip_type="PRIVATE"
                )
                return conn
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                "postgresql+pg8000://",
                creator=getconn,
                pool_size=5,
                max_overflow=2,
                pool_pre_ping=True,
                pool_recycle=300
            )
            
            logger.info("Database engine created successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if not self.engine:
                logger.error("Database engine not initialized")
                return False
            
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                
                if row and row[0] == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed - unexpected result")
                    return False
        
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries representing rows
        """
        try:
            if not self.engine:
                raise Exception("Database engine not initialized")
            
            logger.debug(f"Executing query: {query[:100]}...")
            if params:
                logger.debug(f"Parameters: {params}")
            
            with self.engine.connect() as connection:
                if params:
                    result = connection.execute(text(query), params)
                else:
                    result = connection.execute(text(query))
                
                # Convert to list of dictionaries
                columns = result.keys()
                rows = []
                
                for row in result.fetchall():
                    row_dict = {}
                    for i, column in enumerate(columns):
                        row_dict[column] = row[i]
                    rows.append(row_dict)
                
                logger.info(f"Query executed successfully, returned {len(rows)} rows")
                return rows
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise

    def get_available_namespaces(self) -> List[Dict[str, Any]]:
        """Get available namespaces in ACTIVE_XML_DATA_STORE"""
        try:
            query = f"""
            SELECT namespace, COUNT(*) as record_count
            FROM {self.schema}.active_xml_data_store
            GROUP BY namespace
            ORDER BY record_count DESC
            """
            
            return self.execute_query(query)
        
        except Exception as e:
            logger.error(f"Failed to get namespaces: {e}")
            raise

    def get_sample_guids(self, namespace: str = 'GlobeUserSignature', limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample GUIDs for testing"""
        try:
            query = f"""
            SELECT GUID, namespace,
            CASE 
                WHEN LENGTH(XMLListext) > 100 THEN LEFT(XMLListext, 100) || '...'
                ELSE XMLListext
            END as xml_sample
            FROM {self.schema}.active_xml_data_store
            WHERE namespace = :namespace
            ORDER BY GUID
            LIMIT :limit
            """
            
            return self.execute_query(query, {'namespace': namespace, 'limit': limit})
        
        except Exception as e:
            logger.error(f"Failed to get sample GUIDs: {e}")
            raise

    def check_table_structure(self, table_name: str = 'active_xml_data_store') -> List[Dict[str, Any]]:
        """Check table structure"""
        try:
            query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = :schema
            AND table_name = :table_name
            ORDER BY ordinal_position
            """
            
            return self.execute_query(query, {'schema': self.schema, 'table_name': table_name})
        
        except Exception as e:
            logger.error(f"Failed to check table structure: {e}")
            raise

    def get_record_count(self, table_name: str = 'active_xml_data_store') -> int:
        """Get total record count"""
        try:
            query = f"SELECT COUNT(*) as count FROM {self.schema}.{table_name}"
            result = self.execute_query(query)
            
            return result[0]['count'] if result else 0
        
        except Exception as e:
            logger.error(f"Failed to get record count: {e}")
            raise

    def test_xml_xpath_functionality(self) -> bool:
        """Test if XML xpath functions work"""
        try:
            query = f"""
            SELECT 
                GUID,
                xpath('//globeUserSignature/globeUserSignatureInfo/globeFirstID/text()', XMLListext) as gfid_test
            FROM {self.schema}.active_xml_data_store
            WHERE namespace = 'GlobeUserSignature'
            AND XML IS NOT NULL
            LIMIT 1
            """
            
            result = self.execute_query(query)
            
            if result and len(result) > 0:
                logger.info("XML xpath functionality test successful")
                return True
            else:
                logger.warning("XML xpath test returned no results")
                return False
        
        except Exception as e:
            logger.error(f"XML xpath functionality test failed: {e}")
            return False

    def cleanup(self):
        """Clean up database connections"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database engine disposed")
            
            if self.connector:
                self.connector.close()
                logger.info("Database connector closed")
        
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class MultiInstanceManager:
    """Manages connections to multiple database instances"""
    
    def __init__(self):
        self.connectors = {}
    
    def get_connector(self, instance_name: str) -> DatabaseConnector:
        """Get or create connector for specified instance"""
        if instance_name not in self.connectors:
            self.connectors[instance_name] = DatabaseConnector(instance_name)
        return self.connectors[instance_name]
    
    def test_all_connections(self):
        """Test connections to all instances"""
        print("\n" + "="*60)
        print("TESTING ALL DATABASE CONNECTIONS")
        print("="*60)
        
        results = {}
        for instance_name in DB_INSTANCES.keys():
            print(f"\nTesting {instance_name.upper()} instance...")
            try:
                connector = self.get_connector(instance_name)
                success = connector.test_connection()
                results[instance_name] = success
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"{instance_name.upper()}: {status}")
            except Exception as e:
                results[instance_name] = False
                print(f"{instance_name.upper()}: ✗ FAILED - {str(e)}")
        
        print("\n" + "="*60)
        print("CONNECTION TEST SUMMARY")
        print("="*60)
        for instance, success in results.items():
            status = "✓ Connected" if success else "✗ Failed"
            print(f"{instance.upper():<10}: {status}")
        
        return results
    
    def get_instance_info(self, instance_name: str):
        """Get detailed information about a specific instance"""
        print(f"\n" + "="*60)
        print(f"{instance_name.upper()} INSTANCE INFORMATION")
        print("="*60)
        
        try:
            connector = self.get_connector(instance_name)
            config = DB_INSTANCES[instance_name]
            
            print(f"Description: {config['description']}")
            print(f"Project: {config['project']}")
            print(f"Region: {config['region']}")
            print(f"Instance: {config['instance']}")
            print(f"Database: {config['database']}")
            print(f"Schema: {config['schema']}")
            print(f"User: {config['user']}")
            print(f"Connection String: {connector.instance_connection_name}")
            
            # Test connection
            print(f"\nTesting connection...")
            if connector.test_connection():
                print("✓ Connection successful")
                
                # Get record count
                try:
                    count = connector.get_record_count()
                    print(f"✓ Total records in active_xml_data_store: {count:,}")
                except:
                    print("✗ Could not get record count")
                
                # Get namespaces
                try:
                    namespaces = connector.get_available_namespaces()
                    print(f"✓ Available namespaces: {len(namespaces)}")
                    if namespaces:
                        print("  Top namespaces:")
                        for ns in namespaces[:5]:
                            print(f"    - {ns['namespace']}: {ns['record_count']:,} records")
                except:
                    print("✗ Could not get namespaces")
            else:
                print("✗ Connection failed")
                
        except Exception as e:
            print(f"✗ Error getting instance info: {e}")
    
    def test_xml_functionality(self, instance_name: str):
        """Test XML xpath functionality for specific instance"""
        print(f"\n" + "="*60)
        print(f"TESTING XML FUNCTIONALITY - {instance_name.upper()}")
        print("="*60)
        
        try:
            connector = self.get_connector(instance_name)
            
            if connector.test_xml_xpath_functionality():
                print("✓ XML xpath functionality working")
            else:
                print("✗ XML xpath functionality failed")
                
        except Exception as e:
            print(f"✗ XML functionality test failed: {e}")
    
    def get_sample_data(self, instance_name: str, namespace: str = 'GlobeUserSignature', limit: int = 3):
        """Get sample data from specific instance"""
        print(f"\n" + "="*60)
        print(f"SAMPLE DATA - {instance_name.upper()} - {namespace}")
        print("="*60)
        
        try:
            connector = self.get_connector(instance_name)
            samples = connector.get_sample_guids(namespace, limit)
            
            if samples:
                print(f"Found {len(samples)} sample records:")
                for i, sample in enumerate(samples, 1):
                    print(f"\n{i}. GUID: {sample['guid']}")
                    print(f"   Namespace: {sample['namespace']}")
                    print(f"   XML Sample: {sample['xml_sample']}")
            else:
                print("No sample data found")
                
        except Exception as e:
            print(f"✗ Error getting sample data: {e}")
    
    def cleanup_all(self):
        """Clean up all connections"""
        for connector in self.connectors.values():
            connector.cleanup()


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Multi-Instance Database Connector CLI')
    parser.add_argument('command', choices=[
        'test-all', 'test-connection', 'info', 'test-xml', 'sample-data', 'list-instances'
    ], help='Command to execute')
    parser.add_argument('--instance', choices=list(DB_INSTANCES.keys()), 
                       help='Database instance name')
    parser.add_argument('--namespace', default='GlobeUserSignature',
                       help='Namespace for sample data (default: GlobeUserSignature)')
    parser.add_argument('--limit', type=int, default=3,
                       help='Limit for sample data (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    manager = MultiInstanceManager()
    
    try:
        if args.command == 'list-instances':
            print("\nAVAILABLE DATABASE INSTANCES:")
            print("="*60)
            for name, config in DB_INSTANCES.items():
                print(f"{name.upper():<10}: {config['description']}")
                print(f"            Project: {config['project']}")
                print(f"            Instance: {config['instance']}")
                print()
        
        elif args.command == 'test-all':
            manager.test_all_connections()
        
        elif args.command == 'test-connection':
            if not args.instance:
                print("Error: --instance required for test-connection command")
                sys.exit(1)
            connector = manager.get_connector(args.instance)
            success = connector.test_connection()
            print(f"\n{args.instance.upper()} connection test: {'✓ SUCCESS' if success else '✗ FAILED'}")
        
        elif args.command == 'info':
            if not args.instance:
                print("Error: --instance required for info command")
                sys.exit(1)
            manager.get_instance_info(args.instance)
        
        elif args.command == 'test-xml':
            if not args.instance:
                print("Error: --instance required for test-xml command")
                sys.exit(1)
            manager.test_xml_functionality(args.instance)
        
        elif args.command == 'sample-data':
            if not args.instance:
                print("Error: --instance required for sample-data command")
                sys.exit(1)
            manager.get_sample_data(args.instance, args.namespace, args.limit)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)
    finally:
        manager.cleanup_all()


# Convenience functions for direct usage
def get_connector(instance_name: str) -> DatabaseConnector:
    """Get database connector for specified instance"""
    return DatabaseConnector(instance_name)

def test_all_instances():
    """Quick test of all instances"""
    manager = MultiInstanceManager()
    return manager.test_all_connections()


if __name__ == "__main__":
    print("Multi-Instance Database Connector")
    print("Supports: OSCAR, Star, EDBCore, Copper instances")
    print("="*60)
    
    if len(sys.argv) == 1:
        print("\nUsage examples:")
        print("python database_connector.py list-instances")
        print("python database_connector.py test-all")
        print("python database_connector.py test-connection --instance oscar")
        print("python database_connector.py info --instance star")
        print("python database_connector.py test-xml --instance copper")
        print("python database_connector.py sample-data --instance edbcore --namespace GlobeUserSignature --limit 5")
        print("\nFor help: python database_connector.py -h")
        print("\nQuick test all instances:")
        test_all_instances()
    else:
        main()
