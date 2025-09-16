#!/usr/bin/env python3
"""
Enhanced Multi-Instance Database Connector for OSCAR Reconciliation Tool
Updated for simplified OSCAR-Copper reconciliation with proper table structure
"""

import logging
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import json

from google.cloud.sql.connector import Connector, create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from google.auth.transport import requests
import google.auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DB Instance configurations
DB_INSTANCES = {
    'oscar': {
        'project': 'prj-dv-oscar-8302',
        'region': 'us-central1',
        'instance': 'csql-dv-uscl-1316-oscar-8004-m',
        'database': 'pgdb',
        'user': 'lakshya.vijay@cmegroup.com',
        'schema': 'dv01cosrs',
        'description': 'OSCAR Database Instance'
    },
    'copper': {
        'project': 'prj-dv-copper-5609',
        'region': 'us-central1',
        'instance': 'csql-dv-uscl-747-refdata-0005-m',
        'database': 'pgdb',
        'user': 'lakshya.vijay@cmegroup.com',
        'schema': 'dv00ccrdb',
        'description': 'Copper Database Instance'
    }
}

class ReconciliationQueries:
    """Centralized query definitions for OSCAR-Copper reconciliation"""
    
    @staticmethod
    def get_oscar_guid_data(guid_id: str) -> str:
        """Get OSCAR data for a specific GUID"""
        return """
        SELECT 
            guid,
            (xpath('/globalUserSignature/globalUserSignatureInfo/idNumber/text()', xml))[1]::text as gus_id,
            (xpath('/globalUserSignature/globalUserSignatureInfo/globalFirmId/text()', xml))[1]::text as gfid,
            (xpath('/globalUserSignature/globalUserSignatureInfo/expDate/text()', xml))[1]::text as exp_date,
            CASE 
                WHEN (xpath('/globalUserSignature/globalUserSignatureInfo/expDate/text()', xml))[1]::text IS NULL 
                THEN 'MISSING'
                WHEN TO_DATE((xpath('/globalUserSignature/globalUserSignatureInfo/expDate/text()', xml))[1]::text, 'DD-MON-YYYY') >= CURRENT_DATE 
                THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM active_xml_data_store 
        WHERE guid = %(guid)s 
        AND namespace = 'globalUserSignature'
        """
    
    @staticmethod
    def get_oscar_gfid_gus_data(gfid: str, gus_id: str) -> str:
        """Get OSCAR data for GFID and GUS combination"""
        return """
        SELECT 
            guid,
            (xpath('/globalUserSignature/globalUserSignatureInfo/idNumber/text()', xml))[1]::text as gus_id,
            (xpath('/globalUserSignature/globalUserSignatureInfo/globalFirmId/text()', xml))[1]::text as gfid,
            (xpath('/globalUserSignature/globalUserSignatureInfo/expDate/text()', xml))[1]::text as exp_date,
            CASE 
                WHEN (xpath('/globalUserSignature/globalUserSignatureInfo/expDate/text()', xml))[1]::text IS NULL 
                THEN 'MISSING'
                WHEN TO_DATE((xpath('/globalUserSignature/globalUserSignatureInfo/expDate/text()', xml))[1]::text, 'DD-MON-YYYY') >= CURRENT_DATE 
                THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM active_xml_data_store 
        WHERE (xpath('/globalUserSignature/globalUserSignatureInfo/globalFirmId/text()', xml))[1]::text = %(gfid)s
        AND (xpath('/globalUserSignature/globalUserSignatureInfo/idNumber/text()', xml))[1]::text = %(gus_id)s
        AND namespace = 'globalUserSignature'
        """
    
    @staticmethod
    def get_copper_guid_data(guid_id: str) -> str:
        """Get Copper data for a specific GUID"""
        return """
        SELECT 
            guid,
            gpid_id as gus_id,
            gfid_id as gfid,
            eff_to,
            CASE 
                WHEN eff_to IS NULL THEN 'MISSING'
                WHEN eff_to >= CURRENT_DATE THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM trd_gpid 
        WHERE guid = %(guid)s
        """
    
    @staticmethod
    def get_copper_gfid_gus_data(gfid: str, gus_id: str) -> str:
        """Get Copper data for GFID and GUS (GPID) combination"""
        return """
        SELECT 
            guid,
            gpid_id as gus_id,
            gfid_id as gfid,
            eff_to,
            CASE 
                WHEN eff_to IS NULL THEN 'MISSING'
                WHEN eff_to >= CURRENT_DATE THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM trd_gpid 
        WHERE gfid_id = %(gfid)s AND gpid_id = %(gus_id)s
        """


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
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            logger.info(f"Initializing {self.config['description']} connector for {self.instance_connection_name}")
            
            # Test authentication
            credentials, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/sqlservice.admin"]
            )
            if credentials.has_non_expired_token():
                credentials.refresh(requests.Request())
            logger.info("Authentication successful")

            # Initialize connector
            self.connector = Connector()

            # Create SQLAlchemy engine
            self.engine = create_engine(
                "postgresql+pg8000://",
                creator=lambda: self.connector.connect(
                    self.instance_connection_name,
                    "pg8000",
                    user=self.user,
                    db=self.database,
                    enable_iam_auth=True,
                    ip_type="PUBLIC"
                ),
                pool_size=2,
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
                    logger.info(f"{self.instance_name.upper()} connection test successful")
                    return True
                else:
                    logger.error(f"{self.instance_name.upper()} connection test failed - unexpected result")
                    return False
        
        except Exception as e:
            logger.error(f"{self.instance_name.upper()} connection test failed: {e}")
            return False

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        if not self.engine:
            raise Exception("Database engine not initialized")
        
        try:
            logger.debug(f"Executing query on {self.instance_name}: {query[:100]}...")
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
                        value = row[i]
                        # Handle datetime objects
                        if isinstance(value, (datetime, date)):
                            value = value.isoformat()
                        row_dict[column] = value
                    rows.append(row_dict)

                logger.info(f"Query executed successfully on {self.instance_name}, returned {len(rows)} rows")
                return rows

        except Exception as e:
            logger.error(f"Query execution failed on {self.instance_name}: {e}")
            logger.error(f"Query: {query}")
            if params:
                logger.error(f"Params: {params}")
            raise

    def cleanup(self):
        """Clean up database connections"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info(f"{self.instance_name.upper()} database engine disposed")
            if self.connector:
                self.connector.close()
                logger.info(f"{self.instance_name.upper()} database connector closed")
        except Exception as e:
            logger.warning(f"{self.instance_name.upper()} cleanup warning: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class ReconciliationManager:
    """Manages reconciliation operations between OSCAR and Copper"""
    
    def __init__(self):
        self.oscar_connector = None
        self.copper_connector = None
        self.queries = ReconciliationQueries()
        
    def initialize_connections(self):
        """Initialize both database connections"""
        try:
            self.oscar_connector = DatabaseConnector('oscar')
            self.copper_connector = DatabaseConnector('copper')
            return True
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            return False
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test all database connections"""
        results = {}
        try:
            if self.oscar_connector:
                results['oscar'] = self.oscar_connector.test_connection()
            else:
                results['oscar'] = False
                
            if self.copper_connector:
                results['copper'] = self.copper_connector.test_connection()
            else:
                results['copper'] = False
                
        except Exception as e:
            logger.error(f"Connection testing failed: {e}")
            results = {'oscar': False, 'copper': False}
            
        return results
    
    def reconcile_by_guid(self, guid_id: str) -> Dict[str, Any]:
        """Reconcile data by GUID ID"""
        try:
            # Get OSCAR data
            oscar_query = self.queries.get_oscar_guid_data(guid_id)
            oscar_data = self.oscar_connector.execute_query(oscar_query, {'guid': guid_id})
            
            # Get Copper data
            copper_query = self.queries.get_copper_guid_data(guid_id)
            copper_data = self.copper_connector.execute_query(copper_query, {'guid': guid_id})
            
            # Perform comparison and determine scenario
            comparison_result = self._perform_comparison(oscar_data, copper_data, 'GUID', guid_id)
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"GUID reconciliation failed for {guid_id}: {e}")
            raise
    
    def reconcile_by_gfid_gus(self, gfid: str, gus_id: str) -> Dict[str, Any]:
        """Reconcile data by GFID and GUS ID combination"""
        try:
            # Get OSCAR data
            oscar_query = self.queries.get_oscar_gfid_gus_data(gfid, gus_id)
            oscar_data = self.oscar_connector.execute_query(oscar_query, {'gfid': gfid, 'gus_id': gus_id})
            
            # Get Copper data
            copper_query = self.queries.get_copper_gfid_gus_data(gfid, gus_id)
            copper_data = self.copper_connector.execute_query(copper_query, {'gfid': gfid, 'gus_id': gus_id})
            
            # Perform comparison and determine scenario
            comparison_result = self._perform_comparison(oscar_data, copper_data, 'GFID_GUS', f"{gfid}-{gus_id}")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"GFID-GUS reconciliation failed for {gfid}-{gus_id}: {e}")
            raise
    
    def _perform_comparison(self, oscar_data: List[Dict], copper_data: List[Dict], input_type: str, input_value: str) -> Dict[str, Any]:
        """Perform detailed comparison between OSCAR and Copper data"""
        
        oscar_found = len(oscar_data) > 0
        copper_found = len(copper_data) > 0
        
        oscar_record = oscar_data[0] if oscar_found else {}
        copper_record = copper_data[0] if copper_found else {}
        
        # Determine scenario type
        scenario = self._determine_scenario_type(oscar_found, copper_found, oscar_record, copper_record)
        
        # Format status with dates
        oscar_status = self._format_status_with_date(oscar_record, 'oscar')
        copper_status = self._format_status_with_date(copper_record, 'copper')
        
        return {
            'success': True,
            'input_value': input_value,
            'input_type': input_type,
            'oscar_data': {
                'found': oscar_found,
                'record': oscar_record,
                'status_with_date': oscar_status
            },
            'copper_data': {
                'found': copper_found,
                'record': copper_record,
                'status_with_date': copper_status
            },
            'scenario': scenario,
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_status_with_date(self, record: Dict, system: str) -> str:
        """Format status with appropriate date information"""
        if not record:
            return f"MISSING (as of {datetime.now().strftime('%Y-%m-%d')})"
        
        status = record.get('status', 'UNKNOWN')
        
        if system == 'oscar':
            date_field = record.get('exp_date', '')
        else:  # copper
            date_field = record.get('eff_to', '')
        
        if status == 'ACTIVE':
            return f"ACTIVE till {date_field}" if date_field else f"ACTIVE (as of {datetime.now().strftime('%Y-%m-%d')})"
        elif status == 'INACTIVE':
            return f"INACTIVE from {date_field}" if date_field else f"INACTIVE (as of {datetime.now().strftime('%Y-%m-%d')})"
        else:
            return f"MISSING (as of {datetime.now().strftime('%Y-%m-%d')})"
    
    def _determine_scenario_type(self, oscar_found: bool, copper_found: bool, oscar_record: Dict, copper_record: Dict) -> Dict[str, Any]:
        """Determine the reconciliation scenario and recommended actions"""
        
        if not oscar_found and not copper_found:
            return {
                'type': 'BOTH_MISSING',
                'description': 'Data not found in either system',
                'recommended_actions': ['Verify input value', 'Check data creation'],
                'severity': 'HIGH',
                'final_status': 'MISSING'
            }
        
        elif oscar_found and not copper_found:
            oscar_status = oscar_record.get('status', 'UNKNOWN')
            if oscar_status == 'ACTIVE':
                return {
                    'type': 'SCENARIO_2_4',
                    'description': 'Active data in OSCAR but missing in Copper',
                    'recommended_actions': ['Run Sync Job'],
                    'severity': 'HIGH',
                    'final_status': 'MISMATCH'
                }
            else:
                return {
                    'type': 'SCENARIO_2_3',
                    'description': 'Inactive data in OSCAR and missing in Copper',
                    'recommended_actions': ['Run Sync Flag'],
                    'severity': 'MEDIUM',
                    'final_status': 'MISMATCH'
                }
        
        elif not oscar_found and copper_found:
            return {
                'type': 'OSCAR_MISSING',
                'description': 'Data found in Copper but missing in OSCAR',
                'recommended_actions': ['Investigate OSCAR data', 'Manual sync required'],
                'severity': 'HIGH',
                'final_status': 'MISMATCH'
            }
        
        else:  # Both found - compare statuses
            oscar_status = oscar_record.get('status', 'UNKNOWN')
            copper_status = copper_record.get('status', 'UNKNOWN')
            
            if oscar_status == copper_status:
                return {
                    'type': 'SYNCHRONIZED',
                    'description': 'Data synchronized between systems',
                    'recommended_actions': ['No action required'],
                    'severity': 'LOW',
                    'final_status': 'MATCH'
                }
            else:
                if oscar_status == 'INACTIVE' and copper_status == 'ACTIVE':
                    return {
                        'type': 'SCENARIO_2_2',
                        'description': 'OSCAR inactive but Copper active',
                        'recommended_actions': ['Run Sync Job'],
                        'severity': 'MEDIUM',
                        'final_status': 'MISMATCH'
                    }
                else:
                    return {
                        'type': 'SCENARIO_2_1',
                        'description': 'Status mismatch between systems',
                        'recommended_actions': ['Run Sync Job', 'Run Sync Flag'],
                        'severity': 'MEDIUM',
                        'final_status': 'MISMATCH'
                    }
    
    def cleanup(self):
        """Clean up all database connections"""
        try:
            if self.oscar_connector:
                self.oscar_connector.cleanup()
            if self.copper_connector:
                self.copper_connector.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()




















#!/usr/bin/env python3
"""
Flask Application for OSCAR Reconciliation Tool
Simplified OSCAR-Copper reconciliation with updated table structure
"""

import sys
import os
import logging
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our enhanced database connector
from enhanced_database_connector import ReconciliationManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'oscar_reconciliation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'oscar-reconciliation-secret-key-change-in-production')

# CORS configuration
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000').split(',')
CORS(app, origins=cors_origins)

# Global reconciliation manager
reconciliation_manager = None

def initialize_system():
    """Initialize the reconciliation system"""
    global reconciliation_manager
    try:
        reconciliation_manager = ReconciliationManager()
        success = reconciliation_manager.initialize_connections()
        if success:
            logger.info("Reconciliation system initialized successfully")
            return True
        else:
            logger.error("Failed to initialize reconciliation system")
            return False
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False

@app.route('/')
def index():
    """Main dashboard page - serve the updated index.html"""
    return render_template('index.html')

@app.route('/api/health_check', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if reconciliation_manager:
            connection_status = reconciliation_manager.test_all_connections()
        else:
            connection_status = {'oscar': False, 'copper': False}
        
        overall_status = 'healthy' if all(connection_status.values()) else 'degraded'
        
        return jsonify({
            'status': overall_status,
            'timestamp': datetime.datetime.now().isoformat(),
            'database_status': connection_status,
            'system_initialized': reconciliation_manager is not None
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat(),
            'database_status': {'oscar': False, 'copper': False}
        }), 500

@app.route('/api/reconcile', methods=['POST'])
def reconcile_data():
    """Main reconciliation endpoint"""
    try:
        if not reconciliation_manager:
            return jsonify({
                'success': False,
                'error': 'Reconciliation system not initialized'
            }), 500
        
        data = request.get_json()
        
        # Extract and validate data
        guid = data.get('guid', '').strip().upper()
        gfid = data.get('gfid', '').strip().upper()
        gus_id = data.get('gus_id', '').strip().upper()
        
        # Validation logic: GUID can be alone, but GFID and GUS must be together
        if guid and (gfid or gus_id):
            return jsonify({
                'success': False,
                'error': 'Please provide either GUID alone OR both GFID and GUS ID together'
            }), 400
        
        if not guid and not (gfid and gus_id):
            return jsonify({
                'success': False,
                'error': 'Please provide either GUID alone OR both GFID and GUS ID together'
            }), 400
        
        if (gfid and not gus_id) or (gus_id and not gfid):
            return jsonify({
                'success': False,
                'error': 'GFID and GUS ID must be provided together'
            }), 400
        
        # Perform reconciliation
        if guid:
            result = reconciliation_manager.reconcile_by_guid(guid)
        else:
            result = reconciliation_manager.reconcile_by_gfid_gus(gfid, gus_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Reconciliation error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Reconciliation failed: {str(e)}"
        }), 500

@app.route('/api/execute_action', methods=['POST'])
def execute_action():
    """Execute reconciliation actions"""
    try:
        data = request.get_json()
        action = data.get('action')
        input_value = data.get('input_value')
        parameters = data.get('parameters', {})
        
        if not action:
            return jsonify({
                'success': False,
                'error': 'Action type is required'
            }), 400
        
        # Simulate action execution based on action type
        if 'Sync Job' in action:
            result = simulate_sync_job(input_value, parameters)
        elif 'Sync Flag' in action:
            result = simulate_sync_flag(input_value, parameters)
        elif 'Investigate' in action or 'investigation' in action.lower():
            result = simulate_investigation(input_value, parameters)
        elif 'Verify input value' in action:
            result = simulate_verification(input_value, parameters)
        elif 'Check data creation' in action:
            result = simulate_data_check(input_value, parameters)
        else:
            result = simulate_generic_action(action, input_value, parameters)
        
        result.update({
            'action': action,
            'input_value': input_value,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        logger.info(f"Executed action {action} for {input_value}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Action execution error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Action execution failed: {str(e)}"
        }), 500

def simulate_sync_job(input_value: str, parameters: Dict) -> Dict[str, Any]:
    """Simulate sync job operation"""
    import time
    import random
    
    # Simulate processing time
    time.sleep(1)
    
    records_processed = random.randint(5, 25)
    records_updated = random.randint(1, records_processed)
    
    return {
        'success': True,
        'message': 'Sync job completed successfully',
        'records_processed': records_processed,
        'records_updated': records_updated,
        'execution_time': '1.2 seconds',
        'details': f'Synchronized {records_updated} records out of {records_processed} processed for {input_value}'
    }

def simulate_sync_flag(input_value: str, parameters: Dict) -> Dict[str, Any]:
    """Simulate sync flag operation"""
    import time
    import random
    
    # Simulate processing time
    time.sleep(0.5)
    
    records_flagged = random.randint(1, 5)
    
    return {
        'success': True,
        'message': 'Sync flag operation completed successfully',
        'records_flagged': records_flagged,
        'execution_time': '0.8 seconds',
        'details': f'Set sync flag for {records_flagged} records related to {input_value}'
    }

def simulate_investigation(input_value: str, parameters: Dict) -> Dict[str, Any]:
    """Simulate investigation process"""
    return {
        'success': True,
        'message': 'Investigation ticket created',
        'ticket_id': f'INV-{datetime.datetime.now().strftime("%Y%m%d")}-{input_value}',
        'assigned_to': 'Data Operations Team',
        'priority': 'High',
        'details': f'Investigation ticket created for {input_value} data discrepancy'
    }

def simulate_verification(input_value: str, parameters: Dict) -> Dict[str, Any]:
    """Simulate input verification"""
    return {
        'success': True,
        'message': 'Input verification completed',
        'input_format': 'Valid',
        'input_length': len(input_value),
        'details': f'Verified {input_value} format and structure'
    }

def simulate_data_check(input_value: str, parameters: Dict) -> Dict[str, Any]:
    """Simulate data creation check"""
    return {
        'success': True,
        'message': 'Data creation check completed',
        'last_created': '2024-01-15 14:30:00',
        'source_system': 'OSCAR',
        'creation_status': 'Pending approval',
        'details': f'Data creation timeline verified for {input_value}'
    }

def simulate_generic_action(action: str, input_value: str, parameters: Dict) -> Dict[str, Any]:
    """Simulate generic action"""
    import time
    time.sleep(0.8)
    
    return {
        'success': True,
        'message': f'{action} completed successfully',
        'execution_time': '0.8 seconds',
        'details': f'Action "{action}" executed for {input_value}'
    }

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Initialize the system
    logger.info("Starting OSCAR Reconciliation Tool")
    
    if initialize_system():
        logger.info("System initialized successfully - starting Flask application")
        
        # Get configuration from environment
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('DEBUG', 'True').lower() == 'true'
        
        app.run(debug=debug, host=host, port=port)
    else:
        logger.error("Failed to initialize system - exiting")
        sys.exit(1)






















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSCAR Reconciliation Tool</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/animation.css') }}">
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-exchange-alt"></i>
                    <h1>OSCAR Reconciliation Tool</h1>
                </div>
                <div class="connection-status">
                    <span class="status-dot" id="connection-status"></span>
                    <span id="connection-text">Checking...</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <div class="container">
            <!-- Welcome Section -->
            <section class="welcome-section fade-in">
                <div class="welcome-content">
                    <h2>Data Reconciliation Between OSCAR & CoPPER</h2>
                    <p>Compare and synchronize financial trading data across systems</p>
                </div>
            </section>

            <!-- Search Section -->
            <section class="search-section slide-up">
                <div class="search-card">
                    <div class="search-header">
                        <h3><i class="fas fa-search"></i> Reconciliation Parameters</h3>
                        <p>Enter GUID alone OR both GFID and GUS ID together</p>
                    </div>

                    <form id="reconcile-form">
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="guid"><i class="fas fa-key"></i> GUID</label>
                                <input type="text" id="guid" name="guid" placeholder="e.g., XFSHSYBIKCLZ">
                                <div class="input-info">Global Unique Identifier (can be used alone)</div>
                            </div>
                            <div class="form-group">
                                <label for="gfid"><i class="fas fa-building"></i> GFID</label>
                                <input type="text" id="gfid" name="gfid" placeholder="e.g., SAT0">
                                <div class="input-info">Globex Firm ID (required with GUS ID)</div>
                            </div>
                            <div class="form-group">
                                <label for="gus-id"><i class="fas fa-user"></i> GUS ID</label>
                                <input type="text" id="gus-id" name="gus_id" placeholder="e.g., 123X">
                                <div class="input-info">Globex User Signature ID (required with GFID)</div>
                            </div>
                        </div>
                        
                        <div class="scenario-selector">
                            <h4><i class="fas fa-cogs"></i> Comparison Options</h4>
                            <div class="scenario-grid">
                                <div class="form-group">
                                    <label for="comparison-type">Primary Comparison</label>
                                    <select id="comparison-type" name="comparison_type">
                                        <option value="standard_lookup">Standard Lookup</option>
                                        <option value="scenario_2_1">Scenario 2.1 - Both Expired</option>
                                        <option value="scenario_2_2">Scenario 2.2 - OSCAR Expired, COPPER Active</option>
                                        <option value="scenario_2_3">Scenario 2.3 - OSCAR Expired, COPPER Missing</option>
                                        <option value="scenario_2_4">Scenario 2.4 - OSCAR Active, COPPER Missing</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="submit-container">
                            <button type="submit" class="btn-primary" id="submit-btn">
                                <i class="fas fa-sync-alt"></i>
                                <span>Execute Reconciliation</span>
                            </button>
                        </div>
                    </form>
                </div>
            </section>
            
            <!-- Results Section -->
            <section class="results-section" id="results-section">
                <div class="results-header">
                    <h3><i class="fas fa-chart-bar"></i> Reconciliation Results</h3>
                </div>

                <!-- Scenario Information Card -->
                <div id="scenario-info" class="scenario-info" style="display: none;">
                    <div class="scenario-header">
                        <h4 class="scenario-title"></h4>
                        <span class="severity-badge"></span>
                    </div>
                    <div class="scenario-content">
                        <p><strong>Input:</strong> <span id="scenario-input"></span></p>
                        <p><strong>Description:</strong> <span id="scenario-description"></span></p>
                        <div id="recommended-actions" class="recommended-actions">
                            <h5>Recommended Actions:</h5>
                            <div id="action-buttons"></div>
                        </div>
                    </div>
                </div>

                <!-- Comparison Table -->
                <div class="comparison-tables">
                    <div class="table-section">
                        <div class="table-header">
                            <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER Comparison</h4>
                        </div>
                        <div class="table-container">
                            <table class="comparison-table" id="main-comparison-table">
                                <thead>
                                    <tr>
                                        <th rowspan="2">System</th>
                                        <th class="oscar-col">GUID</th>
                                        <th class="oscar-col">GFID</th>
                                        <th class="oscar-col">GUS ID</th>
                                        <th class="oscar-col">Status</th>
                                        <th rowspan="2">Final Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr id="oscar-row">
                                        <td class="oscar-col"><strong>OSCAR</strong></td>
                                        <td class="oscar-col" id="oscar-guid">-</td>
                                        <td class="oscar-col" id="oscar-gfid">-</td>
                                        <td class="oscar-col" id="oscar-gus">-</td>
                                        <td class="oscar-col" id="oscar-status">-</td>
                                        <td rowspan="2" id="final-status-cell">
                                            <span class="status-badge" id="final-status-badge">PROCESSING</span>
                                        </td>
                                    </tr>
                                    <tr id="copper-row">
                                        <td class="copper-col"><strong>CoPPER</strong></td>
                                        <td class="copper-col" id="copper-guid">-</td>
                                        <td class="copper-col" id="copper-gfid">-</td>
                                        <td class="copper-col" id="copper-gus">-</td>
                                        <td class="copper-col" id="copper-status">-</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <script src="{{ url_for('static', filename='js/dynamic_javascript_original_ui.js') }}"></script>
</body>
</html>




















// Dynamic JavaScript for OSCAR Reconciliation Tool
// Updated for simplified OSCAR-Copper reconciliation with validation

document.addEventListener("DOMContentLoaded", function() {
    // Get DOM elements
    const form = document.getElementById("reconcile-form");
    const resultsSection = document.getElementById("results-section");
    const submitBtn = document.getElementById("submit-btn");
    
    // Form input elements
    const guidInput = document.getElementById("guid");
    const gfidInput = document.getElementById("gfid");
    const gusInput = document.getElementById("gus-id");
    
    // Results elements
    const scenarioInfo = document.getElementById("scenario-info");
    const scenarioTitle = document.querySelector(".scenario-title");
    const scenarioInput = document.getElementById("scenario-input");
    const scenarioDescription = document.getElementById("scenario-description");
    const severityBadge = document.querySelector(".severity-badge");
    const actionButtons = document.getElementById("action-buttons");
    
    // Table elements
    const oscarGuid = document.getElementById("oscar-guid");
    const oscarGfid = document.getElementById("oscar-gfid");
    const oscarGus = document.getElementById("oscar-gus");
    const oscarStatus = document.getElementById("oscar-status");
    const copperGuid = document.getElementById("copper-guid");
    const copperGfid = document.getElementById("copper-gfid");
    const copperGus = document.getElementById("copper-gus");
    const copperStatus = document.getElementById("copper-status");
    const finalStatusBadge = document.getElementById("final-status-badge");

    // Initialize system
    checkSystemHealth();
    
    // Check system health every 5 minutes
    setInterval(checkSystemHealth, 300000);

    // Form submission handler
    if (form) {
        form.addEventListener("submit", function(e) {
            e.preventDefault();
            executeReconciliation();
        });
    }

    // Input validation handlers
    guidInput.addEventListener('input', handleGuidInput);
    gfidInput.addEventListener('input', handleGfidGusInput);
    gusInput.addEventListener('input', handleGfidGusInput);

    function handleGuidInput() {
        if (guidInput.value.trim()) {
            // If GUID is entered, clear and disable GFID/GUS
            gfidInput.value = '';
            gusInput.value = '';
            gfidInput.disabled = true;
            gusInput.disabled = true;
            showInputMessage(guidInput, 'GUID mode - GFID and GUS disabled', 'info');
        } else {
            // Enable GFID/GUS if GUID is empty
            gfidInput.disabled = false;
            gusInput.disabled = false;
            clearInputMessage(guidInput);
        }
    }

    function handleGfidGusInput() {
        const gfidValue = gfidInput.value.trim();
        const gusValue = gusInput.value.trim();
        
        if (gfidValue || gusValue) {
            // If either GFID or GUS is entered, disable GUID
            guidInput.value = '';
            guidInput.disabled = true;
            
            // Check if both are filled
            if (gfidValue && gusValue) {
                showInputMessage(gfidInput, 'GFID + GUS mode - Both required', 'success');
                showInputMessage(gusInput, 'GFID + GUS mode - Both required', 'success');
            } else if (gfidValue) {
                showInputMessage(gfidInput, 'GUS ID also required', 'warning');
                clearInputMessage(gusInput);
            } else if (gusValue) {
                showInputMessage(gusInput, 'GFID also required', 'warning');
                clearInputMessage(gfidInput);
            }
        } else {
            // Enable GUID if both GFID/GUS are empty
            guidInput.disabled = false;
            clearInputMessage(gfidInput);
            clearInputMessage(gusInput);
        }
    }

    function showInputMessage(input, message, type) {
        const infoDiv = input.nextElementSibling;
        if (infoDiv && infoDiv.classList.contains('input-info')) {
            infoDiv.textContent = message;
            infoDiv.className = `input-info ${type}`;
        }
    }

    function clearInputMessage(input) {
        const infoDiv = input.nextElementSibling;
        if (infoDiv && infoDiv.classList.contains('input-info')) {
            // Reset to original message based on input type
            if (input.id === 'guid') {
                infoDiv.textContent = 'Global Unique Identifier (can be used alone)';
            } else if (input.id === 'gfid') {
                infoDiv.textContent = 'Globex Firm ID (required with GUS ID)';
            } else if (input.id === 'gus-id') {
                infoDiv.textContent = 'Globex User Signature ID (required with GFID)';
            }
            infoDiv.className = 'input-info';
        }
    }

    async function checkSystemHealth() {
        try {
            const response = await fetch('/api/health_check');
            const healthData = await response.json();
            
            updateConnectionStatus(healthData);
            
        } catch (error) {
            console.error('Health check failed:', error);
            updateConnectionStatus({ 
                status: 'unhealthy', 
                database_status: { oscar: false, copper: false } 
            });
        }
    }

    function updateConnectionStatus(healthData) {
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        
        if (statusDot && statusText) {
            if (healthData.status === 'healthy') {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Connected';
            } else {
                statusDot.className = 'status-dot disconnected';
                statusText.textContent = 'Degraded';
            }
        }
    }

    async function executeReconciliation() {
        // Show loading state
        setLoadingState(true);
        
        try {
            // Get and validate form data
            const formData = getFormData();
            const validation = validateFormData(formData);
            
            if (!validation.valid) {
                showToast(validation.message, 'error');
                setLoadingState(false);
                return;
            }

            // Execute reconciliation
            const response = await fetch('/api/reconcile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                // Store result for any future operations
                window.lastReconciliationData = result;
                
                // Update UI with results
                updateResultsDisplay(result);
                
                // Show results section
                if (resultsSection) {
                    resultsSection.classList.add('show');
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                }
                
                showToast('Reconciliation completed successfully', 'success');
            } else {
                showToast(result.error || 'Reconciliation failed', 'error');
            }

        } catch (error) {
            console.error('Reconciliation error:', error);
            showToast('Network error - please check connection', 'error');
        } finally {
            setLoadingState(false);
        }
    }

    function getFormData() {
        return {
            guid: guidInput.value.trim().upper() || '',
            gfid: gfidInput.value.trim().upper() || '',
            gus_id: gusInput.value.trim().upper() || '',
            comparison_type: document.getElementById('comparison-type')?.value || 'standard_lookup'
        };
    }

    function validateFormData(data) {
        // Validation: GUID alone OR both GFID and GUS together
        if (data.guid && (data.gfid || data.gus_id)) {
            return {
                valid: false,
                message: 'Please provide either GUID alone OR both GFID and GUS ID together'
            };
        }
        
        if (!data.guid && !data.gfid && !data.gus_id) {
            return {
                valid: false,
                message: 'Please provide either GUID alone OR both GFID and GUS ID together'
            };
        }
        
        if ((data.gfid && !data.gus_id) || (data.gus_id && !data.gfid)) {
            return {
                valid: false,
                message: 'GFID and GUS ID must be provided together'
            };
        }

        return { valid: true, message: 'Valid' };
    }

    function updateResultsDisplay(result) {
        // Update scenario information
        updateScenarioInfo(result);
        
        // Update comparison table
        updateComparisonTable(result);
        
        // Show scenario info
        if (scenarioInfo) {
            scenarioInfo.style.display = 'block';
        }
    }

    function updateScenarioInfo(result) {
        if (scenarioTitle) {
            scenarioTitle.textContent = `${result.scenario.type}: ${result.scenario.description}`;
        }
        
        if (scenarioInput) {
            scenarioInput.textContent = `${result.input_value} (${result.input_type})`;
        }
        
        if (scenarioDescription) {
            scenarioDescription.textContent = result.scenario.description;
        }
        
        if (severityBadge) {
            severityBadge.textContent = result.scenario.severity;
            severityBadge.className = `severity-badge ${result.scenario.severity.toLowerCase()}`;
        }
        
        // Update action buttons
        if (actionButtons) {
            actionButtons.innerHTML = '';
            result.scenario.recommended_actions.forEach(action => {
                const button = document.createElement('button');
                button.className = 'action-btn';
                button.innerHTML = `<i class="fas fa-cogs"></i> ${action}`;
                button.addEventListener('click', () => executeAction(action, result.input_value));
                actionButtons.appendChild(button);
            });
        }
    }

    function updateComparisonTable(result) {
        // Update OSCAR row
        const oscarRecord = result.oscar_data.record || {};
        oscarGuid.textContent = oscarRecord.guid || '-';
        oscarGfid.textContent = oscarRecord.gfid || '-';
        oscarGus.textContent = oscarRecord.gus_id || '-';
        oscarStatus.textContent = result.oscar_data.status_with_date || 'Not Found';
        
        // Update Copper row
        const copperRecord = result.copper_data.record || {};
        copperGuid.textContent = copperRecord.guid || '-';
        copperGfid.textContent = copperRecord.gfid || '-';
        copperGus.textContent = copperRecord.gus_id || '-';
        copperStatus.textContent = result.copper_data.status_with_date || 'Not Found';
        
        // Update final status
        updateFinalStatusBadge(result.scenario.final_status);
    }

    function updateFinalStatusBadge(status) {
        finalStatusBadge.textContent = status;
        finalStatusBadge.className = 'status-badge';
        
        switch (status) {
            case 'MATCH':
                finalStatusBadge.classList.add('status-match');
                break;
            case 'MISMATCH':
                finalStatusBadge.classList.add('status-mismatch');
                break;
            case 'MISSING':
                finalStatusBadge.classList.add('status-missing');
                break;
            default:
                finalStatusBadge.classList.add('status-mismatch');
        }
    }

    async function executeAction(action, inputValue) {
        try {
            setLoadingState(true, `Executing ${action}...`);
            
            const response = await fetch('/api/execute_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: action,
                    input_value: inputValue
                })
            });

            const result = await response.json();

            if (result.success) {
                showActionResult(result);
                showToast(result.message || `${action} completed successfully`, 'success');
            } else {
                showToast(result.error || `${action} failed`, 'error');
            }

        } catch (error) {
            console.error('Action execution error:', error);
            showToast(`Failed to execute ${action}`, 'error');
        } finally {
            setLoadingState(false);
        }
    }

    function showActionResult(result) {
        // Create a detailed result modal
        const resultModal = document.createElement('div');
        resultModal.className = 'action-result-modal';
        resultModal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-check-circle"></i> Action Completed</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <h4>${result.action}</h4>
                    <p><strong>Status:</strong> ${result.message}</p>
                    ${result.details ? `<p><strong>Details:</strong> ${result.details}</p>` : ''}
                    ${result.execution_time ? `<p><strong>Execution Time:</strong> ${result.execution_time}</p>` : ''}
                    ${result.records_processed ? `<p><strong>Records Processed:</strong> ${result.records_processed}</p>` : ''}
                    ${result.records_updated ? `<p><strong>Records Updated:</strong> ${result.records_updated}</p>` : ''}
                    ${result.records_flagged ? `<p><strong>Records Flagged:</strong> ${result.records_flagged}</p>` : ''}
                    ${result.ticket_id ? `<p><strong>Ticket ID:</strong> ${result.ticket_id}</p>` : ''}
                    <p><strong>Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
                </div>
            </div>
        `;

        document.body.appendChild(resultModal);

        // Close modal functionality
        const closeBtn = resultModal.querySelector('.modal-close');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(resultModal);
        });

        // Auto-close after 15 seconds
        setTimeout(() => {
            if (document.body.contains(resultModal)) {
                document.body.removeChild(resultModal);
            }
        }, 15000);
    }

    function setLoadingState(loading, message = 'Processing reconciliation...') {
        if (submitBtn) {
            if (loading) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitBtn.disabled = true;
            } else {
                submitBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Execute Reconciliation';
                submitBtn.disabled = false;
            }
        }
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <i class="fas fa-${getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Style the toast
        Object.assign(toast.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            background: 'var(--bg-card)',
            color: 'var(--white)',
            padding: '1rem 1.5rem',
            borderRadius: '0.5rem',
            boxShadow: 'var(--shadow-xl)',
            zIndex: '1000',
            borderLeft: `4px solid ${getToastColor(type)}`,
            maxWidth: '400px',
            animation: 'slideInRight 0.3s ease-out'
        });
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            }, 300);
        }, 4000);
    }

    function getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }

    function getToastColor(type) {
        const colors = {
            success: 'var(--success-color)',
            error: 'var(--error-color)',
            warning: 'var(--warning-color)',
            info: 'var(--accent-color)'
        };
        return colors[type] || colors.info;
    }

    // Add CSS for new components if not already present
    if (!document.querySelector('#dynamic-styles-updated')) {
        const style = document.createElement('style');
        style.id = 'dynamic-styles-updated';
        style.textContent = `
            .input-info.success { color: var(--success-color); font-weight: 600; }
            .input-info.warning { color: var(--warning-color); font-weight: 600; }
            .input-info.info { color: var(--accent-color); font-weight: 600; }
            
            input:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                background: var(--gray-700) !important;
            }
            
            .action-result-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 2000;
                animation: fadeIn 0.3s ease-out;
            }
            
            .modal-content {
                background: var(--bg-card);
                border-radius: var(--radius-xl);
                max-width: 600px;
                width: 90%;
                max-height: 80vh;
                overflow: auto;
                box-shadow: var(--shadow-xl);
                border: 2px solid var(--primary-color);
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: var(--spacing-xl);
                border-bottom: 2px solid var(--primary-color);
                background: var(--gradient-primary);
            }
            
            .modal-header h3 {
                margin: 0;
                color: var(--white);
                font-size: var(--font-size-lg);
            }
            
            .modal-close {
                background: none;
                border: none;
                font-size: 1.5rem;
                cursor: pointer;
                color: var(--white);
                padding: var(--spacing-xs);
                border-radius: var(--radius-sm);
                transition: var(--transition-normal);
            }
            
            .modal-close:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            
            .modal-body {
                padding: var(--spacing-xl);
                color: var(--white);
            }
            
            .modal-body h4 {
                color: var(--accent-color);
                margin-bottom: var(--spacing-lg);
                font-size: var(--font-size-lg);
            }
            
            .modal-body p {
                margin-bottom: var(--spacing-sm);
                line-height: 1.6;
            }
            
            .modal-body strong {
                color: var(--white);
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
});




















