# enhanced_database_connector.py v2.0
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date

from google.cloud.sql.connector import Connector
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import google.auth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database instance configurations
DB_INSTANCES = {
    'oscar': {
        'instance': 'prj-dr-oscar-dss2:europe-west1:scl-e-oscrdss-3316-oscar-dss-a',
        'user': 'oscar.online@prj-dr-oscar-dss2.iam',
        'project': 'prj-dr-oscar-dss2',
        'region': 'europe-west1',
        'schema': 'g01cicrs',
        'driver': 'pg8000',
        'description': "OSCAR Database"
    },
    'copper': {
        'instance': 'prj-c-it-dwh-stg-a:europe-west1:scl-s-dwh-postg-747-refdata-0001-a',
        'user': 'copper.online@prj-c-it-dwh-stg-a.iam',
        'project': 'prj-c-it-dwh-stg-a',
        'region': 'europe-west1',
        'schema': 'g01ds2ordr',
        'driver': 'pg8000',
        'description': "Copper Database"
    }
}


class ReconciliationQueries:
    """SQL query definitions for reconciliation"""

    @staticmethod
    def get_oscar_data_by_guid(guid: str) -> str:
        """Get OSCAR data for a specific GUID"""
        return f"""
        SELECT
            guid,
            (xpath('/globexUserSignature/globexUserSignatureInfo/globexFirmId/text()', guses.XML))[1]::text as "GFID",
            (xpath('/globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text as "GUS_ID",
            (xpath('/globexUserSignature/globexUserSignatureInfo/exchange/text()', guses.XML))[1]::text as "Exchange",
            namespace,
            CASE
                WHEN (xpath('/globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.XML))[1]::text IS NULL
                THEN 'MISSING'
                WHEN TO_DATE((xpath('/globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.XML))[1]::text, 'DD-MON-YYYY') >= CURRENT_DATE
                THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM g01cicrs.active_xml_data_store guses
        WHERE guses.GUID = '{guid}'
        AND (guses.namespace = 'GlobexUserSignature')
        LIMIT 1
        """

    @staticmethod
    def get_oscar_gfid_gus_data(gfid: str, gus_id: str) -> str:
        """Get OSCAR data for GFID and GUS combination"""
        return f"""
        SELECT
            guid,
            (xpath('/globexUserSignature/globexUserSignatureInfo/globexFirmId/text()', guses.XML))[1]::text as "GFID",
            (xpath('/globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text as "GUS_ID",
            (xpath('/globexUserSignature/globexUserSignatureInfo/exchange/text()', guses.XML))[1]::text as "Exchange",
            namespace,
            CASE
                WHEN (xpath('/globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.XML))[1]::text IS NULL
                THEN 'MISSING'
                WHEN TO_DATE((xpath('/globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.XML))[1]::text, 'DD-MON-YYYY') >= CURRENT_DATE
                THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM g01cicrs.active_xml_data_store guses
        WHERE (xpath('/globexUserSignature/globexUserSignatureInfo/globexFirmId/text()', guses.XML))[1]::text = '{gfid}'
        AND (xpath('/globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text = '{gus_id}'
        AND namespace = 'GlobexUserSignature'
        LIMIT 1
        """

    @staticmethod
    def get_copper_data_by_guid(guid: str) -> str:
        """Get Copper data for a specific GUID"""
        return f"""
            SELECT
                guid,
                guid_id as gus_id,
                gfid_id as gfid,
                eff_to,
                CASE
                    WHEN eff_to IS NULL THEN 'MISSING'
                    WHEN eff_to >= CURRENT_DATE THEN 'ACTIVE'
                    ELSE 'INACTIVE'
                END as status
            FROM g01ds2ordr.trd_guid
            WHERE guid = '{guid}'
        """

    @staticmethod
    def get_copper_gfid_gus_data(gfid: str, gus_id: str) -> str:
        """Get Copper data for GFID and GUS combination"""
        return f"""
            SELECT
                guid,
                guid_id as gus_id,
                gfid_id as gfid,
                eff_to,
                CASE
                    WHEN eff_to IS NULL THEN 'MISSING'
                    WHEN eff_to >= CURRENT_DATE THEN 'ACTIVE'
                    ELSE 'INACTIVE'
                END as status
            FROM g01ds2ordr.trd_guid
            WHERE gfid_id = '{gfid}' AND guid_id = '{gus_id}'
        """


class DatabaseConnector:
    """Manages connection to a database instance using Cloud SQL Connector"""

    def __init__(self, instance_name: str):
        """Initialize database connector"""
        if instance_name not in DB_INSTANCES:
            raise ValueError(f"Unknown instance: {instance_name}. Available: {list(DB_INSTANCES.keys())}")

        self.instance_name = instance_name
        self.config = DB_INSTANCES[instance_name]
        self.connector = None
        self.engine = None

    def _initialize_connection(self):
        """Initialize database connection"""
        try:
            logger.info(f"Initializing connection for {self.config['description']}")

            # Verify Google Cloud authentication
            credentials, project = google.auth.default()
            logger.info(f"Google Auth credentials loaded for project: {project}")

            # Initialize Cloud SQL Connector
            self.connector = Connector()

            # Create SQLAlchemy engine
            self.engine = create_engine(
                f"postgresql+{self.config['driver']}://",
                creator=lambda: self.connector.connect(
                    self.config['instance'],
                    self.config['driver'],
                    user=self.config['user'],
                    enable_iam_auth=True,
                ),
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            logger.info(f"{self.instance_name.upper()} database engine created successfully")

        except Exception as e:
            logger.error(f"Database initialization failed for {self.instance_name}: {e}")
            raise

    def test_connection(self) -> bool:
        """Test database connection"""
        if not self.engine:
            try:
                self._initialize_connection()
            except Exception:
                return False

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                if row and row[0] == 1:
                    logger.info(f"{self.instance_name.upper()} connection test successful")
                    return True
                return False
        except Exception as e:
            logger.error(f"{self.instance_name.upper()} connection test failed: {e}")
            return False

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        if not self.engine:
            self._initialize_connection()

        try:
            logger.debug(f"Executing query on {self.instance_name}: {query[:100]}...")

            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]

                # Convert datetime/date objects to ISO format strings
                for row in rows:
                    for key, value in row.items():
                        if isinstance(value, (datetime, date)):
                            row[key] = value.isoformat()

                logger.info(f"Query executed successfully on {self.instance_name}, returned {len(rows)} row(s)")
                return rows

        except Exception as e:
            logger.error(f"Query execution failed on {self.instance_name}: {e}")
            raise Exception(f"Query error on {self.instance_name}: {e}")

    def close(self):
        """Close database connections"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info(f"{self.instance_name.upper()} database engine disposed")
            if self.connector:
                self.connector.close()
                logger.info(f"{self.instance_name.upper()} database connector closed")
        except Exception as e:
            logger.warning(f"{self.instance_name.upper()} cleanup warning: {e}")


class ReconciliationManager:
    """Handles reconciliation operations between OSCAR and Copper"""

    def __init__(self):
        self.oscar_connector = None
        self.copper_connector = None
        self.queries = ReconciliationQueries()

    def initialize_connections(self) -> bool:
        """Initialize database connections"""
        try:
            self.oscar_connector = DatabaseConnector('oscar')
            self.copper_connector = DatabaseConnector('copper')
            return True
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            return False

    def test_all_connections(self) -> Dict[str, bool]:
        """Test all database connections"""
        results = {'oscar': False, 'copper': False}
        try:
            if self.oscar_connector:
                results['oscar'] = self.oscar_connector.test_connection()
            if self.copper_connector:
                results['copper'] = self.copper_connector.test_connection()
        except Exception as e:
            logger.error(f"Connection testing failed: {e}")
        return results

    def reconcile_by_guid(self, guid: str) -> Dict[str, Any]:
        """Reconcile data by GUID"""
        try:
            oscar_query = self.queries.get_oscar_data_by_guid(guid)
            copper_query = self.queries.get_copper_data_by_guid(guid)

            oscar_data = self.oscar_connector.execute_query(oscar_query)
            copper_data = self.copper_connector.execute_query(copper_query)

            return self._perform_comparison(oscar_data, copper_data, 'guid', guid)

        except Exception as e:
            logger.error(f"GUID reconciliation failed for {guid}: {e}")
            raise

    def reconcile_by_gfid_gus(self, gfid: str, gus_id: str) -> Dict[str, Any]:
        """Reconcile data by GFID and GUS ID"""
        try:
            oscar_query = self.queries.get_oscar_gfid_gus_data(gfid, gus_id)
            copper_query = self.queries.get_copper_gfid_gus_data(gfid, gus_id)

            oscar_data = self.oscar_connector.execute_query(oscar_query)
            copper_data = self.copper_connector.execute_query(copper_query)

            return self._perform_comparison(oscar_data, copper_data, 'gfid-gus', f"{gfid}-{gus_id}")

        except Exception as e:
            logger.error(f"GFID-GUS reconciliation failed for {gfid}-{gus_id}: {e}")
            raise

    def _perform_comparison(self, oscar_data: List[Dict], copper_data: List[Dict], 
                           input_type: str, input_value: str) -> Dict[str, Any]:
        """Perform detailed comparison between OSCAR and Copper data"""
        oscar_record = oscar_data[0] if oscar_data else None
        copper_record = copper_data[0] if copper_data else None

        oscar_found = bool(oscar_record)
        copper_found = bool(copper_record)

        scenario = self._determine_scenario(oscar_found, copper_found, oscar_record, copper_record)

        return {
            'success': True,
            'input_type': input_type,
            'input_value': input_value,
            'reconciliation_timestamp': datetime.now().isoformat(),
            'oscar_data_record': {
                'found': oscar_found,
                'status': oscar_record.get('status') if oscar_record else 'NOT_FOUND',
                'GFID': oscar_record.get('GFID') if oscar_record else 'N/A',
                'GUS_ID': oscar_record.get('GUS_ID') if oscar_record else 'N/A',
                'guid': oscar_record.get('guid') if oscar_record else 'N/A',
                'full_data': oscar_record
            },
            'cofdm_data_record': {
                'found': copper_found,
                'status': copper_record.get('status') if copper_record else 'NOT_FOUND',
                'gfid': copper_record.get('gfid') if copper_record else 'N/A',
                'gus_id': copper_record.get('gus_id') if copper_record else 'N/A',
                'guid': copper_record.get('guid') if copper_record else 'N/A',
                'eff_to': copper_record.get('eff_to') if copper_record else 'N/A',
                'full_data': copper_record
            },
            'scenario': scenario
        }

    def _determine_scenario(self, oscar_found: bool, copper_found: bool, 
                           oscar_record: Optional[Dict], copper_record: Optional[Dict]) -> Dict[str, Any]:
        """Determine reconciliation scenario and recommended actions"""
        
        if not oscar_found and not copper_found:
            return {
                'type': 'BOTH_MISSING',
                'description': 'Data not found in either system',
                'recommended_actions': ['Verify input value', 'Check data creation'],
                'severity': 'HIGH',
                'final_status': 'MISMATCH'
            }

        if oscar_found and not copper_found:
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

        if not oscar_found and copper_found:
            return {
                'type': 'OSCAR_MISSING',
                'description': 'Data found in Copper but missing in OSCAR',
                'recommended_actions': ['Investigate OSCAR data', 'Manual sync required'],
                'severity': 'HIGH',
                'final_status': 'MISMATCH'
            }

        # Both found - compare statuses
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
                    'description': 'OSCAR Inactive but Copper active',
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

    def close(self):
        """Close all database connections"""
        try:
            if self.oscar_connector:
                self.oscar_connector.close()
            if self.copper_connector:
                self.copper_connector.close()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


















# app.py
import os
import logging
import traceback
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from typing import Dict, Any
from dotenv import load_dotenv
import io

from enhanced_database_connector import ReconciliationManager

load_dotenv()

# Configure logging
log_file = os.getenv('LOG_FILE', 'reconciliation.log')
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter(log_format))

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))

logger = logging.getLogger(__name__)
logger.setLevel(log_level)
logger.addHandler(handler)
logger.addHandler(stream_handler)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'reconciliation-secret-key-change-in-prod')

# CORS configuration
origins = os.getenv('CORS_ORIGINS', 'http://localhost:8050,http://127.0.0.1:8050').split(',')
CORS(app, origins=origins, supports_credentials=True)

# Global reconciliation manager
reconciliation_manager = None
initialization_system = False


def initialize_system():
    """Initialize the reconciliation system"""
    global reconciliation_manager, initialization_system
    logger.info("Starting Reconciliation system")
    try:
        reconciliation_manager = ReconciliationManager()
        success = reconciliation_manager.initialize_connections()
        if success:
            logger.info("Reconciliation system initialized successfully")
            initialization_system = True
            return True
        else:
            logger.error("Failed to initialize reconciliation system")
            return False
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/health_check', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if reconciliation_manager:
            connection_status = reconciliation_manager.test_all_connections()
            overall_status = 'healthy' if all(connection_status.values()) else 'degraded'
        else:
            connection_status = {}
            overall_status = 'unhealthy'

        return jsonify({
            'success': True,
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'system_initialized': initialization_system,
            'oscar_status': 'healthy' if connection_status.get('oscar') else 'error',
            'cofdm_status': 'healthy' if connection_status.get('copper') else 'error'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'system_initialized': initialization_system,
            'oscar_status': 'error',
            'cofdm_status': 'error'
        }), 500


@app.route('/api/reconcile', methods=['POST'])
def reconcile():
    """Main reconciliation endpoint"""
    global reconciliation_manager
    
    if not reconciliation_manager:
        return jsonify({
            'success': False,
            'error': 'Reconciliation system not initialized'
        }), 500

    data = request.get_json()

    # Extract input parameters
    recon_id = data.get('recon_id', '').strip().upper()  # GUID
    fn_build_id = data.get('fn_build_id', '').strip().upper()  # AFID (not used currently)
    fn_firm_id = data.get('fn_firm_id', '').strip().upper()  # GUB ID (GFID)
    fn_user_id = data.get('fn_user_id', '').strip().upper()  # DFID (GUS_ID)

    # Validation: Either GUID alone OR both GFID and GUS_ID together
    if not recon_id and not (fn_firm_id and fn_user_id):
        return jsonify({
            'success': False,
            'error': 'Please provide either GFID (GUID) alone OR both GUB ID (GFID) and DFID (GUS_ID) together'
        }), 400

    if recon_id and (fn_firm_id or fn_user_id):
        return jsonify({
            'success': False,
            'error': 'Please provide either GFID (GUID) alone OR both GUB ID and DFID together, not both'
        }), 400

    if (fn_firm_id and not fn_user_id) or (fn_user_id and not fn_firm_id):
        return jsonify({
            'success': False,
            'error': 'Both GUB ID (GFID) and DFID (GUS_ID) must be provided together'
        }), 400

    try:
        # Perform reconciliation
        if recon_id:
            result = reconciliation_manager.reconcile_by_guid(recon_id)
        else:
            result = reconciliation_manager.reconcile_by_gfid_gus(fn_firm_id, fn_user_id)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Reconciliation error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Reconciliation failed: {str(e)}"
        }), 500


@app.route('/api/execute_action', methods=['POST'])
def execute_action():
    """Execute reconciliation actions"""
    try:
        data = request.get_json()
        action = data.get('action', '')
        input_value = data.get('input_value', '')

        if not action:
            return jsonify({
                'success': False,
                'error': 'Action type is required'
            }), 400

        # Simulate action execution
        import random
        import time
        time.sleep(0.5)

        result = {
            'success': True,
            'action': action,
            'input_value': input_value,
            'timestamp': datetime.now().isoformat(),
            'records_processed': random.randint(1, 10),
            'details': {
                'status': 'completed',
                'message': f'Action "{action}" executed successfully for {input_value}'
            }
        }

        logger.info(f"Executed action {action} for {input_value}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Action execution error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Action execution failed: {str(e)}"
        }), 500


@app.route('/api/export_xml', methods=['POST'])
def export_xml():
    """Export reconciliation results as XML"""
    try:
        data = request.get_json()
        results_data = data.get('results_data', {})

        # Create XML structure
        root = ET.Element('ReconciliationReport')
        
        # Add metadata
        metadata = ET.SubElement(root, 'Metadata')
        ET.SubElement(metadata, 'ExportDate').text = datetime.now().isoformat()
        ET.SubElement(metadata, 'Source').text = 'Reconciliation Tool'
        ET.SubElement(metadata, 'Version').text = '2.0'
        
        # Add reconciliation data
        recon_data = ET.SubElement(root, 'ReconciliationData')
        ET.SubElement(recon_data, 'InputType').text = str(results_data.get('input_type', 'N/A'))
        ET.SubElement(recon_data, 'InputValue').text = str(results_data.get('input_value', 'N/A'))
        ET.SubElement(recon_data, 'Timestamp').text = str(results_data.get('reconciliation_timestamp', 'N/A'))
        
        # OSCAR data
        oscar_data = ET.SubElement(recon_data, 'OSCARData')
        oscar_record = results_data.get('oscar_data_record', {})
        ET.SubElement(oscar_data, 'Found').text = str(oscar_record.get('found', False))
        ET.SubElement(oscar_data, 'Status').text = str(oscar_record.get('status', 'N/A'))
        ET.SubElement(oscar_data, 'GFID').text = str(oscar_record.get('GFID', 'N/A'))
        ET.SubElement(oscar_data, 'GUS_ID').text = str(oscar_record.get('GUS_ID', 'N/A'))
        
        # Copper data
        copper_data = ET.SubElement(recon_data, 'CopperData')
        copper_record = results_data.get('cofdm_data_record', {})
        ET.SubElement(copper_data, 'Found').text = str(copper_record.get('found', False))
        ET.SubElement(copper_data, 'Status').text = str(copper_record.get('status', 'N/A'))
        ET.SubElement(copper_data, 'GFID').text = str(copper_record.get('gfid', 'N/A'))
        ET.SubElement(copper_data, 'GUS_ID').text = str(copper_record.get('gus_id', 'N/A'))
        
        # Scenario
        scenario_elem = ET.SubElement(recon_data, 'Scenario')
        scenario = results_data.get('scenario', {})
        ET.SubElement(scenario_elem, 'Type').text = str(scenario.get('type', 'N/A'))
        ET.SubElement(scenario_elem, 'Description').text = str(scenario.get('description', 'N/A'))
        ET.SubElement(scenario_elem, 'FinalStatus').text = str(scenario.get('final_status', 'N/A'))
        ET.SubElement(scenario_elem, 'Severity').text = str(scenario.get('severity', 'N/A'))
        
        # Convert to string
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        
        # Create file buffer
        xml_buffer = io.BytesIO()
        xml_buffer.write(xml_str.encode('utf-8'))
        xml_buffer.seek(0)
        
        filename = f'reconciliation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xml'
        
        return send_file(
            xml_buffer,
            mimetype='application/xml',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"XML export error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Export failed: {str(e)}"
        }), 500


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 Not Found"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server Error"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    logger.info("Initializing the system")
    if initialize_system():
        logger.info("System initialized successfully - starting Flask application")
    else:
        logger.error("Failed to initialize system - exiting")
        import sys
        sys.exit(1)

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8050))
    debug = os.getenv('DEBUG', 'true').lower() == 'true'

    app.run(debug=debug, host=host, port=port)



















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recon Portal</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css">
    <link rel="stylesheet" href="/static/css/style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <header class="page-header">
        <div class="header-content">
            <div class="logo">
                <h1>Recon Portal</h1>
            </div>
            <nav class="main-nav"></nav>
            <div class="connection-status-container">
                <div id="connection-status" class="connection-status">
                    <span class="connection-dot"></span>
                    <span id="connection-text">Connecting...</span>
                </div>
            </div>
        </div>
    </header>

    <main class="main-content">
        <section class="intro-section fade-in">
            <div class="intro-content">
                <h1>Financial Data Reconciliation</h1>
                <p>Compare and synchronize financial trading data across systems.</p>
            </div>
        </section>

        <!-- Environment Selection Section -->
        <section class="environment-section">
            <div class="environment-card">
                <h2>Environment Selection</h2>
                <div class="form-group">
                    <label class="form-label" for="environment-select">Select Environment:</label>
                    <select id="environment-select" class="form-input">
                        <option value="dev">Development (DEV)</option>
                        <option value="qa" disabled>Quality Assurance (QA) - Coming Soon</option>
                        <option value="uat" disabled>User Acceptance Testing (UAT) - Coming Soon</option>
                        <option value="prod" disabled>Production (PROD) - Coming Soon</option>
                    </select>
                    <small class="form-hint">Select the target environment for reconciliation</small>
                </div>
            </div>
        </section>

        <!-- Reconciliation Type Selection -->
        <section class="recon-type-section">
            <div class="recon-type-card">
                <h2>Reconciliation Type</h2>
                <div class="tab-container">
                    <button class="tab-button active" data-tab="oscar-copper-star">
                        <i class="fas fa-database"></i> OSCAR - Copper - Star
                    </button>
                    <button class="tab-button" data-tab="oscar-copper-edb">
                        <i class="fas fa-server"></i> OSCAR - Copper - EDB
                    </button>
                </div>
            </div>
        </section>

        <!-- Reconciliation Form Section -->
        <section class="reconciliation-section" id="reconciliation-section">
            <div class="recon-header">
                <h2>Initiate Reconciliation</h2>
                <p>Enter GUID or combination of GFID and GUS ID</p>
            </div>

            <form id="reconcile-form">
                <div class="form-row form-row-centered">
                    <div class="form-group">
                        <label class="form-label" for="recon-id">GFID (GUID):</label>
                        <input type="text" id="recon-id" name="recon-id" class="form-input" placeholder="e.g., GFID251214">
                        <small class="form-hint">Global Firm ID (GUID) - enter alone</small>
                    </div>
                </div>
                <div class="form-row form-row-split">
                    <div class="form-group form-group-half">
                        <label class="form-label" for="fn-firm-id">GUB ID (GFID):</label>
                        <input type="text" id="fn-firm-id" name="fn-firm-id" class="form-input" placeholder="e.g., GUB_ID_FIRM_1">
                        <small class="form-hint">Enter with DFID below</small>
                    </div>
                    <div class="form-group form-group-half">
                        <label class="form-label" for="fn-user-id">DFID (GUS_ID):</label>
                        <input type="text" id="fn-user-id" name="fn-user-id" class="form-input" placeholder="e.g., DFID_USER_1">
                        <small class="form-hint">Enter with GUB ID above</small>
                    </div>
                </div>
                <div class="submit-container">
                    <button type="submit" class="btn btn-primary" id="submit-btn">
                        <span class="btn-text">Reconcile</span>
                    </button>
                </div>
            </form>
        </section>

        <!-- Results Section -->
        <section class="result-section" id="result-section">
            <div class="results-header">
                <h2>Reconciliation Results</h2>
                <button id="export-xml-btn" class="btn-secondary" style="display: none;">
                    <i class="fas fa-download"></i> Export XML
                </button>
            </div>
            
            <div class="single-comparison-table">
                <div class="table-header">
                    <h3>SSOAR vs COFDM Comparison</h3>
                </div>
                <div class="table-container">
                    <table id="comparison-table" class="comparison-table">
                        <thead class="table-header-sticky">
                            <tr>
                                <th scope="col" class="th-category">SSOAR-cat</th>
                                <th scope="col" class="th-value">SSOAR-val</th>
                                <th scope="col" class="th-id">SSOAR-id</th>
                                <th scope="col" class="th-status">Recon-Status</th>
                                <th scope="col" class="th-status">P-Final Status</th>
                                <th scope="col" class="th-id">COFDM-id</th>
                                <th scope="col" class="th-value">COFDM-val</th>
                                <th scope="col" class="th-category">COFDM-cat</th>
                            </tr>
                        </thead>
                        <tbody id="comparison-table-body">
                        </tbody>
                    </table>
                </div>
            </div>

            <div id="scenario-info-container" class="scenario-hidden">
            </div>
        </section>
    </main>
    
    <script src="/static/js/app.js"></script>
</body>
</html>










document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const reconcileForm = document.getElementById('reconcile-form');
    const resultSection = document.getElementById('result-section');
    const comparisonTableBody = document.getElementById('comparison-table-body');
    const scenarioInfoContainer = document.getElementById('scenario-info-container');
    const exportXmlBtn = document.getElementById('export-xml-btn');

    // Input fields
    const reconIdInput = document.getElementById('recon-id');
    const fnFirmIdInput = document.getElementById('fn-firm-id');
    const fnUserIdInput = document.getElementById('fn-user-id');

    // Tab functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            const tabType = this.getAttribute('data-tab');
            console.log('Selected tab:', tabType);
            // For now, both tabs use the same backend
        });
    });

    // Initialize system health check
    checkSystemHealth();
    setInterval(checkSystemHealth, 300000); // Check every 5 minutes

    // Form submission handler
    if (reconcileForm) {
        reconcileForm.addEventListener('submit', function(e) {
            e.preventDefault();
            executeReconciliation();
        });
    }

    // Input validation - mutual exclusivity
    reconIdInput.addEventListener('input', function() {
        if (this.value.trim()) {
            fnFirmIdInput.disabled = true;
            fnUserIdInput.disabled = true;
        } else {
            fnFirmIdInput.disabled = false;
            fnUserIdInput.disabled = false;
        }
    });

    [fnFirmIdInput, fnUserIdInput].forEach(input => {
        input.addEventListener('input', function() {
            if (fnFirmIdInput.value.trim() || fnUserIdInput.value.trim()) {
                reconIdInput.disabled = true;
            } else {
                reconIdInput.disabled = false;
            }
        });
    });

    // Export XML handler
    if (exportXmlBtn) {
        exportXmlBtn.addEventListener('click', function() {
            if (window.lastReconciliationData) {
                exportToXML(window.lastReconciliationData);
            } else {
                showToast('No reconciliation data available to export', 'error');
            }
        });
    }

    async function checkSystemHealth() {
        try {
            const response = await fetch('/api/health_check');
            const healthData = await response.json();
            updateConnectionStatus(healthData);
        } catch (error) {
            console.error('Health check failed:', error);
            updateConnectionStatus({
                'oscar_status': 'error',
                'cofdm_status': 'error'
            });
        }
    }

    function updateConnectionStatus(healthData) {
        const statusDot = document.querySelector('.connection-dot');
        const statusText = document.getElementById('connection-text');

        if (healthData.oscar_status === 'healthy' && healthData.cofdm_status === 'healthy') {
            statusDot.classList.remove('status-dot-degraded', 'disconnected');
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusDot.classList.add('status-dot-degraded');
            statusText.textContent = 'Degraded';
        }
    }

    async function executeReconciliation() {
        setLoadingState(true);

        try {
            const formData = getFormData();
            const validation = validateFormData(formData);
            
            if (!validation.valid) {
                showToast(validation.message, 'error');
                setLoadingState(false);
                return;
            }

            const response = await fetch('/api/reconcile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (result.success) {
                window.lastReconciliationData = result;
                updateUITableDisplay(result);
                resultSection.classList.add('show');
                resultSection.scrollIntoView({ behavior: 'smooth' });
                exportXmlBtn.style.display = 'inline-flex';
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
            recon_id: reconIdInput.value.trim() || '',
            fn_firm_id: fnFirmIdInput.value.trim() || '',
            fn_user_id: fnUserIdInput.value.trim() || ''
        };
    }

    function validateFormData(data) {
        if (!data.recon_id && !data.fn_firm_id && !data.fn_user_id) {
            return { valid: false, message: 'Please provide either GFID (GUID) or both GUB ID and DFID' };
        }

        if (data.recon_id && (data.fn_firm_id || data.fn_user_id)) {
            return { valid: false, message: 'Please provide either GFID alone or both GUB ID and DFID, not both' };
        }

        if ((data.fn_firm_id && !data.fn_user_id) || (!data.fn_firm_id && data.fn_user_id)) {
            return { valid: false, message: 'GUB ID and DFID must be provided together' };
        }

        return { valid: true, message: 'Valid' };
    }

    function updateUITableDisplay(result) {
        comparisonTableBody.innerHTML = '';
        scenarioInfoContainer.classList.add('hidden');

        const row = createTableRow(result);
        comparisonTableBody.appendChild(row);

        if (result.scenario && result.scenario.recommended_actions && result.scenario.recommended_actions.length > 0) {
            createScenarioInfo(result);
        }
    }

    function createTableRow(result) {
        const row = document.createElement('tr');

        const ssoarData = result.oscar_data_record || {};
        const cofdmData = result.cofdm_data_record || {};
        const scenario = result.scenario || {};

        const ssoarCat = ssoarData.GFID ? 'GFID' : 'N/A';
        const ssoarVal = ssoarData.found ? 'Found' : 'Not Found';
        const ssoarGfid = ssoarData.GFID || 'N/A';
        const ssoarGusId = ssoarData.GUS_ID || 'N/A';

        const cofdmGuid = cofdmData.guid || 'N/A';
        const cofdmGfid = cofdmData.gfid || 'N/A';
        const cofdmVal = cofdmData.found ? 'Found' : 'Not Found';
        const cofdmCat = cofdmData.gfid ? 'GFID' : 'Unknown';

        const finalStatus = scenario.final_status || 'UNKNOWN';

        row.innerHTML = `
            <td class="ssoar-cat">${ssoarCat}</td>
            <td class="ssoar-val">${ssoarVal}</td>
            <td class="ssoar-id">${ssoarGfid} / ${ssoarGusId}</td>
            <td>${createStatusBadge(ssoarData.status)}</td>
            <td>${createStatusBadge(finalStatus)}</td>
            <td class="cofdm-id">${cofdmGfid} / ${cofdmData.gus_id || 'N/A'}</td>
            <td class="cofdm-val">${cofdmVal}</td>
            <td class="cofdm-cat">${cofdmCat}</td>
        `;

        return row;
    }

    function createStatusBadge(status) {
        let badgeClass = '';
        let icon = '';
        const statusUpper = String(status).toUpperCase();
        
        switch (statusUpper) {
            case 'MATCH':
            case 'ACTIVE':
                badgeClass = 'status-match';
                icon = '<i class="fa fa-check-circle"></i>';
                break;
            case 'MISMATCH':
            case 'INACTIVE':
                badgeClass = 'status-mismatch';
                icon = '<i class="fa fa-exclamation-triangle"></i>';
                break;
            case 'MISSING':
            case 'NOT_FOUND':
                badgeClass = 'status-missing';
                icon = '<i class="fa fa-question-circle"></i>';
                break;
            default:
                badgeClass = 'status-missing';
                icon = '<i class="fa fa-question-circle"></i>';
                break;
        }
        return `<span class="status-badge ${badgeClass}">${icon} ${statusUpper}</span>`;
    }

    function createScenarioInfo(result) {
        const scenario = result.scenario;
        scenarioInfoContainer.innerHTML = `
            <div class="scenario-card">
                <div class="scenario-header">
                    <h4 class="scenario-title"><i class="fa fa-info-circle"></i> Scenario: ${scenario.description}</h4>
                    <span class="severity-badge severity-${scenario.severity.toLowerCase()}">${scenario.severity}</span>
                </div>
                <div class="scenario-content">
                    <p><strong>Input Value:</strong> ${result.input_value}</p>
                    <p><strong>OSCAR Status:</strong> ${result.oscar_data_record.status}</p>
                    <p><strong>Copper Status:</strong> ${result.cofdm_data_record.status}</p>
                    <h5 class="recommended-actions-h5">Recommended Actions:</h5>
                    <div class="actions-buttons">
                        ${scenario.recommended_actions.map(action => 
                            `<button class="action-btn" onclick="executeAction('${action}', '${result.input_value}')">
                                <i class="fa fa-${getActionIcon(action)}"></i> ${action}
                            </button>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
        scenarioInfoContainer.classList.remove('hidden');
    }

    function getActionIcon(action) {
        if (action.includes('Sync Job')) return 'sync-alt';
        if (action.includes('Sync Flag')) return 'flag';
        if (action.includes('Investigate')) return 'search';
        if (action.includes('Manual')) return 'hand-paper';
        if (action.includes('Verify')) return 'check-circle';
        return 'cogs';
    }

    window.executeAction = async function(action, inputValue) {
        setLoadingState(true, `Executing ${action}...`);

        try {
            const response = await fetch('/api/execute_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: action,
                    input_value: inputValue,
                }),
            });

            const result = await response.json();

            if (result.success) {
                showToast(`Action '${action}' completed successfully`, 'success');
            } else {
                showToast(result.error || 'Action failed', 'error');
            }
        } catch (error) {
            console.error('Action execution error:', error);
            showToast('Network error, action failed', 'error');
        } finally {
            setLoadingState(false);
        }
    };

    async function exportToXML(data) {
        try {
            const response = await fetch('/api/export_xml', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    results_data: data
                }),
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `reconciliation_${new Date().toISOString().replace(/[:.]/g, '-')}.xml`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                showToast('XML exported successfully', 'success');
            } else {
                showToast('Export failed', 'error');
            }
        } catch (error) {
            console.error('Export error:', error);
            showToast('Export failed - network error', 'error');
        }
    }

    function setLoadingState(loading, message = 'Processing reconciliation...') {
        const submitBtn = document.getElementById('submit-btn');
        if (loading) {
            submitBtn.innerHTML = `<i class="fa fa-spinner fa-spin"></i> ${message}`;
            submitBtn.disabled = true;
        } else {
            submitBtn.innerHTML = `<span class="btn-text">Reconcile</span>`;
            submitBtn.disabled = false;
        }
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type} show`;
        toast.innerHTML = `<div class="toast-content"><i class="fa fa-${getToastIcon(type)}"></i> <p>${message}</p></div>`;

        Object.assign(toast.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            backgroundColor: getToastColor(type),
            color: 'white',
            padding: '1rem',
            borderRadius: '0.5rem',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
            zIndex: '1000',
            maxWidth: '400px',
            animation: 'slideInRight 0.3s ease-out',
        });

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease-out';
            toast.addEventListener('animationend', () => {
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            });
        }, 4000);
    }

    function getToastIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }

    function getToastColor(type) {
        const colors = {
            success: '#34a853',
            error: '#ea4335',
            info: '#1a73e8',
        };
        return colors[type] || colors.info;
    }

    console.log('Reconciliation Tool JavaScript loaded successfully');
});












/* static/css/style.css - Updated with new sections */

/* --- Base Styles & Variables --- */
*,
*::before,
*::after {
    box-sizing: border-box;
}

:root {
    --primary-color: #1a73e8;
    --primary-light: #669df6;
    --primary-dark: #174ea6;
    --accent-color: #fbbc05;
    --success-color: #34a853;
    --error-color: #ea4335;
    --warning-color: #f29900;
    --bg-color: #1e1e1e;

    --white: #ffffff;
    --gray-100: #f8f9fa;
    --gray-200: #e9ecef;
    --gray-300: #dee2e6;
    --gray-400: #ced4da;
    --gray-500: #adb5bd;
    --gray-600: #6c757d;
    --gray-700: #495057;
    --gray-800: #343a40;
    --gray-900: #212529;

    --bg-card: #2a2a2e;
    --bg-secondary: #3c4043;
    --bg-darker: #121212;

    --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
    --gradient-success: linear-gradient(135deg, var(--success-color) 0%, #66bb6a 100%);
    --gradient-error: linear-gradient(135deg, var(--error-color) 0%, #ef5350 100%);

    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    --font-size-sm: 0.875rem;
    --font-size-base: 1rem;
    --font-size-md: 1.125rem;
    --font-size-lg: 1.5rem;
    --font-size-xl: 2.25rem;
    --font-size-xxl: 2.5rem;

    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --spacing-xxl: 3rem;

    --radius-sm: 0.25rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-xl: 1rem;
    --radius-full: 9999px;

    --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.15);

    --transition-fast: 0.15s ease-in-out;
    --transition-normal: 0.3s ease-in-out;
    --transition-slow: 0.5s ease-in-out;
}

body {
    font-family: var(--font-family);
    line-height: 1.6;
    color: var(--gray-200);
    background-color: var(--bg-color);
    margin: 0;
    padding: 0;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--white);
    font-weight: 700;
}

/* --- Header --- */
.page-header {
    background: var(--gradient-primary);
    padding: var(--spacing-lg) 0;
    position: sticky;
    top: 0;
    z-index: 100;
    border-bottom: 2px solid var(--accent-color);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 var(--spacing-lg);
}

.logo h1 {
    font-size: var(--font-size-lg);
    font-weight: 700;
    color: var(--white);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    margin: 0;
}

.connection-status {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    background: rgba(255, 255, 255, 0.15);
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-full);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.connection-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--warning-color);
    border: 2px solid var(--white);
}

.connection-dot.connected {
    background: var(--success-color);
}

.connection-dot.status-dot-degraded {
    background: var(--error-color);
}

/* --- Main Content --- */
.main-content {
    padding: var(--spacing-xxl) 0;
    min-height: calc(100vh - 100px);
    max-width: 1400px;
    margin: 0 auto;
    padding-left: var(--spacing-lg);
    padding-right: var(--spacing-lg);
}

/* --- Intro Section --- */
.intro-section {
    text-align: center;
    margin-bottom: var(--spacing-xxl);
}

.intro-content h1 {
    font-size: var(--font-size-xxl);
    font-weight: 700;
    margin-bottom: var(--spacing-md);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.intro-content p {
    font-size: var(--font-size-lg);
    max-width: 600px;
    margin: 0 auto;
}

/* --- Environment Selection Section --- */
.environment-section {
    margin-bottom: var(--spacing-xxl);
}

.environment-card {
    background: var(--bg-card);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-xl);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
}

.environment-card h2 {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
    margin-bottom: var(--spacing-lg);
    text-align: center;
}

/* --- Reconciliation Type Section --- */
.recon-type-section {
    margin-bottom: var(--spacing-xxl);
}

.recon-type-card {
    background: var(--bg-card);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-xl);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
}

.recon-type-card h2 {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
    margin-bottom: var(--spacing-lg);
    text-align: center;
}

.tab-container {
    display: flex;
    gap: var(--spacing-md);
    justify-content: center;
    flex-wrap: wrap;
}

.tab-button {
    background: var(--bg-darker);
    color: var(--white);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-md) var(--spacing-xl);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-md);
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition-normal);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.tab-button:hover {
    border-color: var(--accent-color);
    background: var(--primary-dark);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.tab-button.active {
    background: var(--gradient-primary);
    border-color: var(--accent-color);
    box-shadow: var(--shadow-lg);
}

.tab-button i {
    font-size: var(--font-size-md);
}

/* --- Reconciliation Section --- */
.reconciliation-section {
    margin-bottom: var(--spacing-xxl);
}

.recon-header {
    text-align: center;
    margin-bottom: var(--spacing-xl);
}

.recon-header h2 {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
}

.recon-header p {
    color: var(--gray-300);
    font-size: var(--font-size-md);
}

/* --- Form Styles --- */
#reconcile-form {
    background: var(--bg-card);
    padding: var(--spacing-xl);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    border: 2px solid var(--primary-color);
}

.form-row {
    margin-bottom: var(--spacing-lg);
}

.form-row-centered {
    display: flex;
    justify-content: center;
    margin-bottom: var(--spacing-xl);
}

.form-row-centered .form-group {
    width: 100%;
    max-width: 500px;
}

.form-row-split {
    display: flex;
    gap: var(--spacing-lg);
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.form-group-half {
    flex: 1;
}

.form-label {
    font-size: var(--font-size-base);
    color: var(--white);
    font-weight: 600;
}

.form-input {
    padding: var(--spacing-md);
    border: 2px solid var(--primary-color);
    border-radius: var(--radius-md);
    font-size: var(--font-size-base);
    color: var(--white);
    background: var(--bg-darker);
    transition: var(--transition-normal);
    box-shadow: var(--shadow-sm);
}

.form-input:focus {
    outline: none;
    border-color: var(--accent-color);
    box-shadow: 0 0 8px 0 rgba(251, 188, 5, 0.3);
    transform: translateY(-2px);
}

.form-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.form-input::placeholder {
    color: var(--gray-600);
}

.form-hint {
    font-size: var(--font-size-sm);
    color: var(--gray-400);
}

select.form-input {
    cursor: pointer;
}

select.form-input option:disabled {
    color: var(--gray-600);
}

.submit-container {
    text-align: center;
    margin-top: var(--spacing-xl);
}

/* --- Buttons --- */
.btn {
    background: var(--gradient-primary);
    color: var(--white);
    padding: var(--spacing-md) var(--spacing-xxl);
    border: 2px solid transparent;
    border-radius: var(--radius-full);
    font-size: var(--font-size-lg);
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    justify-content: center;
    align-items: center;
    gap: var(--spacing-sm);
    transition: var(--transition-normal);
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.btn:hover:not(:disabled) {
    box-shadow: var(--shadow-md);
    border-color: var(--accent-color);
    transform: translateY(-2px);
}

.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.btn-secondary {
    background: var(--bg-card);
    color: var(--white);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-full);
    font-size: var(--font-size-md);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    box-shadow: var(--shadow-md);
}

.btn-secondary:hover {
    border-color: var(--accent-color);
    background: var(--primary-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

/* --- Results Section --- */
.result-section {
    margin-bottom: var(--spacing-xxl);
    display: none;
}

.result-section.show {
    display: block;
    animation: fadeInUp 0.8s ease-out;
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-xl);
}

.results-header h2 {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
}

/* --- Comparison Table --- */
.single-comparison-table {
    background: var(--bg-card);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    overflow: hidden;
    border: 2px solid var(--primary-color);
    box-shadow: var(--shadow-lg);
}

.table-header {
    background: var(--gradient-primary);
    padding: var(--spacing-md);
    text-align: center;
    border-bottom: 2px solid var(--accent-color);
    margin: calc(-1 * var(--spacing-xl)) calc(-1 * var(--spacing-xl)) var(--spacing-lg);
}

.table-header h3 {
    color: var(--white);
    font-size: var(--font-size-lg);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    margin: 0;
}

.table-container {
    overflow-x: auto;
}

.comparison-table {
    width: 100%;
    border-collapse: collapse;
}

.comparison-table th {
    background: var(--primary-dark);
    padding: var(--spacing-md);
    text-align: center;
    border-bottom: 2px solid var(--primary-color);
    color: var(--white);
    font-weight: 600;
    font-size: var(--font-size-sm);
    position: sticky;
    top: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.comparison-table td {
    padding: var(--spacing-md);
    border: 1px solid var(--gray-800);
    font-size: var(--font-size-sm);
    vertical-align: middle;
    text-align: center;
}

.comparison-table tbody tr:hover {
    background: var(--gray-900);
}

/* Column Styling */
.ssoar-cat, .cofdm-cat {
    font-weight: bold;
    background: rgba(26, 63, 136, 0.3);
}

.ssoar-val, .cofdm-val {
    background: rgba(10, 100, 216, 0.3);
}

.ssoar-id, .cofdm-id {
    background: rgba(219, 137, 52, 0.3);
}

/* --- Status Badges --- */
.status-badge {
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-sm);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--white);
    box-shadow: var(--shadow-sm);
    border: 1px solid transparent;
    display: inline-block;
    text-align: center;
}

.status-badge.status-match {
    background: var(--success-color);
    border-color: var(--white);
}

.status-badge.status-mismatch {
    background: var(--error-color);
    border-color: var(--white);
}

.status-badge.status-missing {
    background: var(--warning-color);
    color: var(--white);
}

/* --- Scenario Info --- */
.scenario-hidden {
    display: none;
}

.scenario-card {
    background: var(--bg-card);
    padding: var(--spacing-xl);
    border-radius: var(--radius-xl);
    margin-top: var(--spacing-xl);
    border: 2px solid var(--accent-color);
    box-shadow: var(--shadow-lg);
}

.scenario-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-lg);
    flex-wrap: wrap;
    gap: var(--spacing-md);
}

.scenario-title {
    font-size: var(--font-size-lg);
    color: var(--white);
    margin: 0;
}

.severity-badge {
    padding: var(--spacing-xs) var(--spacing-md);
    border-radius: var(--radius-full);
    font-size: var(--font-size-sm);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.severity-badge.severity-low {
    background: var(--success-color);
    color: var(--white);
}

.severity-badge.severity-medium {
    background: var(--warning-color);
    color: var(--white);
}

.severity-badge.severity-high {
    background: var(--error-color);
    color: var(--white);
}

.scenario-content p {
    color: var(--gray-300);
    font-size: var(--font-size-base);
    margin-bottom: var(--spacing-sm);
}

.scenario-content strong {
    color: var(--white);
}

.recommended-actions-h5 {
    margin-top: var(--spacing-lg);
    margin-bottom: var(--spacing-md);
    color: var(--accent-color);
}

.actions-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-md);
}

.action-btn {
    background: var(--gradient-primary);
    color: white;
    border: 1px solid var(--accent-color);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: var(--white);
}

/* --- Animations --- */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes slideOutRight {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}

.fade-in {
    animation: fadeInUp 0.8s ease-out;
}

/* --- Scrollbar --- */
::-webkit-scrollbar {
    height: 8px;
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-darker);
    border-radius: var(--radius-sm);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: var(--radius-sm);
    border: 2px solid var(--accent-color);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-light);
}

/* --- Responsive Design --- */
@media (max-width: 768px) {
    .header-content {
        flex-direction: column;
        gap: var(--spacing-md);
    }

    .form-row-split {
        flex-direction: column;
    }

    .results-header {
        flex-direction: column;
        gap: var(--spacing-md);
        align-items: stretch;
    }

    .tab-container {
        flex-direction: column;
    }

    .tab-button {
        width: 100%;
        justify-content: center;
    }

    .comparison-table {
        font-size: var(--font-size-sm);
    }

    .comparison-table th,
    .comparison-table td {
        padding: var(--spacing-sm);
    }
}

@media (max-width: 480px) {
    .intro-content h1 {
        font-size: var(--font-size-xl);
    }
}

/* --- Utilities --- */
.hidden {
    display: none !important;
}













#!/usr/bin/env python3
"""
Reconciliation Tool Launcher
Script with startup health checks and validation
"""

import sys
import os
import importlib
import time

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version is compatible: {sys.version}")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("📦 Checking required dependencies...")
    required_packages = [
        ('flask', 'Flask'),
        ('google.cloud.sql.connector', 'cloud-sql-python-connector'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('google.auth', 'google-auth'),
        ('dotenv', 'python-dotenv'),
        ('flask_cors', 'Flask-CORS')
    ]
    
    missing_packages = []
    for package_import, package_name in required_packages:
        try:
            importlib.import_module(package_import)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print(f"❌ The following packages are missing: {', '.join(missing_packages)}")
        print("💡 Please run: pip install -r requirements.txt")
        return False
    print("✅ All dependencies are installed")
    return True

def check_google_cloud_authentication():
    """Check for Google Cloud authentication credentials"""
    print("☁️  Checking Google Cloud authentication...")
    try:
        from google.auth import default
        credentials, project = default()
        print(f"✅ Google Cloud authentication successful (project: {project})")
        return True
    except Exception as e:
        print(f"❌ Google Cloud authentication failed: {e}")
        print("💡 Run: gcloud auth application-default login")
        return False

def check_env_file():
    """Check if .env file exists"""
    print("📄 Checking for .env file...")
    if os.path.exists('.env'):
        print("✅ .env file found")
        return True
    else:
        print("⚠️  .env file not found (will use defaults)")
        return True  # Not a fatal error

def test_database_connections():
    """Test connections to all required databases"""
    try:
        from enhanced_database_connector import ReconciliationManager
        print("🧪 Testing database connections...")
        manager = ReconciliationManager()
        success = manager.initialize_connections()

        if success:
            results = manager.test_all_connections()
            for db_name, status in results.items():
                if status:
                    print(f"  ✅ {db_name.upper()}: Connected")
                else:
                    print(f"  ❌ {db_name.upper()}: Failed")

            manager.close()
            return all(results.values())
        else:
            print("❌ Failed to initialize database connections")
            return False

    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def start_application():
    """Start the Flask application"""
    try:
        print("\n🚀 Starting Reconciliation Tool...")
        print(f"🔗 URL: http://localhost:8050")
        print(f"🩺 Health Check: http://localhost:8050/api/health_check")
        print("💡 Use Ctrl+C to stop the application\n")

        from app import app, initialize_system

        if initialize_system():
            app.run(debug=True, host="0.0.0.0", port=8050)
        else:
            print("❌ Failed to initialize application")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Application startup failed: {e}")
        return False

def print_usage_info():
    """Print usage information"""
    print("\n" + "="*60)
    print("✨ RECONCILIATION TOOL - USAGE GUIDE ✨")
    print("="*60)
    print("🔹 Available Input Types:")
    print("  ▪ GFID (GUID): Global Firm ID")
    print("  ▪ GUB ID + DFID: GFID and GUS_ID combination")
    print("\n🔹 Supported Reconciliation Scenarios:")
    print("  ▪ SCENARIO_2_1: Status mismatch between systems")
    print("  ▪ SCENARIO_2_2: OSCAR Inactive but Copper active")
    print("  ▪ SCENARIO_2_3: Inactive in OSCAR, missing in Copper")
    print("  ▪ SCENARIO_2_4: Active in OSCAR, missing in Copper")
    print("\n🔹 Available Actions:")
    print("  ▪ Run Sync Job: Automatic synchronization")
    print("  ▪ Run Sync Flag: Mark for synchronization")
    print("  ▪ Investigate: Raise investigation ticket")
    print("  ▪ Manual sync: Manual synchronization required")
    print("\n🔹 Export Options:")
    print("  ▪ XML Export: Download reconciliation results as XML")
    print("="*60)

def main():
    """Main launcher function"""
    print("✨ Reconciliation Tool - Startup Validator ✨")
    print("="*60 + "\n")

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Step 3: Check Google Cloud authentication
    if not check_google_cloud_authentication():
        print("\n💡 To fix authentication:")
        print("   gcloud auth login")
        print("   gcloud auth application-default login")
        sys.exit(1)

    # Step 4: Check environment file
    check_env_file()

    # Step 5: Test database connections
    if not test_database_connections():
        print("\n💡 For database connection troubleshooting:")
        print("   - Verify VPN/network connectivity")
        print("   - Check .env file configuration")
        print("   - Ensure Cloud SQL instances are running")

        response = input("\n🤔 Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Step 6: Print usage information
    print_usage_info()

    # Step 7: Start application
    input("\n▶️  Press Enter to start the application...")
    start_application()

if __name__ == "__main__":
    main()














# Core Framework
Flask
Flask-CORS

# Database & Cloud SQL
SQLAlchemy
psycopg2-binary
cloud-sql-python-connector[pg8000]

# Google Cloud
google-auth
google-cloud-sql-connector

# Configuration & Environment
python-dotenv

# Date/Time handling
python-dateutil

# Utilities
requests










/* static/css/animation.css */
/* Base & Keyframe Animations */

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes fadeOut {
    from { opacity: 1; }
    to { opacity: 0; }
}

@keyframes bounceIn {
    0% {
        opacity: 0;
        transform: translateY(-30px);
    }
    60% {
        opacity: 1;
        transform: translateY(5px);
    }
    100% {
        transform: translateY(0);
    }
}

@keyframes slideInFromLeft {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInFromRight {
    from {
        opacity: 0;
        transform: translateX(20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideOutToRight {
    from {
        opacity: 1;
        transform: translateX(0);
    }
    to {
        opacity: 0;
        transform: translateX(20px);
    }
}

@keyframes scaleIn {
    from {
        opacity: 0.8;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.05);
    }
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
        transform: translateY(0);
    }
    40% {
        transform: translateY(-10px);
    }
    60% {
        transform: translateY(-5px);
    }
    90% {
        transform: translateY(-2px);
    }
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes shimmer {
    0% {
        background-position: -200px 0;
    }
    100% {
        background-position: calc(200px + 100%) 0;
    }
}

@keyframes slideOutRight {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 5px rgba(74, 144, 226, 0.3);
    }
    50% {
        box-shadow: 0 0 20px rgba(74, 144, 226, 0.6);
    }
}

@keyframes typewriter {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blinkCursor {
    from, to { border-color: transparent; }
    50% { border-color: var(--primary-blue); }
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Page Load Animations */
.intro-section {
    animation: fadeInDown 0.8s ease-out;
}

.form-card {
    animation: fadeInUp 0.8s ease-out 0.2s both;
}

.results-section.show {
    animation: fadeInUp 0.8s ease-out;
}

/* Component Animations */
.scenario-info {
    animation: scaleIn 0.3s ease-out;
}

.summary-card:nth-child(1) { animation-delay: 0.1s; }
.summary-card:nth-child(2) { animation-delay: 0.2s; }
.summary-card:nth-child(3) { animation-delay: 0.3s; }

.table-container {
    animation: fadeInUp 0.6s ease-out;
}

.table-container tbody tr:nth-child(1) { animation-delay: 0.1s; }
.table-container tbody tr:nth-child(2) { animation-delay: 0.2s; }

/* Interactive Animations */
.execute-btn {
    overflow: hidden;
    position: relative;
}

.execute-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -80%;
    width: 30%;
    height: 100%;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s;
}

.execute-btn:hover::before {
    left: 100%;
}

.btn-secondary {
    position: relative;
    overflow: hidden;
}

.btn-secondary:after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 5px;
    height: 5px;
    background: rgba(0,0,0,0.05);
    border-radius: 50%;
    transform: translate(-50%, -50%) scale(1);
    transition: width 0.6s, height 0.6s;
}

.btn-secondary:active:after {
    width: 300px;
    height: 300px;
}

/* Status Badge Animations */
.status-badge {
    position: relative;
    overflow: hidden;
}

.status-badge-match {
    animation: pulse 2s infinite;
}

.status-badge-mismatch {
    animation: pulse 1.5s infinite;
}

/* Form Input Animations */
.form-group:focus-within,
.form-group.input-has-focus {
    animation: glow 1s ease-in-out infinite;
}

.input-info {
    transform: translateY(-10px);
    opacity: 0;
    transition: all 0.3s;
}

.form-group:focus-within .input-info,
.form-group.input-has-focus .input-info {
    transform: translateY(0);
    opacity: 1;
}

/* Connection Status Animation */
.connection-status.connected {
    animation: pulse 2s infinite;
}

.connection-status.disconnected {
    animation: bounce 1s infinite;
}

/* Table Row Animations */
.comparison-table tbody tr {
    transform: translateX(-20px);
    opacity: 0;
    animation: slideInFromLeft 0.5s ease-out forwards;
}
.comparison-table tbody tr:nth-child(1) { animation-delay: 0.1s; }
.comparison-table tbody tr:nth-child(2) { animation-delay: 0.2s; }
.comparison-table tbody tr:nth-child(3) { animation-delay: 0.3s; }
.comparison-table tbody tr:nth-child(4) { animation-delay: 0.4s; }
.comparison-table tbody tr:nth-child(5) { animation-delay: 0.5s; }

/* Action Icon Animations */
.action-icon {
    transition: all 0.3s cubic-bezier(0.42, -0.55, 0.58, 1.55);
}

.action-icon:hover {
    animation: bounce 0.8s ease;
}

/* Card Hover Effects */
.summary-card {
    transition: all 0.3s ease;
}
.summary-card:hover {
    animation: none;
    transform: translateY(-4px) scale(1.03);
}

/* Loading Animations */
.loading-dots::after {
    content: '...';
    animation: typewriter 1.5s steps(3) infinite;
}

/* Shimmer Effect for Loading States */
.shimmer {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}

/* Scenario Info Animation */
.scenario-info {
    animation: slideInFromLeft 0.6s ease-out;
}
.scenario-info .info-item {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 0.6s ease-out forwards;
}

.scenario-info .action-btn:nth-child(1) { animation-delay: 0.1s; }
.scenario-info .action-btn:nth-child(2) { animation-delay: 0.2s; }
.scenario-info .action-btn:nth-child(3) { animation-delay: 0.3s; }

/* Toast Notification Animations */
.toast {
    animation: slideInRight 0.3s ease-out;
}
.toast.removing {
    animation: slideOutRight 0.3s ease-out;
}

/* Progress Bar Animation */
.progress-bar-inner {
    width: 0%;
    background: var(--primary-blue);
    height: 100%;
    transition: width 0.5s ease;
}
.progress-bar.loading .progress-bar-inner {
    animation: none 1.5s infinite;
}

/* Number Counter Animation */
.card-number {
    animation: fadeIn 0.8s ease-out 0.5s forwards;
    opacity: 0;
}
.card-number.counting {
    animation: bounce 0.8s ease-in-out;
}

/* Stagger Animation for Grid Items */
.form-row.growth {
    transform: translateY(20px);
    opacity: 0;
    animation: fadeInUp 0.8s ease-out forwards;
}
.form-row.growth:nth-child(1) { animation-delay: 0.1s; }
.form-row.growth:nth-child(2) { animation-delay: 0.2s; }
.form-row.growth:nth-child(3) { animation-delay: 0.3s; }
.form-row.growth:nth-child(4) { animation-delay: 0.4s; }

/* Header Animation */
.logo {
    animation: fadeInDown 0.8s ease-out 0.2s both;
}
.connection-status {
    animation: fadeInDown 0.8s ease-out 0.4s both;
}

/* Elastic Scale Animation */
@keyframes elasticScale {
    0% { transform: scale(1); }
    30% { transform: scale(1.1); }
    60% { transform: scale(0.95); }
    80% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.execute-btn:active {
    animation: elasticScale 0.6s ease-out;
}

/* Floating Animation */
@keyframes float {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-5px);
    }
}
.floating {
    animation: float 3s ease-in-out infinite;
}

/* Attention Seeking Animations */
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}
.shake {
    animation: shake 0.6s ease-in-out;
}

/* Success Animation */
@keyframes successPulse {
    0% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 10px 10px rgba(39, 174, 96, 0);
    }
    100% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(39, 174, 96, 0);
    }
}
.success-animation {
    animation: successPulse 0.8s ease-out;
}

/* Error Animation */
@keyframes errorShake {
    0%, 100% {
        transform: translateX(0);
    }
    25% {
        transform: translateX(-8px);
    }
    75% {
        transform: translateX(8px);
    }
}
.error-animation {
    animation: errorShake 0.5s ease-out;
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

/* Animation Utility Classes */
.animate-fade-in { animation: fadeIn 0.5s ease-out; }
.animate-fade-out { animation: fadeOut 0.5s ease-out; }
.animate-slide-in-left { animation: slideInFromLeft 0.5s ease-out; }
.animate-slide-in-right { animation: slideInFromRight 0.5s ease-out; }
.animate-bounce-in { animation: bounceIn 0.6s ease-out; }
.animate-scale-in { animation: scaleIn 0.3s ease-out; }
.animate-pulse { animation: pulse 1.5s ease-in-out infinite; }
.animate-shimmer { animation: shimmer 1.5s ease-in-out infinite; }
.animate-spin { animation: spin 1s linear infinite; }
.animate-glow { animation: glow 2s ease-in-out infinite; }

/* Animation Speeds */
.delay-100 { animation-delay: 0.1s; }
.delay-200 { animation-delay: 0.2s; }
.delay-300 { animation-delay: 0.3s; }
.delay-400 { animation-delay: 0.4s; }
.delay-500 { animation-delay: 0.5s; }

.animation-fast { animation-duration: 0.3s; }
.animation-slow { animation-duration: 2s; }














