from flask import Flask, render_template, request, jsonify
from google.cloud.sql.connector import Connector
import google.auth
from sqlalchemy import create_engine, text
import oracledb
import sqlalchemy
import re
import os
from ldap3 import Server, Connection, ALL
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Database configurations
CA_CERT_PATH = "/app/new.pem"  # Update path for Docker

DB_INSTANCES = {
    'oscar': {
        'project': 'pri-dv-oscar-0302',
        'region': 'us-central1',
        'instance': 'csal-dv-usc1-1316-oscar-0004-m',
        'database': 'padk',
        'user': 'lakshya.vijay@cmegroup.com',
        'schema': 'dv01cosrs',
        'description': 'OSCAR Database Instance'
    }
}

ORACLE_INSTANCES = {
    'copper': {
        'user': 'QF44CCRDO',
        'password': 'Felty_22_PAMPA_TRJ_44',
        'url': 'ldap://ORAQALDAP:3060/DCCRDC,cn=OracleContext,dc=world',
        'description': 'CoPPER Database'
    },
    'edb': {
        'user': 'DV02CEDBO',
        'password': 'eh02eedbo',
        'url': 'ldap://ORAQALDAP:3060/DCEDBA,cn=OracleContext,dc=world',
        'description': 'EDB Database'
    },
    'star': {
        'user': 'DV99CFASO',
        'password': 'DV99CFASOJ',
        'url': 'ldap://ORAQALDAP:3060/DCFASA,cn=OracleContext,dc=world',
        'description': 'STAR Database'
    }
}


class DatabaseConnector:
    """PostgreSQL connector for OSCAR"""
    
    def __init__(self, instance_name: str):
        if instance_name not in DB_INSTANCES:
            raise ValueError(f"Unknown instance: {instance_name}")
        
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
        try:
            logger.info(f"Initializing {self.config['description']} connector")
            
            # SSL disable hack
            if os.path.exists(CA_CERT_PATH):
                os.environ['REQUESTS_CA_BUNDLE'] = CA_CERT_PATH
                os.environ['GOOGLE_AUTH_DISABLE_TLS_VERIFY'] = 'True'
                logger.warning(f"SSL verification disabled: {CA_CERT_PATH}")
            else:
                logger.error(f"CA file not found at: {CA_CERT_PATH}")
                raise FileNotFoundError(f"Required CA file not found at {CA_CERT_PATH}")
            
            # Test authentication
            credentials, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/sqlservice.admin"]
            )
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
                    ip_type="PRIVATE"
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
                    logger.error(f"{self.instance_name.upper()} connection test failed")
                    return False
                    
        except Exception as e:
            logger.error(f"{self.instance_name.upper()} connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                columns = result.keys()
                rows = []
                for row in result:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert date objects to string
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[col] = value
                    rows.append(row_dict)
                return rows
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise


class OracleConnector:
    """Oracle DB connector for CoPPER, EDB, and STAR"""
    
    def __init__(self, instance_name: str):
        if instance_name not in ORACLE_INSTANCES:
            raise ValueError(f"Unknown Oracle instance: {instance_name}")
        
        self.instance_name = instance_name
        self.config = ORACLE_INSTANCES[instance_name]
        self.engine = None
        self._initialize_connection()
    
    def get_tns_string_from_ldap_url(self, ldap_url: str) -> Optional[str]:
        try:
            pattern = r"^ldap://(.+)/(.+),(cn=OracleContext.*)$"
            match = re.match(pattern, ldap_url)
            
            if not match:
                return None
            
            ldap_server, db, ora_context = match.groups()
            
            server = Server(ldap_server)
            conn = Connection(server)
            conn.bind()
            
            conn.search(ora_context, f"(cn={db})", attributes=['orclNetDescString'])
            
            if conn.entries:
                tns = conn.entries[0].orclNetDescString.value
                return tns
            else:
                return None
                
        except Exception as e:
            logger.error(f"LDAP lookup failed: {e}")
            return None
    
    def decrypt(self, message: str) -> str:
        news = ""
        for car in message:
            news = news + chr(ord(car) - 2)
        return news
    
    def _initialize_connection(self):
        try:
            logger.info(f"Initializing {self.config['description']} connector")
            
            user = self.config['user']
            password = self.config['password']
            url = self.config['url']
            
            tns_string = self.get_tns_string_from_ldap_url(url)
            
            if tns_string:
                self.engine = sqlalchemy.create_engine(
                    f"oracle+oracledb://{user}:{password}@{tns_string}"
                )
                logger.info(f"{self.instance_name.upper()} engine created successfully")
            else:
                raise Exception("Failed to retrieve TNS string from LDAP")
                
        except Exception as e:
            logger.error(f"{self.instance_name.upper()} initialization failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        try:
            if not self.engine:
                logger.error("Database engine not initialized")
                return False
            
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT CURRENT_DATE FROM DUAL"))
                row = result.fetchone()
                
                if row:
                    logger.info(f"{self.instance_name.upper()} connection test successful")
                    return True
                else:
                    logger.error(f"{self.instance_name.upper()} connection test failed")
                    return False
                    
        except Exception as e:
            logger.error(f"{self.instance_name.upper()} connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params or {})
                columns = result.keys()
                rows = []
                for row in result:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Convert date objects to string
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[col] = value
                    rows.append(row_dict)
                return rows
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise


class ReconciliationEngine:
    """Main reconciliation logic"""
    
    def __init__(self):
        self.oscar_conn = None
        self.copper_conn = None
        self.edb_conn = None
        self.star_conn = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        try:
            self.oscar_conn = DatabaseConnector('oscar')
            self.copper_conn = OracleConnector('copper')
            self.edb_conn = OracleConnector('edb')
            self.star_conn = OracleConnector('star')
            
            # Test all connections
            if not all([
                self.oscar_conn.test_connection(),
                self.copper_conn.test_connection(),
                self.edb_conn.test_connection(),
                self.star_conn.test_connection()
            ]):
                raise Exception("One or more database connections failed")
                
            logger.info("All database connections established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    def get_oscar_data_by_guid(self, guid: str) -> List[Dict]:
        query = """
        SELECT
            GUID,
            (xpath('//globexUserSignature/globexUserSignatureInfo/globexFirmIDText()', guses.XML))[1]::text as "GFID",
            (xpath('//globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text as "GUS_ID",
            (xpath('//globexUserSignature/globexUserSignatureInfo/exchange/text()', guses.XML))[1]::text as "Exchange",
            namespace,
            CASE
                WHEN (xpath('//globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.xml))[1]::text IS NULL
                THEN 'MISSING'
                WHEN TO_DATE((xpath('//globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.xml))[1]::text, 'DD-MON-YYYY') >= CURRENT_DATE
                THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM dv01cosrs.active_xml_data_store guses
        WHERE guses.GUID = :guid
        AND (guses.namespace = 'GlobexUserSignature')
        LIMIT 1
        """
        return self.oscar_conn.execute_query(query, {'guid': guid})
    
    def get_oscar_data_by_gfid_gusid(self, gfid: str, gus_id: str) -> List[Dict]:
        query = """
        SELECT
            guid,
            (xpath('//globexUserSignature/globexUserSignatureInfo/globexFirmIDText()', guses.XML))[1]::text as "GFID",
            (xpath('//globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text as "GUS_ID",
            (xpath('//globexUserSignature/globexUserSignatureInfo/exchange/text()', guses.XML))[1]::text as "Exchange",
            namespace,
            CASE
                WHEN (xpath('//globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.xml))[1]::text IS NULL
                THEN 'MISSING'
                WHEN TO_DATE((xpath('//globexUserSignature/globexUserSignatureInfo/expDate/text()', guses.xml))[1]::text, 'DD-MON-YYYY') >= CURRENT_DATE
                THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM dv01cosrs.active_xml_data_store guses
        WHERE (xpath('//globexUserSignature/globexUserSignatureInfo/globexFirmIDText()', guses.XML))[1]::text = :gfid
        AND (xpath('//globexUserSignature/globexUserSignatureInfo/idNumber/text()', guses.XML))[1]::text = :gus_id
        AND namespace = 'GlobexUserSignature'
        LIMIT 1
        """
        return self.oscar_conn.execute_query(query, {'gfid': gfid, 'gus_id': gus_id})
    
    def get_copper_data_by_guid(self, guid: str) -> List[Dict]:
        query = """
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
        FROM QF44CCRDO.trd_gpid
        WHERE guid = :guid
        """
        return self.copper_conn.execute_query(query, {'guid': guid})
    
    def get_copper_data_by_gfid_gusid(self, gfid: str, gus_id: str) -> List[Dict]:
        query = """
        SELECT
            guid,
            gpid_id as gus_id,
            gfid_id as gfid,
            eff_to,
            exch_id,
            CASE
                WHEN eff_to IS NULL THEN 'MISSING'
                WHEN eff_to >= CURRENT_DATE THEN 'ACTIVE'
                ELSE 'INACTIVE'
            END as status
        FROM QF44CCRDO.trd_gpid
        WHERE gfid_id = :gfid AND gpid_id = :gus_id
        """
        return self.copper_conn.execute_query(query, {'gfid': gfid, 'gus_id': gus_id})
    
    def get_edb_data(self, gfid: str, gus_id: str) -> List[Dict]:
        query = """
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
        FROM DV02CEDBO.trd_gpid
        WHERE gfid_id = :gfid AND gpid_id = :gus_id
        """
        return self.edb_conn.execute_query(query, {'gfid': gfid, 'gus_id': gus_id})
    
    def get_star_data(self, gfid: str) -> List[Dict]:
        query = """
        SELECT gfid, GFID_DESCRIPTION, exchange_id
        FROM ebs_gfid
        WHERE gfid = :gfid
        
        UNION ALL
        
        SELECT gfid, GFID_DESCRIPTION, gfid_type as exchange_id
        FROM bmc_gfid
        WHERE gfid = :gfid
        """
        return self.star_conn.execute_query(query, {'gfid': gfid})
    
    def determine_scenario(self, oscar_status: str, copper_status: str, 
                          oscar_data: Dict, copper_data: Dict) -> Tuple[str, str]:
        """Determine which scenario applies and return scenario name and recommended action"""
        
        # Check if data exists
        oscar_exists = bool(oscar_data)
        copper_exists = bool(copper_data)
        
        if not oscar_exists and not copper_exists:
            return "No Data", "No records found in either OSCAR or CoPPER"
        
        if not copper_exists:
            if oscar_status == 'ACTIVE':
                return "Scenario 2.4", "OSCAR Active GUID is not present in CoPPER - Check why it is missing and take appropriate action"
            elif oscar_status == 'INACTIVE':
                return "Scenario 2.3", "OSCAR Expired GUID is missing in CoPPER - Run Sync FLAG to N Job"
        
        if not oscar_exists and copper_exists:
            return "Data Mismatch", "Record exists in CoPPER but not in OSCAR - Investigate data inconsistency"
        
        # Both exist - compare statuses
        if oscar_status == 'INACTIVE' and copper_status == 'INACTIVE':
            # Check if GFID and GUS_ID match
            if oscar_data.get('gfid') == copper_data.get('gfid') and \
               oscar_data.get('gus_id') == copper_data.get('gus_id'):
                return "Scenario 2.1", "Both Expired with same GFID and GUS ID - Run MASS SYNC job and then run Sync FLAG to N Job"
            else:
                return "Scenario 2.1 (Mismatch)", "Both Expired but GFID/GUS_ID mismatch - Verify data integrity before sync"
        
        if oscar_status == 'INACTIVE' and copper_status == 'ACTIVE':
            # Check if present with different GUID
            if oscar_data.get('gfid') != copper_data.get('gfid') or \
               oscar_data.get('gus_id') != copper_data.get('gus_id'):
                return "Scenario 2.2", "OSCAR expired, CoPPER active with different GUID and GFID/GUS combination - Run Sync FLAG to N Job"
            else:
                return "Scenario 2.2 (Variant)", "OSCAR expired but CoPPER active - Run Sync FLAG to N Job"
        
        if oscar_status == 'ACTIVE' and copper_status == 'ACTIVE':
            # Verify all fields match
            if oscar_data.get('guid') == copper_data.get('guid') and \
               oscar_data.get('gfid') == copper_data.get('gfid') and \
               oscar_data.get('gus_id') == copper_data.get('gus_id'):
                return "Match", "All fields match - No action required"
            else:
                return "Mismatch", "Active in both but data mismatch - Verify and reconcile data"
        
        return "Unknown Scenario", "Unable to determine scenario - Manual review required"
    
    def perform_reconciliation(self, guid: str = None, gfid: str = None, 
                              gus_id: str = None) -> Dict:
        """Main reconciliation function"""
        
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'search_criteria': {},
                'oscar_copper_edb': {},
                'oscar_copper_star': {},
                'summary': {},
                'scenario': '',
                'recommended_action': ''
            }
            
            # Determine search mode
            if guid:
                result['search_criteria']['type'] = 'guid'
                result['search_criteria']['guid'] = guid
                
                # Get OSCAR data
                oscar_data = self.get_oscar_data_by_guid(guid)
                oscar_record = oscar_data[0] if oscar_data else {}
                
                # Get CoPPER data
                copper_data = self.get_copper_data_by_guid(guid)
                copper_record = copper_data[0] if copper_data else {}
                
            elif gfid and gus_id:
                result['search_criteria']['type'] = 'gfid_gusid'
                result['search_criteria']['gfid'] = gfid
                result['search_criteria']['gus_id'] = gus_id
                
                # Get OSCAR data
                oscar_data = self.get_oscar_data_by_gfid_gusid(gfid, gus_id)
                oscar_record = oscar_data[0] if oscar_data else {}
                
                # Get CoPPER data
                copper_data = self.get_copper_data_by_gfid_gusid(gfid, gus_id)
                copper_record = copper_data[0] if copper_data else {}
                
            else:
                return {
                    'error': 'Invalid search criteria. Provide either GUID or both GFID and GUS_ID',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Determine scenario
            oscar_status = oscar_record.get('status', 'MISSING')
            copper_status = copper_record.get('status', 'MISSING')
            scenario, action = self.determine_scenario(
                oscar_status, copper_status, oscar_record, copper_record
            )
            
            result['scenario'] = scenario
            result['recommended_action'] = action
            
            # Get GFID and GUS_ID for EDB and STAR queries
            # Prefer CoPPER data if available
            query_gfid = copper_record.get('gfid') or oscar_record.get('gfid')
            query_gus_id = copper_record.get('gus_id') or oscar_record.get('gus_id')
            
            # OSCAR-CoPPER-EDB comparison
            edb_data = []
            if query_gfid and query_gus_id:
                edb_data = self.get_edb_data(query_gfid, query_gus_id)
            
            edb_record = edb_data[0] if edb_data else {}
            
            result['oscar_copper_edb'] = {
                'oscar': oscar_record,
                'copper': copper_record,
                'edb': edb_record,
                'comparison': {
                    'oscar_copper_match': self._compare_records(oscar_record, copper_record),
                    'copper_edb_match': self._compare_records(copper_record, edb_record),
                    'oscar_status': oscar_status,
                    'copper_status': copper_status,
                    'edb_status': edb_record.get('status', 'MISSING')
                }
            }
            
            # OSCAR-CoPPER-STAR comparison
            star_data = []
            if query_gfid:
                star_data = self.get_star_data(query_gfid)
            
            star_record = star_data[0] if star_data else {}
            
            result['oscar_copper_star'] = {
                'oscar': oscar_record,
                'copper': copper_record,
                'star': star_record,
                'comparison': {
                    'oscar_copper_match': self._compare_records(oscar_record, copper_record),
                    'copper_gfid_in_star': bool(star_record),
                    'oscar_status': oscar_status,
                    'copper_status': copper_status,
                    'star_exists': bool(star_record)
                }
            }
            
            # Calculate summary
            result['summary'] = self._calculate_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            }
    
    def _compare_records(self, record1: Dict, record2: Dict) -> str:
        """Compare two records and return match status"""
        if not record1 and not record2:
            return 'BOTH_MISSING'
        if not record1:
            return 'FIRST_MISSING'
        if not record2:
            return 'SECOND_MISSING'
        
        # Compare key fields
        key_fields = ['guid', 'gfid', 'gus_id']
        matches = []
        
        for field in key_fields:
            val1 = record1.get(field)
            val2 = record2.get(field)
            if val1 and val2:
                matches.append(str(val1).strip().upper() == str(val2).strip().upper())
        
        if all(matches) and len(matches) > 0:
            return 'MATCH'
        elif any(matches):
            return 'PARTIAL_MATCH'
        else:
            return 'MISMATCH'
    
    def _calculate_summary(self, result: Dict) -> Dict:
        """Calculate summary statistics"""
        summary = {
            'total_comparisons': 2,  # OSCAR-CoPPER-EDB and OSCAR-CoPPER-STAR
            'matches': 0,
            'mismatches': 0,
            'missing': 0
        }
        
        # Check OSCAR-CoPPER-EDB
        edb_comparison = result['oscar_copper_edb']['comparison']
        if edb_comparison['oscar_copper_match'] == 'MATCH' and \
           edb_comparison['copper_edb_match'] == 'MATCH':
            summary['matches'] += 1
        elif 'MISSING' in edb_comparison['oscar_copper_match'] or \
             'MISSING' in edb_comparison['copper_edb_match']:
            summary['missing'] += 1
        else:
            summary['mismatches'] += 1
        
        # Check OSCAR-CoPPER-STAR
        star_comparison = result['oscar_copper_star']['comparison']
        if star_comparison['oscar_copper_match'] == 'MATCH' and \
           star_comparison['copper_gfid_in_star']:
            summary['matches'] += 1
        elif 'MISSING' in star_comparison['oscar_copper_match'] or \
             not star_comparison['copper_gfid_in_star']:
            summary['missing'] += 1
        else:
            summary['mismatches'] += 1
        
        return summary


# Initialize reconciliation engine
recon_engine = None

def get_recon_engine():
    global recon_engine
    if recon_engine is None:
        recon_engine = ReconciliationEngine()
    return recon_engine


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/reconcile', methods=['POST'])
def reconcile():
    try:
        data = request.get_json()
        
        guid = data.get('guid', '').strip()
        gfid = data.get('gfid', '').strip()
        gus_id = data.get('gus_id', '').strip()
        
        # Validate input
        if not guid and not (gfid and gus_id):
            return jsonify({
                'error': 'Please provide either GUID or both GFID and GUS_ID'
            }), 400
        
        # Get reconciliation engine
        engine = get_recon_engine()
        
        # Perform reconciliation
        result = engine.perform_reconciliation(
            guid=guid if guid else None,
            gfid=gfid if gfid else None,
            gus_id=gus_id if gus_id else None
        )
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    try:
        engine = get_recon_engine()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'databases': {
                'oscar': 'connected',
                'copper': 'connected',
                'edb': 'connected',
                'star': 'connected'
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSCAR Reconciliation Tool</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #1a365d;
            --primary-light: #2c5aa0;
            --accent-color: #00b4d8;
            --success-color: #38a169;
            --warning-color: #ed8936;
            --error-color: #e53e3e;
            --white: #ffffff;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
            --spacing-2xl: 3rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --radius-xl: 1rem;
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            --transition-normal: 0.3s ease-in-out;
        }

        body {
            font-family: var(--font-family);
            font-size: 1rem;
            line-height: 1.6;
            color: var(--gray-800);
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 var(--spacing-lg);
        }

        .header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
            color: var(--white);
            padding: var(--spacing-lg) 0;
            box-shadow: var(--shadow-lg);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: var(--spacing-md);
        }

        .logo i {
            font-size: 1.5rem;
            color: var(--accent-color);
        }

        .logo h1 {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            background: rgba(255, 255, 255, 0.1);
            padding: var(--spacing-sm) var(--spacing-md);
            border-radius: var(--radius-lg);
            backdrop-filter: blur(10px);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success-color);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-content {
            padding: var(--spacing-2xl) 0;
            min-height: calc(100vh - 200px);
        }

        .welcome-section {
            text-align: center;
            margin-bottom: var(--spacing-2xl);
        }

        .welcome-content h2 {
            font-size: 1.875rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: var(--spacing-md);
        }

        .welcome-content p {
            font-size: 1.125rem;
            color: var(--gray-600);
            max-width: 600px;
            margin: 0 auto;
        }

        .search-section {
            margin-bottom: var(--spacing-2xl);
        }

        .search-card {
            background: var(--white);
            border-radius: var(--radius-xl);
            padding: var(--spacing-2xl);
            box-shadow: var(--shadow-xl);
            border: 1px solid var(--gray-200);
        }

        .search-header {
            text-align: center;
            margin-bottom: var(--spacing-xl);
        }

        .search-header h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: var(--spacing-sm);
        }

        .search-header h3 i {
            margin-right: var(--spacing-sm);
            color: var(--accent-color);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: var(--spacing-lg);
            margin-bottom: var(--spacing-xl);
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-sm);
        }

        .form-group label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--gray-700);
        }

        .form-group input {
            padding: var(--spacing-md);
            border: 2px solid var(--gray-300);
            border-radius: var(--radius-lg);
            font-size: 1rem;
            transition: var(--transition-normal);
            background: var(--white);
        }

        .form-group input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
        }

        .input-info {
            font-size: 0.75rem;
            color: var(--gray-500);
        }

        .divider {
            text-align: center;
            margin: var(--spacing-xl) 0;
            position: relative;
        }

        .divider::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            width: 45%;
            height: 1px;
            background: var(--gray-300);
        }

        .divider::after {
            content: '';
            position: absolute;
            right: 0;
            top: 50%;
            width: 45%;
            height: 1px;
            background: var(--gray-300);
        }

        .divider span {
            background: var(--white);
            padding: 0 var(--spacing-md);
            color: var(--gray-500);
            font-weight: 500;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
            color: var(--white);
            border: none;
            padding: var(--spacing-md) var(--spacing-xl);
            border-radius: var(--radius-lg);
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition-normal);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: var(--spacing-sm);
            box-shadow: var(--shadow-md);
            white-space: nowrap;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: var(--white);
            color: var(--gray-700);
            border: 2px solid var(--gray-300);
            padding: var(--spacing-sm) var(--spacing-lg);
            border-radius: var(--radius-lg);
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition-normal);
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
        }

        .btn-secondary:hover {
            border-color: var(--primary-color);
            color: var(--primary-color);
            transform: translateY(-1px);
        }

        .submit-container {
            text-align: center;
            margin-top: var(--spacing-xl);
        }

        .results-section {
            display: none;
            animation: fadeInUp 0.6s ease-out;
        }

        .results-section.show {
            display: block;
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

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-xl);
        }

        .results-header h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--primary-color);
        }

        .results-header h3 i {
            margin-right: var(--spacing-sm);
            color: var(--accent-color);
        }

        .results-actions {
            display: flex;
            gap: var(--spacing-md);
        }

        .scenario-card {
            background: var(--white);
            border-radius: var(--radius-xl);
            padding: var(--spacing-xl);
            box-shadow: var(--shadow-lg);
            margin-bottom: var(--spacing-xl);
            border-left: 4px solid var(--accent-color);
        }

        .scenario-card h4 {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: var(--spacing-md);
        }

        .scenario-card p {
            color: var(--gray-700);
            line-height: 1.6;
        }

        .comparison-tables {
            display: grid;
            grid-template-columns: 1fr;
            gap: var(--spacing-2xl);
            margin-bottom: var(--spacing-xl);
        }

        .table-section {
            background: var(--white);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-xl);
            overflow: hidden;
            border: 1px solid var(--gray-200);
        }

        .table-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
            color: var(--white);
            padding: var(--spacing-lg);
            text-align: center;
        }

        .table-header h4 {
            font-size: 1.125rem;
            font-weight: 600;
            margin: 0;
        }

        .table-container {
            overflow-x: auto;
            max-height: 600px;
            overflow-y: auto;
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
        }

        .comparison-table th {
            background: var(--gray-100);
            padding: var(--spacing-md);
            text-align: left;
            font-weight: 600;
            color: var(--gray-700);
            border-bottom: 2px solid var(--gray-200);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .comparison-table td {
            padding: var(--spacing-md);
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
            vertical-align: middle;
        }

        .comparison-table tbody tr:hover {
            background: var(--gray-50);
        }

        .db-column {
            font-weight: 500;
        }

        .oscar-col {
            background: rgba(26, 54, 93, 0.05);
        }

        .copper-col {
            background: rgba(0, 180, 216, 0.05);
        }

        .star-col {
            background: rgba(56, 161, 105, 0.05);
        }

        .edb-col {
            background: rgba(237, 137, 54, 0.05);
        }

        .status-badge {
            padding: 0.25rem 0.5rem;
            border-radius: var(--radius-md);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            display: inline-block;
        }

        .status-match {
            background: var(--success-color);
            color: var(--white);
        }

        .status-mismatch {
            background: var(--error-color);
            color: var(--white);
        }

        .status-missing {
            background: var(--warning-color);
            color: var(--white);
        }

        .status-active {
            background: var(--success-color);
            color: var(--white);
        }

        .status-inactive {
            background: var(--gray-500);
            color: var(--white);
        }

        .alert {
            padding: var(--spacing-md);
            border-radius: var(--radius-lg);
            margin-bottom: var(--spacing-lg);
        }

        .alert-error {
            background: #fee;
            border: 1px solid var(--error-color);
            color: var(--error-color);
        }

        .alert-success {
            background: #efe;
            border: 1px solid var(--success-color);
            color: var(--success-color);
        }

        .loading {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid var(--white);
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .container {
                padding: 0 var(--spacing-md);
            }
            
            .header-content {
                flex-direction: column;
                gap: var(--spacing-md);
            }
            
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .results-header {
                flex-direction: column;
                gap: var(--spacing-md);
                align-items: stretch;
            }
            
            .results-actions {
                justify-content: center;
            }
            
            .comparison-table {
                font-size: 0.875rem;
            }
            
            .comparison-table th,
            .comparison-table td {
                padding: var(--spacing-sm);
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-exchange-alt"></i>
                    <h1>OSCAR Reconcile</h1>
                </div>
                <div class="connection-status">
                    <div class="status-dot"></div>
                    <span>Connected</span>
                </div>
            </div>
        </div>
    </header>

    <main class="main-content">
        <div class="container">
            <section class="welcome-section">
                <div class="welcome-content">
                    <h2>Data Reconciliation Between OSCAR, CoPPER, STAR & EDB</h2>
                    <p>Compare and synchronize financial trading data across multiple systems</p>
                </div>
            </section>

            <section class="search-section">
                <div class="search-card">
                    <div class="search-header">
                        <h3><i class="fas fa-search"></i> Reconciliation Parameters</h3>
                        <p>Search by GUID or by GFID + GUS ID combination</p>
                    </div>
                    
                    <form id="reconcile-form">
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="guid">
                                    <i class="fas fa-key"></i> GUID (12 chars)
                                </label>
                                <input type="text" id="guid" name="guid" placeholder="e.g., ABCDEFGH1234" maxlength="12">
                                <div class="input-info">Global Unique Identifier</div>
                            </div>
                        </div>

                        <div class="divider">
                            <span>OR</span>
                        </div>

                        <div class="form-grid">
                            <div class="form-group">
                                <label for="gfid">
                                    <i class="fas fa-building"></i> GFID (4 chars)
                                </label>
                                <input type="text" id="gfid" name="gfid" placeholder="e.g., ABCD" maxlength="4">
                                <div class="input-info">Globex Firm ID</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="gus-id">
                                    <i class="fas fa-user"></i> GUS ID (5 chars)
                                </label>
                                <input type="text" id="gus-id" name="gus_id" placeholder="e.g., ABCDE" maxlength="5">
                                <div class="input-info">Globex User Signature ID</div>
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

            <section class="results-section" id="results-section">
                <div class="results-header">
                    <h3><i class="fas fa-chart-bar"></i> Reconciliation Results</h3>
                    <div class="results-actions">
                        <button class="btn-secondary" id="export-btn">
                            <i class="fas fa-download"></i> Export Results
                        </button>
                        <button class="btn-secondary" id="clear-results-btn">
                            <i class="fas fa-times"></i> Clear Results
                        </button>
                    </div>
                </div>

                <div id="error-container"></div>

                <div class="scenario-card" id="scenario-card">
                    <h4><i class="fas fa-lightbulb"></i> Scenario & Recommended Action</h4>
                    <p id="scenario-text"></p>
                    <p style="margin-top: 0.5rem; font-weight: 600;" id="action-text"></p>
                </div>

                <div class="comparison-tables">
                    <div class="table-section">
                        <div class="table-header">
                            <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER ↔ EDB Comparison</h4>
                        </div>
                        <div class="table-container">
                            <table class="comparison-table" id="edb-table">
                                <thead>
                                    <tr>
                                        <th>Field</th>
                                        <th class="oscar-col">OSCAR</th>
                                        <th class="copper-col">CoPPER</th>
                                        <th class="edb-col">EDB</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody id="edb-table-body">
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div class="table-section">
                        <div class="table-header">
                            <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER ↔ STAR Comparison</h4>
                        </div>
                        <div class="table-container">
                            <table class="comparison-table" id="star-table">
                                <thead>
                                    <tr>
                                        <th>Field</th>
                                        <th class="oscar-col">OSCAR</th>
                                        <th class="copper-col">CoPPER</th>
                                        <th class="star-col">STAR</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody id="star-table-body">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <script>
        let currentResults = null;

        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('reconcile-form');
            const resultsSection = document.getElementById('results-section');
            const submitBtn = document.getElementById('submit-btn');
            
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const guid = document.getElementById('guid').value.trim();
                const gfid = document.getElementById('gfid').value.trim();
                const gus_id = document.getElementById('gus-id').value.trim();
                
                if (!guid && (!gfid || !gus_id)) {
                    showError('Please provide either GUID or both GFID and GUS ID');
                    return;
                }
                
                if (guid && (gfid || gus_id)) {
                    showError('Please provide either GUID or GFID+GUS ID, not both');
                    return;
                }
                
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<div class="loading"></div> Processing...';
                
                try {
                    const response = await fetch('/api/reconcile', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ guid, gfid, gus_id })
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.error || 'Reconciliation failed');
                    }
                    
                    currentResults = data;
                    displayResults(data);
                    resultsSection.classList.add('show');
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                    
                } catch (error) {
                    showError(error.message);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Execute Reconciliation';
                }
            });
            
            document.getElementById('clear-results-btn').addEventListener('click', function() {
                resultsSection.classList.remove('show');
                form.reset();
                currentResults = null;
                clearError();
            });
            
            document.getElementById('export-btn').addEventListener('click', function() {
                if (currentResults) {
                    downloadJSON(currentResults, `oscar_reconciliation_${new Date().toISOString().split('T')[0]}.json`);
                }
            });
        });
        
        function displayResults(data) {
            clearError();
            
            document.getElementById('scenario-text').textContent = `Scenario: ${data.scenario}`;
            document.getElementById('action-text').textContent = `Recommended Action: ${data.recommended_action}`;
            
            displayEDBTable(data.oscar_copper_edb);
            displaySTARTable(data.oscar_copper_star);
        }
        
        function displayEDBTable(data) {
            const tbody = document.getElementById('edb-table-body');
            tbody.innerHTML = '';
            
            const fields = ['guid', 'gfid', 'gus_id', 'status', 'eff_to'];
            const fieldLabels = {
                'guid': 'GUID',
                'gfid': 'GFID',
                'gus_id': 'GUS ID',
                'status': 'Status',
                'eff_to': 'Expiry Date'
            };
            
            fields.forEach(field => {
                const row = document.createElement('tr');
                
                const oscarVal = data.oscar[field] || '-';
                const copperVal = data.copper[field] || '-';
                const edbVal = data.edb[field] || '-';
                
                const status = getFieldStatus(oscarVal, copperVal, edbVal);
                
                row.innerHTML = `
                    <td class="db-column">${fieldLabels[field] || field}</td>
                    <td class="oscar-col">${formatValue(oscarVal, field)}</td>
                    <td class="copper-col">${formatValue(copperVal, field)}</td>
                    <td class="edb-col">${formatValue(edbVal, field)}</td>
                    <td>${status}</td>
                `;
                
                tbody.appendChild(row);
            });
            
            const summaryRow = document.createElement('tr');
            summaryRow.innerHTML = `
                <td class="db-column" style="font-weight: 600;">Overall Status</td>
                <td class="oscar-col">${formatValue(data.comparison.oscar_status, 'status')}</td>
                <td class="copper-col">${formatValue(data.comparison.copper_status, 'status')}</td>
                <td class="edb-col">${formatValue(data.comparison.edb_status, 'status')}</td>
                <td>${getOverallStatus(data.comparison.oscar_copper_match, data.comparison.copper_edb_match)}</td>
            `;
            tbody.appendChild(summaryRow);
        }
        
        function displaySTARTable(data) {
            const tbody = document.getElementById('star-table-body');
            tbody.innerHTML = '';
            
            const oscarFields = ['guid', 'gfid', 'gus_id', 'status', 'exchange'];
            const copperFields = ['guid', 'gfid', 'gus_id', 'status', 'exch_id'];
            const starFields = ['gfid', 'gfid_description', 'exchange_id'];
            
            const fieldLabels = {
                'guid': 'GUID',
                'gfid': 'GFID',
                'gus_id': 'GUS ID',
                'status': 'Status',
                'exchange': 'Exchange',
                'exch_id': 'Exchange ID',
                'gfid_description': 'Description'
            };
            
            const allFields = [...new Set([...oscarFields, ...copperFields, ...starFields])];
            
            allFields.forEach(field => {
                const row = document.createElement('tr');
                
                const oscarVal = data.oscar[field] || '-';
                const copperVal = data.copper[field] || '-';
                const starVal = data.star[field] || '-';
                
                const status = getFieldStatus(oscarVal, copperVal, starVal);
                
                row.innerHTML = `
                    <td class="db-column">${fieldLabels[field] || field}</td>
                    <td class="oscar-col">${formatValue(oscarVal, field)}</td>
                    <td class="copper-col">${formatValue(copperVal, field)}</td>
                    <td class="star-col">${formatValue(starVal, field)}</td>
                    <td>${status}</td>
                `;
                
                tbody.appendChild(row);
            });
            
            const summaryRow = document.createElement('tr');
            summaryRow.innerHTML = `
                <td class="db-column" style="font-weight: 600;">Overall Status</td>
                <td class="oscar-col">${formatValue(data.comparison.oscar_status, 'status')}</td>
                <td class="copper-col">${formatValue(data.comparison.copper_status, 'status')}</td>
                <td class="star-col">${data.comparison.star_exists ? 'EXISTS' : 'NOT FOUND'}</td>
                <td>${getOverallStatus(data.comparison.oscar_copper_match, data.comparison.copper_gfid_in_star ? 'MATCH' : 'MISSING')}</td>
            `;
            tbody.appendChild(summaryRow);
        }
        
        function formatValue(value, field) {
            if (value === null || value === undefined || value === '') {
                return '-';
            }
            
            if (field === 'status') {
                const statusClass = value.toLowerCase().includes('active') ? 'status-active' : 
                                   value.toLowerCase().includes('inactive') ? 'status-inactive' :
                                   value.toLowerCase().includes('missing') ? 'status-missing' : '';
                return `<span class="status-badge ${statusClass}">${value}</span>`;
            }
            
            return value;
        }
        
        function getFieldStatus(val1, val2, val3) {
            const values = [val1, val2, val3].filter(v => v !== '-' && v !== null && v !== undefined);
            
            if (values.length === 0) {
                return '<span class="status-badge status-missing">ALL MISSING</span>';
            }
            
            if (values.length < 3) {
                return '<span class="status-badge status-missing">PARTIAL</span>';
            }
            
            const allSame = values.every(v => String(v).trim().toUpperCase() === String(values[0]).trim().toUpperCase());
            
            if (allSame) {
                return '<span class="status-badge status-match">MATCH</span>';
            } else {
                return '<span class="status-badge status-mismatch">MISMATCH</span>';
            }
        }
        
        function getOverallStatus(match1, match2) {
            if (match1 === 'MATCH' && (match2 === 'MATCH' || match2 === true)) {
                return '<span class="status-badge status-match">MATCH</span>';
            } else if (match1.includes('MISSING') || match2.includes('MISSING') || match2 === false) {
                return '<span class="status-badge status-missing">MISSING</span>';
            } else {
                return '<span class="status-badge status-mismatch">MISMATCH</span>';
                }
        }
        
        function showError(message) {
            const errorContainer = document.getElementById('error-container');
            errorContainer.innerHTML = `
                <div class="alert alert-error">
                    <i class="fas fa-exclamation-circle"></i> ${message}
                </div>
            `;
        }
        
        function clearError() {
            const errorContainer = document.getElementById('error-container');
            errorContainer.innerHTML = '';
        }
        
        function downloadJSON(data, filename) {
            const blob = new Blob([JSON.stringify(data, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            showToast('Results exported successfully', 'success');
        }
        
        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                padding: 1rem 1.5rem;
                border-radius: 0.5rem;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                border-left: 4px solid var(--${type === 'success' ? 'success' : 'primary'}-color);
                z-index: 1000;
                animation: slideInRight 0.3s ease-out;
            `;
            
            const colors = {
                success: 'var(--success-color)',
                error: 'var(--error-color)',
                warning: 'var(--warning-color)',
                info: 'var(--accent-color)'
            };
            
            toast.style.borderLeftColor = colors[type] || colors.info;
            toast.textContent = message;
            
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'slideOutRight 0.3s ease-out';
                setTimeout(() => {
                    document.body.removeChild(toast);
                }, 300);
            }, 3000);
        }
        
        const style = document.createElement('style');
        style.textContent = `
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
    </script>
</body>
</html>
