oscar_tool/
├── app.py                 # Flask backend
├── database.py           # PostgreSQL connections (using your existing code)
├── static/
│   ├── style.css         # Simple styling
│   └── script.js         # Frontend logic
├── templates/
│   └── index.html        # Main UI
└── config.py             # Database configurations











oscar-reconciliation-tool/
├── 📄 app.py                          # Main Flask application
├── 📄 database.py                     # PostgreSQL database manager
├── 📄 config.py                       # Configuration & scenarios
├── 📄 run.py                          # Application startup script
├── 📄 requirements.txt                # Python dependencies
├── 📄 .env.template                   # Environment variables template
├── 📄 .env                            # Your actual environment variables (create from template)
├── 📄 README.md                       # Complete documentation
├── 📄 setup.sh                        # Automated setup script
├── 📄 Dockerfile                      # Docker container config
├── 📄 docker-compose.yml              # Docker development environment
├── 📄 .gitignore                      # Git ignore file (recommended)
│
├── 📁 templates/                      # Flask HTML templates
│   └── 📄 index.html                  # Main dashboard page
│
├── 📁 static/                         # Static frontend assets
│   ├── 📄 style.css                   # Main CSS styles
│   ├── 📄 animations.css              # Animation & effects CSS
│   ├── 📄 script.js                   # Frontend JavaScript
│   └── 📁 uploads/                    # File uploads (auto-created)
│
├── 📁 logs/                           # Application logs (auto-created)
│   └── 📄 oscar_tool.log              # Main log file
│
├── 📁 database/                       # Database initialization (auto-created by setup)
│   ├── 📄 oscar_init.sql              # OSCAR DB sample schema
│   └── 📄 copper_init.sql             # CoPPER DB sample schema
│
├── 📁 tests/ (recommended)            # Unit tests
│   ├── 📄 __init__.py
│   ├── 📄 test_app.py
│   ├── 📄 test_database.py
│   └── 📄 test_reconciliation.py
│
├── 📁 nginx/ (for production)         # Nginx configuration
│   ├── 📄 nginx.conf
│   └── 📁 ssl/
│
├── 📁 monitoring/ (optional)          # Monitoring configs
│   ├── 📄 prometheus.yml
│   └── 📁 grafana/
│       ├── 📁 dashboards/
│       └── 📁 datasources/
│
└── 📁 venv/                           # Python virtual environment (auto-created)
    ├── 📁 bin/
    ├── 📁 lib/
    └── 📁 include/
















from flask import Flask, render_template, request, jsonify
from database import DatabaseManager
import json
import logging
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database manager
db_manager = DatabaseManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/reconcile', methods=['POST'])
def reconcile_data():
    """Main reconciliation endpoint"""
    try:
        data = request.get_json()
        input_value = data.get('input_value', '').strip()
        scenario_type = data.get('scenario_type', 'guid_lookup')
        
        if not input_value:
            return jsonify({'error': 'Input value is required'}), 400
        
        # Determine input type based on length
        input_type = determine_input_type(input_value)
        
        # Execute reconciliation based on scenario
        result = execute_reconciliation(input_value, input_type, scenario_type)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Reconciliation error: {str(e)}")
        return jsonify({'error': f'Reconciliation failed: {str(e)}'}), 500

def determine_input_type(value):
    """Determine if input is GUID, GUS, or GFID based on length"""
    length = len(value)
    if length == 12:
        return 'GUID'
    elif length == 5:
        return 'GUS'
    elif length == 4:
        return 'GFID'
    else:
        return 'UNKNOWN'

def execute_reconciliation(input_value, input_type, scenario_type):
    """Execute reconciliation logic"""
    try:
        # Get OSCAR data
        oscar_data = get_oscar_data(input_value, input_type)
        
        # Get CoPPER data
        copper_data = get_copper_data(input_value, input_type)
        
        # Perform comparison
        comparison = compare_data(oscar_data, copper_data)
        
        # Determine scenario and actions
        scenario = determine_scenario(oscar_data, copper_data, comparison)
        
        return {
            'success': True,
            'input_value': input_value,
            'input_type': input_type,
            'oscar_data': oscar_data,
            'copper_data': copper_data,
            'comparison': comparison,
            'scenario': scenario,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Reconciliation execution failed: {str(e)}")

def get_oscar_data(input_value, input_type):
    """Fetch data from OSCAR database"""
    try:
        if input_type == 'GUID':
            query = """
            SELECT distinct EXTRACTVALUE(XML,'//expDate') as exp_date,
            EXTRACTVALUE(XML,'//idNumber') as id_number, 
            EXTRACTVALUE(XML,'//globexFirmID') as globex_firm_id,
            EXTRACTVALUE(XML,'//status') as status
            FROM PR01COSRO.ACTIVE_XML_DATA_STORE WHERE GUID = %s
            """
        else:
            query = """
            SELECT * FROM PR01COSRO.XML_DATA_INTERIM 
            WHERE version='14.01' and GUID = %s
            """
        
        result = db_manager.execute_oscar_query(query, (input_value,))
        
        return {
            'found': len(result) > 0,
            'count': len(result),
            'data': result,
            'status': result[0].get('status', 'N/A') if result else 'NOT_FOUND'
        }
        
    except Exception as e:
        logger.error(f"OSCAR query error: {str(e)}")
        return {'found': False, 'count': 0, 'data': [], 'status': 'ERROR', 'error': str(e)}

def get_copper_data(input_value, input_type):
    """Fetch data from CoPPER database"""
    try:
        if input_type == 'GUID':
            # Multiple queries to check different tables
            queries = [
                "SELECT * FROM pr01ccrdo.trd_guid WHERE GFID_ID=%s and GPID_ID='CAIBE' and GFID_GUID='VGARC'",
                "SELECT * FROM pr01ccrdo.trd_guid WHERE GFID_ID=%s and GPID_ID='CAIBE' and GFID_GUID='JRCOS'",
                "SELECT * FROM NR01CCRDO.TRD_SESSION_PRODUCT_PERMISSION WHERE SESSION_ID in ('MDBLZ','FIF')"
            ]
        else:
            queries = [
                "SELECT * FROM NR01CEBDO.GB_SESSION WHERE SESSION_ID in ('MDBLZ','FIF')"
            ]
        
        all_results = []
        for query in queries:
            result = db_manager.execute_copper_query(query, (input_value,))
            all_results.extend(result)
        
        return {
            'found': len(all_results) > 0,
            'count': len(all_results),
            'data': all_results,
            'status': 'ACTIVE' if all_results else 'NOT_FOUND'
        }
        
    except Exception as e:
        logger.error(f"CoPPER query error: {str(e)}")
        return {'found': False, 'count': 0, 'data': [], 'status': 'ERROR', 'error': str(e)}

def compare_data(oscar_data, copper_data):
    """Compare OSCAR and CoPPER data"""
    comparison = {
        'match_status': 'UNKNOWN',
        'differences': [],
        'summary': ''
    }
    
    oscar_found = oscar_data['found']
    copper_found = copper_data['found']
    oscar_status = oscar_data.get('status', 'UNKNOWN')
    copper_status = copper_data.get('status', 'UNKNOWN')
    
    if not oscar_found and not copper_found:
        comparison['match_status'] = 'BOTH_MISSING'
        comparison['summary'] = 'Data not found in both OSCAR and CoPPER'
    elif oscar_found and not copper_found:
        comparison['match_status'] = 'COPPER_MISSING'
        comparison['summary'] = 'Data exists in OSCAR but missing in CoPPER'
        comparison['differences'].append('CoPPER data missing')
    elif not oscar_found and copper_found:
        comparison['match_status'] = 'OSCAR_MISSING'
        comparison['summary'] = 'Data exists in CoPPER but missing in OSCAR'
        comparison['differences'].append('OSCAR data missing')
    else:
        # Both found, compare statuses
        if oscar_status == copper_status:
            comparison['match_status'] = 'MATCH'
            comparison['summary'] = 'Data matches between OSCAR and CoPPER'
        else:
            comparison['match_status'] = 'STATUS_MISMATCH'
            comparison['summary'] = f'Status mismatch: OSCAR({oscar_status}) vs CoPPER({copper_status})'
            comparison['differences'].append(f'Status: OSCAR={oscar_status}, CoPPER={copper_status}')
    
    return comparison

def determine_scenario(oscar_data, copper_data, comparison):
    """Determine reconciliation scenario and recommended actions"""
    scenario = {
        'type': 'UNKNOWN',
        'description': '',
        'recommended_actions': [],
        'severity': 'LOW'
    }
    
    match_status = comparison['match_status']
    oscar_status = oscar_data.get('status', 'UNKNOWN')
    copper_status = copper_data.get('status', 'UNKNOWN')
    
    if match_status == 'BOTH_MISSING':
        scenario.update({
            'type': 'BOTH_MISSING',
            'description': 'Data not found in either system',
            'recommended_actions': ['Verify input value', 'Check data creation'],
            'severity': 'HIGH'
        })
    
    elif match_status == 'COPPER_MISSING':
        if oscar_status in ['EXPIRED', 'INACTIVE']:
            scenario.update({
                'type': 'SCENARIO_2.3',
                'description': 'Expired GUID in OSCAR and GUID missing in CoPPER',
                'recommended_actions': ['Run Sync FLAG to N Job'],
                'severity': 'MEDIUM'
            })
        else:
            scenario.update({
                'type': 'SCENARIO_2.4',
                'description': 'Active GUID exists in OSCAR but missing in CoPPER',
                'recommended_actions': ['Investigate why CoPPER data is missing', 'Manual sync required'],
                'severity': 'HIGH'
            })
    
    elif match_status == 'STATUS_MISMATCH':
        if oscar_status == 'EXPIRED' and copper_status == 'EXPIRED':
            scenario.update({
                'type': 'SCENARIO_2.1',
                'description': 'Expired GUID in both OSCAR and CoPPER',
                'recommended_actions': ['Run MASS SYNC job', 'Run Sync FLAG to N Job'],
                'severity': 'MEDIUM'
            })
        elif oscar_status == 'EXPIRED' and copper_status == 'ACTIVE':
            scenario.update({
                'type': 'SCENARIO_2.2',
                'description': 'Expired GUID in OSCAR but Active in CoPPER',
                'recommended_actions': ['Run Sync FLAG to N Job'],
                'severity': 'MEDIUM'
            })
    
    elif match_status == 'MATCH':
        scenario.update({
            'type': 'SYNC',
            'description': 'Data is synchronized between systems',
            'recommended_actions': ['No action required'],
            'severity': 'LOW'
        })
    
    return scenario

@app.route('/api/execute_action', methods=['POST'])
def execute_action():
    """Execute reconciliation actions"""
    try:
        data = request.get_json()
        action = data.get('action')
        input_value = data.get('input_value')
        
        # Simulate action execution
        result = {
            'success': True,
            'action': action,
            'input_value': input_value,
            'message': f'Successfully executed {action} for {input_value}',
            'timestamp': datetime.now().isoformat()
        }
        
        # In real implementation, this would trigger actual sync jobs
        logger.info(f"Executed action: {action} for {input_value}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Action execution error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database_status': db_manager.check_connection()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




















import psycopg2
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine
import logging
from config import DATABASE_CONFIG

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.oscar_engine = None
        self.copper_engine = None
        self.initialize_connections()
    
    def initialize_connections(self):
        """Initialize database connections"""
        try:
            # OSCAR Database Connection
            oscar_url = f"postgresql+psycopg2://{DATABASE_CONFIG['oscar']['user']}:{DATABASE_CONFIG['oscar']['password']}@{DATABASE_CONFIG['oscar']['host']}:{DATABASE_CONFIG['oscar']['port']}/{DATABASE_CONFIG['oscar']['database']}"
            self.oscar_engine = create_engine(oscar_url)
            
            # CoPPER Database Connection  
            copper_url = f"postgresql+psycopg2://{DATABASE_CONFIG['copper']['user']}:{DATABASE_CONFIG['copper']['password']}@{DATABASE_CONFIG['copper']['host']}:{DATABASE_CONFIG['copper']['port']}/{DATABASE_CONFIG['copper']['database']}"
            self.copper_engine = create_engine(copper_url)
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def get_oscar_connection(self):
        """Get OSCAR database connection"""
        try:
            return psycopg2.connect(
                host=DATABASE_CONFIG['oscar']['host'],
                port=DATABASE_CONFIG['oscar']['port'],
                database=DATABASE_CONFIG['oscar']['database'],
                user=DATABASE_CONFIG['oscar']['user'],
                password=DATABASE_CONFIG['oscar']['password'],
                cursor_factory=RealDictCursor
            )
        except Exception as e:
            logger.error(f"OSCAR connection failed: {str(e)}")
            raise
    
    def get_copper_connection(self):
        """Get CoPPER database connection"""
        try:
            return psycopg2.connect(
                host=DATABASE_CONFIG['copper']['host'],
                port=DATABASE_CONFIG['copper']['port'],
                database=DATABASE_CONFIG['copper']['database'],
                user=DATABASE_CONFIG['copper']['user'],
                password=DATABASE_CONFIG['copper']['password'],
                cursor_factory=RealDictCursor
            )
        except Exception as e:
            logger.error(f"CoPPER connection failed: {str(e)}")
            raise
    
    def execute_oscar_query(self, query, params=None):
        """Execute query on OSCAR database"""
        connection = None
        cursor = None
        try:
            connection = self.get_oscar_connection()
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"OSCAR query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def execute_copper_query(self, query, params=None):
        """Execute query on CoPPER database"""
        connection = None
        cursor = None
        try:
            connection = self.get_copper_connection()
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"CoPPER query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def execute_batch_queries(self, queries, database='oscar'):
        """Execute multiple queries in batch"""
        results = []
        for query_info in queries:
            query = query_info.get('query')
            params = query_info.get('params')
            
            if database == 'oscar':
                result = self.execute_oscar_query(query, params)
            else:
                result = self.execute_copper_query(query, params)
            
            results.append({
                'query': query,
                'params': params,
                'result': result,
                'count': len(result)
            })
        
        return results
    
    def run_sync_job(self, job_type, params):
        """Execute synchronization jobs"""
        try:
            if job_type == 'MASS_SYNC':
                return self.run_mass_sync(params)
            elif job_type == 'SYNC_FLAG_N':
                return self.run_sync_flag_n(params)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
                
        except Exception as e:
            logger.error(f"Sync job execution failed: {str(e)}")
            raise
    
    def run_mass_sync(self, params):
        """Run MASS SYNC job"""
        # Implementation for mass sync
        logger.info(f"Running MASS SYNC with params: {params}")
        
        # Simulate sync operation
        return {
            'job_type': 'MASS_SYNC',
            'status': 'COMPLETED',
            'records_processed': 150,
            'records_updated': 45,
            'execution_time': '2.5 seconds'
        }
    
    def run_sync_flag_n(self, params):
        """Run Sync FLAG to N Job"""
        # Implementation for sync flag to N
        logger.info(f"Running Sync FLAG to N with params: {params}")
        
        # Simulate sync operation
        return {
            'job_type': 'SYNC_FLAG_N',
            'status': 'COMPLETED',
            'records_processed': 75,
            'records_updated': 25,
            'execution_time': '1.8 seconds'
        }
    
    def check_connection(self):
        """Check database connection health"""
        status = {
            'oscar': False,
            'copper': False
        }
        
        try:
            # Test OSCAR connection
            oscar_conn = self.get_oscar_connection()
            cursor = oscar_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            oscar_conn.close()
            status['oscar'] = True
        except Exception as e:
            logger.error(f"OSCAR health check failed: {str(e)}")
        
        try:
            # Test CoPPER connection
            copper_conn = self.get_copper_connection()
            cursor = copper_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            copper_conn.close()
            status['copper'] = True
        except Exception as e:
            logger.error(f"CoPPER health check failed: {str(e)}")
        
        return status
    
    def get_table_info(self, table_name, database='oscar'):
        """Get table structure information"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        
        if database == 'oscar':
            return self.execute_oscar_query(query, (table_name,))
        else:
            return self.execute_copper_query(query, (table_name,))
    
    def get_record_count(self, table_name, database='oscar', where_clause=None):
        """Get record count for a table"""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if database == 'oscar':
            result = self.execute_oscar_query(query)
        else:
            result = self.execute_copper_query(query)
        
        return result[0]['count'] if result else 0
    
    def execute_transaction(self, queries, database='oscar'):
        """Execute multiple queries in a transaction"""
        connection = None
        cursor = None
        try:
            if database == 'oscar':
                connection = self.get_oscar_connection()
            else:
                connection = self.get_copper_connection()
            
            cursor = connection.cursor()
            
            # Start transaction
            connection.autocommit = False
            
            results = []
            for query_info in queries:
                query = query_info.get('query')
                params = query_info.get('params')
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    results.append(cursor.fetchall())
                else:
                    results.append(cursor.rowcount)
            
            # Commit transaction
            connection.commit()
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Transaction failed: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

















import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration
DATABASE_CONFIG = {
    'oscar': {
        'host': os.getenv('OSCAR_DB_HOST', 'localhost'),
        'port': os.getenv('OSCAR_DB_PORT', '5432'),
        'database': os.getenv('OSCAR_DB_NAME', 'oscar_db'),
        'user': os.getenv('OSCAR_DB_USER', 'oscar_user'),
        'password': os.getenv('OSCAR_DB_PASSWORD', 'oscar_password'),
        'schema': os.getenv('OSCAR_DB_SCHEMA', 'PR01COSRO')
    },
    'copper': {
        'host': os.getenv('COPPER_DB_HOST', 'localhost'),
        'port': os.getenv('COPPER_DB_PORT', '5432'),
        'database': os.getenv('COPPER_DB_NAME', 'copper_db'),
        'user': os.getenv('COPPER_DB_USER', 'copper_user'),
        'password': os.getenv('COPPER_DB_PASSWORD', 'copper_password'),
        'schema': os.getenv('COPPER_DB_SCHEMA', 'pr01ccrdo')
    }
}

# Application Configuration
APP_CONFIG = {
    'secret_key': os.getenv('SECRET_KEY', 'your-secret-key-change-in-production'),
    'debug': os.getenv('DEBUG', 'True').lower() == 'true',
    'host': os.getenv('HOST', '0.0.0.0'),
    'port': int(os.getenv('PORT', '5000'))
}

# Cloud SQL Proxy Configuration (if using Google Cloud)
CLOUD_SQL_CONFIG = {
    'instance_connection_name': os.getenv('CLOUD_SQL_CONNECTION_NAME', ''),
    'enable_iam_auth': os.getenv('ENABLE_IAM_AUTH', 'True').lower() == 'true',
    'ip_type': os.getenv('IP_TYPE', 'private')
}

# Reconciliation Configuration
RECONCILIATION_CONFIG = {
    'max_records_per_query': int(os.getenv('MAX_RECORDS_PER_QUERY', '1000')),
    'query_timeout': int(os.getenv('QUERY_TIMEOUT', '30')),
    'batch_size': int(os.getenv('BATCH_SIZE', '100'))
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.getenv('LOG_FILE', 'oscar_tool.log')
}

# Scenario Definitions
SCENARIOS = {
    'scenario_2_1': {
        'name': 'Expired GUID in OSCAR and CoPPER',
        'description': 'Both systems have expired GUID with same GFID and GUS ID',
        'oscar_query': """
            SELECT distinct EXTRACTVALUE(XML,'//expDate') as exp_date,
            EXTRACTVALUE(XML,'//idNumber') as id_number, 
            EXTRACTVALUE(XML,'//globexFirmID') as globex_firm_id,
            EXTRACTVALUE(XML,'//status') as status
            FROM PR01COSRO.ACTIVE_XML_DATA_STORE WHERE GUID = %s
        """,
        'copper_query': """
            SELECT * FROM pr01ccrdo.trd_guid 
            WHERE GFID_ID=%s and GPID_ID='CAIBE' and GFID_GUID='VGARC'
        """,
        'recommended_actions': ['Run MASS SYNC job', 'Run Sync FLAG to N Job']
    },
    'scenario_2_2': {
        'name': 'Expired GUID in OSCAR and Active CoPPER',
        'description': 'OSCAR expired GUID present in CoPPER but GUS and GFID combination present with Different GUID and active',
        'oscar_query': """
            SELECT distinct EXTRACTVALUE(XML,'//expDate') as exp_date,
            EXTRACTVALUE(XML,'//idNumber') as id_number, 
            EXTRACTVALUE(XML,'//globexFirmID') as globex_firm_id,
            EXTRACTVALUE(XML,'//status') as status
            FROM PR01COSRO.ACTIVE_XML_DATA_STORE WHERE GUID = %s
        """,
        'copper_query': """
            SELECT * FROM pr01ccrdo.trd_guid 
            WHERE GFID_ID=%s and GPID_ID='CAIBE' and GFID_GUID='JRCOS'
        """,
        'recommended_actions': ['Run Sync FLAG to N Job']
    },
    'scenario_2_3': {
        'name': 'Expired GUID in OSCAR and GUID missing CoPPER',
        'description': 'Expired GUID in OSCAR and GUID is missing in CoPPER',
        'oscar_query': """
            SELECT distinct EXTRACTVALUE(XML,'//expDate') as exp_date,
            EXTRACTVALUE(XML,'//idNumber') as id_number, 
            EXTRACTVALUE(XML,'//globexFirmID') as globex_firm_id,
            EXTRACTVALUE(XML,'//status') as status
            FROM PR01COSRO.ACTIVE_XML_DATA_STORE WHERE GUID = %s
        """,
        'copper_query': """
            SELECT * FROM pr01ccrdo.trd_guid WHERE GFID_ID=%s
        """,
        'recommended_actions': ['Run Sync FLAG to N Job']
    },
    'scenario_2_4': {
        'name': 'Active GUID exists in OSCAR and same GUID missing CoPPER',
        'description': 'If OSCAR Active GUID is not present then we need to have CoPPER to check why it is missing then take appropriate action',
        'oscar_query': """
            SELECT distinct EXTRACTVALUE(XML,'//expDate') as exp_date,
            EXTRACTVALUE(XML,'//idNumber') as id_number, 
            EXTRACTVALUE(XML,'//globexFirmID') as globex_firm_id,
            EXTRACTVALUE(XML,'//status') as status
            FROM PR01COSRO.ACTIVE_XML_DATA_STORE WHERE GUID = %s
        """,
        'copper_query': """
            SELECT * FROM pr01ccrdo.trd_guid WHERE GFID_ID=%s
        """,
        'recommended_actions': ['Investigate missing CoPPER data', 'Manual sync required']
    }
}

# GFID Types Configuration
GFID_TYPES = {
    'BTEC_EU': {
        'name': 'BTEC EU GFID',
        'length': 4,
        'validation_pattern': r'^[A-Z0-9]{4}$'
    },
    'BTEC_US': {
        'name': 'BTEC US GFID', 
        'length': 4,
        'validation_pattern': r'^[A-Z0-9]{4}$'
    },
    'EBS': {
        'name': 'EBS GFID',
        'length': 4,
        'validation_pattern': r'^[A-Z0-9]{4}$'
    },
    'CME_FO': {
        'name': 'CME F&O GFID',
        'length': 4,
        'validation_pattern': r'^[A-Z0-9]{4}$'
    }
}

# GUS Types Configuration
GUS_TYPES = {
    'BTEC_US': {
        'name': 'BTEC US GUS',
        'length': 5,
        'validation_pattern': r'^[A-Z0-9]{5}$'
    },
    'BTEC_EU': {
        'name': 'BTEC EU GUS',
        'length': 5,
        'validation_pattern': r'^[A-Z0-9]{5}$'
    },
    'EBS': {
        'name': 'EBS GUS',
        'length': 5,
        'validation_pattern': r'^[A-Z0-9]{5}$'
    }
}

# Session Types Configuration
SESSION_TYPES = {
    'MD': {
        'name': 'Market Data Session',
        'description': 'Market Data session for trading data'
    },
    'OE': {
        'name': 'Order Entry Session',
        'description': 'Order Entry session for order management'
    }
}

# Action Types Configuration
ACTION_TYPES = {
    'MASS_SYNC': {
        'name': 'Mass Sync Job',
        'description': 'Run mass synchronization between OSCAR and CoPPER',
        'estimated_time': '5-10 minutes'
    },
    'SYNC_FLAG_N': {
        'name': 'Sync FLAG to N Job',
        'description': 'Run sync flag to N job for specific GUID',
        'estimated_time': '2-5 minutes'
    },
    'MANUAL_SYNC': {
        'name': 'Manual Sync',
        'description': 'Manual synchronization process',
        'estimated_time': '10-30 minutes'
    },
    'INVESTIGATE': {
        'name': 'Investigate',
        'description': 'Manual investigation required',
        'estimated_time': 'Variable'
    }
}














<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSCAR Reconciliation Tool</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='animations.css') }}">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <!-- Loading Overlay -->
    <div id="loading-overlay" class="loading-overlay hidden">
        <div class="loading-spinner">
            <div class="spinner"></div>
            <p>Processing reconciliation...</p>
        </div>
    </div>

    <!-- Header -->
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-exchange-alt"></i>
                    <h1>OSCAR Reconcile</h1>
                </div>
                <div class="header-actions">
                    <button class="btn-icon" id="health-check-btn" title="Check System Health">
                        <i class="fas fa-heartbeat"></i>
                    </button>
                    <div class="connection-status" id="connection-status">
                        <div class="status-dot"></div>
                        <span>Checking...</span>
                    </div>
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

            <!-- Search Form -->
            <section class="search-section">
                <div class="search-card slide-up">
                    <div class="search-header">
                        <h3><i class="fas fa-search"></i> Enter GUID, GUS, or GFID</h3>
                        <p>Input length: GUID (12), GUS (5), GFID (4)</p>
                    </div>
                    
                    <form id="reconcile-form" class="search-form">
                        <div class="input-group">
                            <div class="input-wrapper">
                                <input 
                                    type="text" 
                                    id="input-value" 
                                    name="input_value" 
                                    placeholder="e.g., XYZ (GFID), ABCDE (GUS), or ABCDEFGH1234 (GUID)"
                                    maxlength="12"
                                    required
                                >
                                <div class="input-info" id="input-info">
                                    <span class="input-type">Type: <span id="input-type-display">-</span></span>
                                    <span class="input-length">Length: <span id="input-length-display">0</span></span>
                                </div>
                            </div>
                            
                            <div class="scenario-selector">
                                <label for="scenario-type">Scenario:</label>
                                <select id="scenario-type" name="scenario_type">
                                    <option value="guid_lookup">Standard Lookup</option>
                                    <option value="scenario_2_1">Scenario 2.1 - Both Expired</option>
                                    <option value="scenario_2_2">Scenario 2.2 - OSCAR Expired, CoPPER Active</option>
                                    <option value="scenario_2_3">Scenario 2.3 - OSCAR Expired, CoPPER Missing</option>
                                    <option value="scenario_2_4">Scenario 2.4 - OSCAR Active, CoPPER Missing</option>
                                </select>
                            </div>
                            
                            <button type="submit" class="btn-primary" id="submit-btn">
                                <i class="fas fa-sync-alt"></i>
                                <span>Compare Data</span>
                            </button>
                        </div>
                    </form>
                </div>
            </section>

            <!-- Results Section -->
            <section class="results-section" id="results-section" style="display: none;">
                <div class="results-header">
                    <h3><i class="fas fa-chart-bar"></i> Reconciliation Results</h3>
                    <div class="results-actions">
                        <button class="btn-secondary" id="export-btn">
                            <i class="fas fa-download"></i> Export
                        </button>
                        <button class="btn-secondary" id="clear-results-btn">
                            <i class="fas fa-times"></i> Clear
                        </button>
                    </div>
                </div>

                <!-- Summary Cards -->
                <div class="summary-cards" id="summary-cards">
                    <div class="summary-card oscar-card">
                        <div class="card-header">
                            <h4><i class="fas fa-database"></i> OSCAR</h4>
                            <div class="status-badge" id="oscar-status">-</div>
                        </div>
                        <div class="card-content">
                            <div class="metric">
                                <span class="metric-value" id="oscar-count">0</span>
                                <span class="metric-label">Records Found</span>
                            </div>
                        </div>
                    </div>

                    <div class="summary-card copper-card">
                        <div class="card-header">
                            <h4><i class="fas fa-database"></i> CoPPER</h4>
                            <div class="status-badge" id="copper-status">-</div>
                        </div>
                        <div class="card-content">
                            <div class="metric">
                                <span class="metric-value" id="copper-count">0</span>
                                <span class="metric-label">Records Found</span>
                            </div>
                        </div>
                    </div>

                    <div class="summary-card comparison-card">
                        <div class="card-header">
                            <h4><i class="fas fa-balance-scale"></i> Comparison</h4>
                            <div class="status-badge" id="comparison-status">-</div>
                        </div>
                        <div class="card-content">
                            <div class="metric">
                                <span class="metric-value" id="differences-count">0</span>
                                <span class="metric-label">Differences</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Scenario Information -->
                <div class="scenario-info" id="scenario-info" style="display: none;">
                    <div class="scenario-card">
                        <div class="scenario-header">
                            <h4 id="scenario-title">Scenario Information</h4>
                            <div class="severity-badge" id="scenario-severity">LOW</div>
                        </div>
                        <div class="scenario-content">
                            <p id="scenario-description">Scenario description will appear here.</p>
                            <div class="recommended-actions" id="recommended-actions">
                                <!-- Action buttons will be populated here -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Data Tables -->
                <div class="data-tables">
                    <!-- OSCAR Data Table -->
                    <div class="table-container">
                        <div class="table-header">
                            <h4><i class="fas fa-table"></i> OSCAR Data</h4>
                            <div class="table-info" id="oscar-table-info">No data</div>
                        </div>
                        <div class="table-wrapper">
                            <table class="data-table" id="oscar-table">
                                <thead id="oscar-table-head">
                                    <!-- Headers will be populated dynamically -->
                                </thead>
                                <tbody id="oscar-table-body">
                                    <tr>
                                        <td colspan="100%" class="no-data">No OSCAR data to display</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- CoPPER Data Table -->
                    <div class="table-container">
                        <div class="table-header">
                            <h4><i class="fas fa-table"></i> CoPPER Data</h4>
                            <div class="table-info" id="copper-table-info">No data</div>
                        </div>
                        <div class="table-wrapper">
                            <table class="data-table" id="copper-table">
                                <thead id="copper-table-head">
                                    <!-- Headers will be populated dynamically -->
                                </thead>
                                <tbody id="copper-table-body">
                                    <tr>
                                        <td colspan="100%" class="no-data">No CoPPER data to display</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Differences Table -->
                <div class="differences-container" id="differences-container" style="display: none;">
                    <div class="table-header">
                        <h4><i class="fas fa-exclamation-triangle"></i> Differences Found</h4>
                    </div>
                    <div class="differences-list" id="differences-list">
                        <!-- Differences will be populated here -->
                    </div>
                </div>
            </section>
        </div>
    </main>

    <!-- Action Modal -->
    <div class="modal" id="action-modal" style="display: none;">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modal-title">Execute Action</h3>
                <button class="modal-close" id="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <p id="modal-description">Are you sure you want to execute this action?</p>
                <div class="modal-details" id="modal-details">
                    <!-- Action details will be populated here -->
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn-secondary" id="modal-cancel">Cancel</button>
                <button class="btn-primary" id="modal-confirm">Execute</button>
            </div>
        </div>
    </div>

    <!-- Toast Notifications -->
    <div class="toast-container" id="toast-container"></div>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 OSCAR Reconciliation Tool. Built for financial data synchronization.</p>
        </div>
    </footer>

    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>






















/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    /* Color Palette */
    --primary-color: #1a365d;
    --primary-light: #2c5aa0;
    --primary-dark: #0f2537;
    --secondary-color: #e53e3e;
    --accent-color: #00b4d8;
    --success-color: #38a169;
    --warning-color: #ed8936;
    --error-color: #e53e3e;
    
    /* Neutral Colors */
    --white: #ffffff;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-400: #9ca3af;
    --gray-500: #6b7280;
    --gray-600: #4b5563;
    --gray-700: #374151;
    --gray-800: #1f2937;
    --gray-900: #111827;
    
    /* Gradients */
    --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
    --gradient-success: linear-gradient(135deg, var(--success-color) 0%, #48bb78 100%);
    --gradient-warning: linear-gradient(135deg, var(--warning-color) 0%, #f6ad55 100%);
    --gradient-error: linear-gradient(135deg, var(--error-color) 0%, #fc8181 100%);
    
    /* Typography */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    --font-size-xs: 0.75rem;
    --font-size-sm: 0.875rem;
    --font-size-base: 1rem;
    --font-size-lg: 1.125rem;
    --font-size-xl: 1.25rem;
    --font-size-2xl: 1.5rem;
    --font-size-3xl: 1.875rem;
    --font-size-4xl: 2.25rem;
    
    /* Spacing */
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --spacing-2xl: 3rem;
    --spacing-3xl: 4rem;
    
    /* Border Radius */
    --radius-sm: 0.375rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-xl: 1rem;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    
    /* Transitions */
    --transition-fast: 0.15s ease-in-out;
    --transition-normal: 0.3s ease-in-out;
    --transition-slow: 0.5s ease-in-out;
}

/* Base Typography */
body {
    font-family: var(--font-family);
    font-size: var(--font-size-base);
    line-height: 1.6;
    color: var(--gray-800);
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    min-height: 100vh;
}

/* Layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--spacing-lg);
}

/* Header */
.header {
    background: var(--gradient-primary);
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
    font-size: var(--font-size-2xl);
    color: var(--accent-color);
}

.logo h1 {
    font-size: var(--font-size-2xl);
    font-weight: 700;
    margin: 0;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: var(--spacing-lg);
}

.btn-icon {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: var(--white);
    padding: var(--spacing-sm);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: var(--transition-normal);
    backdrop-filter: blur(10px);
}

.btn-icon:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
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
    background: var(--warning-color);
    animation: pulse 2s infinite;
}

.status-dot.connected {
    background: var(--success-color);
}

.status-dot.disconnected {
    background: var(--error-color);
}

/* Main Content */
.main-content {
    padding: var(--spacing-2xl) 0;
    min-height: calc(100vh - 200px);
}

/* Welcome Section */
.welcome-section {
    text-align: center;
    margin-bottom: var(--spacing-3xl);
}

.welcome-content h2 {
    font-size: var(--font-size-3xl);
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: var(--spacing-md);
}

.welcome-content p {
    font-size: var(--font-size-lg);
    color: var(--gray-600);
    max-width: 600px;
    margin: 0 auto;
}

/* Search Section */
.search-section {
    margin-bottom: var(--spacing-3xl);
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
    font-size: var(--font-size-xl);
    font-weight: 600;
    color: var(--primary-color);
    margin-bottom: var(--spacing-sm);
}

.search-header h3 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.search-header p {
    color: var(--gray-600);
    font-size: var(--font-size-sm);
}

.search-form {
    max-width: 800px;
    margin: 0 auto;
}

.input-group {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: var(--spacing-lg);
    align-items: end;
}

.input-wrapper {
    position: relative;
}

.input-wrapper input {
    width: 100%;
    padding: var(--spacing-md) var(--spacing-lg);
    border: 2px solid var(--gray-300);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-base);
    transition: var(--transition-normal);
    background: var(--white);
}

.input-wrapper input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
}

.input-info {
    display: flex;
    justify-content: space-between;
    margin-top: var(--spacing-sm);
    font-size: var(--font-size-xs);
    color: var(--gray-500);
}

.scenario-selector {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.scenario-selector label {
    font-size: var(--font-size-sm);
    font-weight: 500;
    color: var(--gray-700);
}

.scenario-selector select {
    padding: var(--spacing-md);
    border: 2px solid var(--gray-300);
    border-radius: var(--radius-lg);
    background: var(--white);
    font-size: var(--font-size-sm);
    min-width: 200px;
}

/* Buttons */
.btn-primary {
    background: var(--gradient-primary);
    color: var(--white);
    border: none;
    padding: var(--spacing-md) var(--spacing-xl);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-base);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    box-shadow: var(--shadow-md);
    white-space: nowrap;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.btn-primary:active {
    transform: translateY(0);
}

.btn-secondary {
    background: var(--white);
    color: var(--gray-700);
    border: 2px solid var(--gray-300);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-sm);
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

/* Results Section */
.results-section {
    margin-bottom: var(--spacing-3xl);
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-xl);
}

.results-header h3 {
    font-size: var(--font-size-xl);
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

/* Summary Cards */
.summary-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
}

.summary-card {
    background: var(--white);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    box-shadow: var(--shadow-lg);
    border-left: 4px solid;
    transition: var(--transition-normal);
}

.summary-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-xl);
}

.oscar-card {
    border-left-color: var(--primary-color);
}

.copper-card {
    border-left-color: var(--accent-color);
}

.comparison-card {
    border-left-color: var(--warning-color);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-lg);
}

.card-header h4 {
    font-size: var(--font-size-lg);
    font-weight: 600;
    color: var(--gray-800);
}

.card-header h4 i {
    margin-right: var(--spacing-sm);
}

.status-badge {
    padding: var(--spacing-xs) var(--spacing-md);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-xs);
    font-weight: 600;
    text-transform: uppercase;
}

.status-badge.found {
    background: var(--success-color);
    color: var(--white);
}

.status-badge.not-found {
    background: var(--error-color);
    color: var(--white);
}

.status-badge.error {
    background: var(--warning-color);
    color: var(--white);
}

.metric {
    text-align: center;
}

.metric-value {
    display: block;
    font-size: var(--font-size-3xl);
    font-weight: 700;
    color: var(--primary-color);
}

.metric-label {
    font-size: var(--font-size-sm);
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Scenario Info */
.scenario-info {
    margin-bottom: var(--spacing-xl);
}

.scenario-card {
    background: var(--white);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--gray-200);
}

.scenario-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-lg);
}

.scenario-header h4 {
    font-size: var(--font-size-lg);
    font-weight: 600;
    color: var(--primary-color);
}

.severity-badge {
    padding: var(--spacing-xs) var(--spacing-md);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-xs);
    font-weight: 600;
    text-transform: uppercase;
}

.severity-badge.low {
    background: var(--success-color);
    color: var(--white);
}

.severity-badge.medium {
    background: var(--warning-color);
    color: var(--white);
}

.severity-badge.high {
    background: var(--error-color);
    color: var(--white);
}

.scenario-content p {
    margin-bottom: var(--spacing-lg);
    color: var(--gray-700);
}

.recommended-actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-md);
}

.action-btn {
    background: var(--gradient-primary);
    color: var(--white);
    border: none;
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.action-btn.danger {
    background: var(--gradient-error);
}

.action-btn.warning {
    background: var(--gradient-warning);
}

/* Data Tables */
.data-tables {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--spacing-xl);
    margin-bottom: var(--spacing-xl);
}

.table-container {
    background: var(--white);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
}

.table-header {
    padding: var(--spacing-lg);
    background: var(--gray-50);
    border-bottom: 1px solid var(--gray-200);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.table-header h4 {
    font-size: var(--font-size-lg);
    font-weight: 600;
    color: var(--gray-800);
}

.table-header h4 i {
    margin-right: var(--spacing-sm);
}

.table-info {
    font-size: var(--font-size-sm);
    color: var(--gray-600);
}

.table-wrapper {
    max-height: 400px;
    overflow: auto;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
}

.data-table th {
    background: var(--gray-100);
    padding: var(--spacing-md);
    text-align: left;
    font-weight: 600;
    color: var(--gray-700);
    border-bottom: 1px solid var(--gray-200);
    position: sticky;
    top: 0;
    z-index: 10;
}

.data-table td {
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--gray-100);
    vertical-align: top;
}

.data-table tbody tr:hover {
    background: var(--gray-50);
}

.no-data {
    text-align: center;
    color: var(--gray-500);
    font-style: italic;
    padding: var(--spacing-xl) !important;
}

/* Differences Container */
.differences-container {
    background: var(--white);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    padding: var(--spacing-xl);
    border-left: 4px solid var(--error-color);
}

.differences-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.difference-item {
    padding: var(--spacing-md);
    background: var(--gray-50);
    border-radius: var(--radius-lg);
    border-left: 3px solid var(--error-color);
}

/* Modal */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
    backdrop-filter: blur(5px);
}

.modal-content {
    background: var(--white);
    border-radius: var(--radius-xl);
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow: auto;
    box-shadow: var(--shadow-xl);
}

.modal-header {
    padding: var(--spacing-xl);
    border-bottom: 1px solid var(--gray-200);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h3 {
    font-size: var(--font-size-xl);
    font-weight: 600;
    color: var(--primary-color);
}

.modal-close {
    background: none;
    border: none;
    font-size: var(--font-size-xl);
    cursor: pointer;
    color: var(--gray-500);
    padding: var(--spacing-xs);
}

.modal-close:hover {
    color: var(--gray-800);
}

.modal-body {
    padding: var(--spacing-xl);
}

.modal-footer {
    padding: var(--spacing-xl);
    border-top: 1px solid var(--gray-200);
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-md);
}

/* Loading Overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.95);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 2000;
    backdrop-filter: blur(10px);
}

.loading-spinner {
    text-align: center;
}

.spinner {
    width: 50px;
    height: 50px;
    border: 4px solid var(--gray-300);
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto var(--spacing-lg);
}

.loading-spinner p {
    color: var(--gray-700);
    font-weight: 500;
}

/* Toast Notifications */
.toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 3000;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.toast {
    background: var(--white);
    padding: var(--spacing-lg);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-xl);
    border-left: 4px solid;
    max-width: 400px;
    animation: slideInRight 0.3s ease-out;
}

.toast.success {
    border-left-color: var(--success-color);
}

.toast.error {
    border-left-color: var(--error-color);
}

.toast.warning {
    border-left-color: var(--warning-color);
}

.toast.info {
    border-left-color: var(--accent-color);
}

/* Footer */
.footer {
    background: var(--gray-800);
    color: var(--white);
    text-align: center;
    padding: var(--spacing-xl) 0;
    margin-top: auto;
}

/* Utilities */
.hidden {
    display: none !important;
}

.text-center {
    text-align: center;
}

.text-success {
    color: var(--success-color);
}

.text-error {
    color: var(--error-color);
}

.text-warning {
    color: var(--warning-color);
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        padding: 0 var(--spacing-md);
    }
    
    .header-content {
        flex-direction: column;
        gap: var(--spacing-md);
    }
    
    .input-group {
        grid-template-columns: 1fr;
        gap: var(--spacing-md);
    }
    
    .summary-cards {
        grid-template-columns: 1fr;
    }
    
    .data-tables {
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
    
    .scenario-header {
        flex-direction: column;
        gap: var(--spacing-md);
        align-items: stretch;
    }
    
    .recommended-actions {
        justify-content: center;
    }
}

@media (max-width: 480px) {
    .welcome-content h2 {
        font-size: var(--font-size-2xl);
    }
    
    .search-card {
        padding: var(--spacing-lg);
    }
    
    .modal-content {
        width: 95%;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    :root {
        --white: #1a1a1a;
        --gray-50: #262626;
        --gray-100: #404040;
        --gray-200: #525252;
        --gray-300: #737373;
        --gray-400: #a3a3a3;
        --gray-500: #d4d4d4;
        --gray-600: #e5e5e5;
        --gray-700: #f5f5f5;
        --gray-800: #fafafa;
        --gray-900: #ffffff;
    }
    
    body {
        background: linear-gradient(135deg, #1a1a1a 0%, #262626 100%);
    }
}


















/* Animation Keyframes */
@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
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

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes fadeInRight {
    from {
        opacity: 0;
        transform: translateX(30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInUp {
    from {
        transform: translateY(100%);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes slideInDown {
    from {
        transform: translateY(-100%);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes slideInLeft {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
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

@keyframes slideOutUp {
    from {
        transform: translateY(0);
        opacity: 1;
    }
    to {
        transform: translateY(-100%);
        opacity: 0;
    }
}

@keyframes slideOutDown {
    from {
        transform: translateY(0);
        opacity: 1;
    }
    to {
        transform: translateY(100%);
        opacity: 0;
    }
}

@keyframes slideOutLeft {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(-100%);
        opacity: 0;
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

@keyframes scaleIn {
    from {
        transform: scale(0.8);
        opacity: 0;
    }
    to {
        transform: scale(1);
        opacity: 1;
    }
}

@keyframes scaleOut {
    from {
        transform: scale(1);
        opacity: 1;
    }
    to {
        transform: scale(0.8);
        opacity: 0;
    }
}

@keyframes bounce {
    0%, 20%, 53%, 80%, 100% {
        transform: translate3d(0, 0, 0);
    }
    40%, 43% {
        transform: translate3d(0, -15px, 0);
    }
    70% {
        transform: translate3d(0, -7px, 0);
    }
    90% {
        transform: translate3d(0, -2px, 0);
    }
}

@keyframes pulse {
    0% {
        transform: scale(1);
        opacity: 1;
    }
    50% {
        transform: scale(1.1);
        opacity: 0.7;
    }
    100% {
        transform: scale(1);
        opacity: 1;
    }
}

@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

@keyframes wiggle {
    0%, 7% {
        transform: rotateZ(0);
    }
    15% {
        transform: rotateZ(-15deg);
    }
    20% {
        transform: rotateZ(10deg);
    }
    25% {
        transform: rotateZ(-10deg);
    }
    30% {
        transform: rotateZ(6deg);
    }
    35% {
        transform: rotateZ(-4deg);
    }
    40%, 100% {
        transform: rotateZ(0);
    }
}

@keyframes swing {
    20% {
        transform: rotate(15deg);
    }
    40% {
        transform: rotate(-10deg);
    }
    60% {
        transform: rotate(5deg);
    }
    80% {
        transform: rotate(-5deg);
    }
    100% {
        transform: rotate(0deg);
    }
}

@keyframes heartbeat {
    0% {
        transform: scale(1);
    }
    14% {
        transform: scale(1.3);
    }
    28% {
        transform: scale(1);
    }
    42% {
        transform: scale(1.3);
    }
    70% {
        transform: scale(1);
    }
}

@keyframes flash {
    0%, 50%, 100% {
        opacity: 1;
    }
    25%, 75% {
        opacity: 0;
    }
}

@keyframes rubberBand {
    0% {
        transform: scale(1);
    }
    30% {
        transform: scaleX(1.25) scaleY(0.75);
    }
    40% {
        transform: scaleX(0.75) scaleY(1.25);
    }
    50% {
        transform: scaleX(1.15) scaleY(0.85);
    }
    65% {
        transform: scaleX(0.95) scaleY(1.05);
    }
    75% {
        transform: scaleX(1.05) scaleY(0.95);
    }
    100% {
        transform: scale(1);
    }
}

@keyframes zoomIn {
    from {
        opacity: 0;
        transform: scale3d(0.3, 0.3, 0.3);
    }
    50% {
        opacity: 1;
    }
    to {
        opacity: 1;
        transform: scale3d(1, 1, 1);
    }
}

@keyframes zoomOut {
    from {
        opacity: 1;
        transform: scale3d(1, 1, 1);
    }
    50% {
        opacity: 0;
        transform: scale3d(0.3, 0.3, 0.3);
    }
    to {
        opacity: 0;
        transform: scale3d(0.3, 0.3, 0.3);
    }
}

@keyframes rotateIn {
    from {
        transform-origin: center;
        transform: rotate(-200deg);
        opacity: 0;
    }
    to {
        transform-origin: center;
        transform: rotate(0);
        opacity: 1;
    }
}

@keyframes rotateOut {
    from {
        transform-origin: center;
        opacity: 1;
    }
    to {
        transform-origin: center;
        transform: rotate(200deg);
        opacity: 0;
    }
}

@keyframes flipInX {
    from {
        transform: perspective(400px) rotate3d(1, 0, 0, 90deg);
        animation-timing-function: ease-in;
        opacity: 0;
    }
    40% {
        transform: perspective(400px) rotate3d(1, 0, 0, -20deg);
        animation-timing-function: ease-in;
    }
    60% {
        transform: perspective(400px) rotate3d(1, 0, 0, 10deg);
        opacity: 1;
    }
    80% {
        transform: perspective(400px) rotate3d(1, 0, 0, -5deg);
    }
    to {
        transform: perspective(400px);
    }
}

@keyframes flipInY {
    from {
        transform: perspective(400px) rotate3d(0, 1, 0, 90deg);
        animation-timing-function: ease-in;
        opacity: 0;
    }
    40% {
        transform: perspective(400px) rotate3d(0, 1, 0, -20deg);
        animation-timing-function: ease-in;
    }
    60% {
        transform: perspective(400px) rotate3d(0, 1, 0, 10deg);
        opacity: 1;
    }
    80% {
        transform: perspective(400px) rotate3d(0, 1, 0, -5deg);
    }
    to {
        transform: perspective(400px);
    }
}

@keyframes lightSpeedInRight {
    from {
        transform: translate3d(100%, 0, 0) skewX(-30deg);
        opacity: 0;
    }
    60% {
        transform: skewX(20deg);
        opacity: 1;
    }
    80% {
        transform: skewX(-5deg);
    }
    to {
        transform: translate3d(0, 0, 0);
    }
}

@keyframes lightSpeedInLeft {
    from {
        transform: translate3d(-100%, 0, 0) skewX(30deg);
        opacity: 0;
    }
    60% {
        transform: skewX(-20deg);
        opacity: 1;
    }
    80% {
        transform: skewX(5deg);
    }
    to {
        transform: translate3d(0, 0, 0);
    }
}

@keyframes rollIn {
    from {
        opacity: 0;
        transform: translate3d(-100%, 0, 0) rotate3d(0, 0, 1, -120deg);
    }
    to {
        opacity: 1;
        transform: translate3d(0, 0, 0);
    }
}

@keyframes rollOut {
    from {
        opacity: 1;
    }
    to {
        opacity: 0;
        transform: translate3d(100%, 0, 0) rotate3d(0, 0, 1, 120deg);
    }
}

@keyframes hinge {
    0% {
        transform-origin: top left;
        animation-timing-function: ease-in-out;
    }
    20%, 60% {
        transform: rotate(80deg);
        transform-origin: top left;
        animation-timing-function: ease-in-out;
    }
    40%, 80% {
        transform: rotate(60deg);
        transform-origin: top left;
        animation-timing-function: ease-in-out;
        opacity: 1;
    }
    to {
        transform: translate3d(0, 700px, 0);
        opacity: 0;
    }
}

@keyframes jackInTheBox {
    from {
        opacity: 0;
        transform: scale(0.1) rotate(30deg);
        transform-origin: center bottom;
    }
    50% {
        transform: rotate(-10deg);
    }
    70% {
        transform: rotate(3deg);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* Animation Utility Classes */
.animate-fade-in {
    animation: fadeIn 0.5s ease-out;
}

.animate-fade-in-up {
    animation: fadeInUp 0.6s ease-out;
}

.animate-fade-in-down {
    animation: fadeInDown 0.6s ease-out;
}

.animate-fade-in-left {
    animation: fadeInLeft 0.6s ease-out;
}

.animate-fade-in-right {
    animation: fadeInRight 0.6s ease-out;
}

.animate-slide-in-up {
    animation: slideInUp 0.5s ease-out;
}

.animate-slide-in-down {
    animation: slideInDown 0.5s ease-out;
}

.animate-slide-in-left {
    animation: slideInLeft 0.5s ease-out;
}

.animate-slide-in-right {
    animation: slideInRight 0.5s ease-out;
}

.animate-scale-in {
    animation: scaleIn 0.4s ease-out;
}

.animate-bounce {
    animation: bounce 1s;
}

.animate-pulse {
    animation: pulse 2s infinite;
}

.animate-spin {
    animation: spin 1s linear infinite;
}

.animate-wiggle {
    animation: wiggle 1s ease-in-out;
}

.animate-swing {
    animation: swing 1s ease-in-out;
}

.animate-heartbeat {
    animation: heartbeat 1.5s ease-in-out infinite;
}

.animate-flash {
    animation: flash 1s infinite;
}

.animate-rubber-band {
    animation: rubberBand 1s;
}

.animate-zoom-in {
    animation: zoomIn 0.5s;
}

.animate-rotate-in {
    animation: rotateIn 0.6s;
}

.animate-flip-in-x {
    animation: flipInX 0.75s;
}

.animate-flip-in-y {
    animation: flipInY 0.75s;
}

.animate-light-speed-in-right {
    animation: lightSpeedInRight 1s ease-out;
}

.animate-light-speed-in-left {
    animation: lightSpeedInLeft 1s ease-out;
}

.animate-roll-in {
    animation: rollIn 1s;
}

.animate-jack-in-the-box {
    animation: jackInTheBox 1s;
}

/* Page Load Animations */
.page-enter {
    opacity: 0;
    transform: translateY(20px);
}

.page-enter-active {
    opacity: 1;
    transform: translateY(0);
    transition: opacity 0.3s ease-out, transform 0.3s ease-out;
}

.page-exit {
    opacity: 1;
    transform: translateY(0);
}

.page-exit-active {
    opacity: 0;
    transform: translateY(-20px);
    transition: opacity 0.3s ease-out, transform 0.3s ease-out;
}

/* Element Animation Classes with Delays */
.fade-in {
    opacity: 0;
    animation: fadeIn 0.8s ease-out forwards;
}

.fade-in-delay-1 {
    animation-delay: 0.1s;
}

.fade-in-delay-2 {
    animation-delay: 0.2s;
}

.fade-in-delay-3 {
    animation-delay: 0.3s;
}

.fade-in-delay-4 {
    animation-delay: 0.4s;
}

.fade-in-delay-5 {
    animation-delay: 0.5s;
}

.slide-up {
    opacity: 0;
    transform: translateY(30px);
    animation: fadeInUp 0.8s ease-out forwards;
}

.slide-up-delay-1 {
    animation-delay: 0.1s;
}

.slide-up-delay-2 {
    animation-delay: 0.2s;
}

.slide-up-delay-3 {
    animation-delay: 0.3s;
}

.slide-up-delay-4 {
    animation-delay: 0.4s;
}

.slide-up-delay-5 {
    animation-delay: 0.5s;
}

/* Hover Animations */
.hover-scale {
    transition: transform 0.3s ease;
}

.hover-scale:hover {
    transform: scale(1.05);
}

.hover-lift {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.hover-lift:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.hover-glow {
    transition: box-shadow 0.3s ease;
}

.hover-glow:hover {
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
}

.hover-rotate {
    transition: transform 0.3s ease;
}

.hover-rotate:hover {
    transform: rotate(5deg);
}

.hover-shake {
    transition: transform 0.3s ease;
}

.hover-shake:hover {
    animation: wiggle 0.5s ease-in-out;
}

/* Loading Animations */
.loading-dots::after {
    content: '';
    animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
    0%, 20% {
        color: rgba(0, 0, 0, 0);
        text-shadow: 0.25em 0 0 rgba(0, 0, 0, 0),
                     0.5em 0 0 rgba(0, 0, 0, 0);
    }
    40% {
        color: currentColor;
        text-shadow: 0.25em 0 0 rgba(0, 0, 0, 0),
                     0.5em 0 0 rgba(0, 0, 0, 0);
    }
    60% {
        text-shadow: 0.25em 0 0 currentColor,
                     0.5em 0 0 rgba(0, 0, 0, 0);
    }
    80%, 100% {
        text-shadow: 0.25em 0 0 currentColor,
                     0.5em 0 0 currentColor;
    }
}

.loading-bars {
    display: inline-block;
    position: relative;
    width: 40px;
    height: 20px;
}

.loading-bars div {
    display: inline-block;
    position: absolute;
    left: 4px;
    width: 6px;
    background: currentColor;
    animation: loading-bars 1.2s cubic-bezier(0, 0.5, 0.5, 1) infinite;
}

.loading-bars div:nth-child(1) {
    left: 4px;
    animation-delay: -0.24s;
}

.loading-bars div:nth-child(2) {
    left: 14px;
    animation-delay: -0.12s;
}

.loading-bars div:nth-child(3) {
    left: 24px;
    animation-delay: 0;
}

@keyframes loading-bars {
    0% {
        top: 4px;
        height: 12px;
    }
    50%, 100% {
        top: 12px;
        height: 4px;
    }
}

/* Success/Error Animation States */
.success-animation {
    animation: scaleIn 0.3s ease-out, pulse 0.6s ease-out 0.3s;
    background: linear-gradient(135deg, #38a169 0%, #48bb78 100%);
    color: white;
}

.error-animation {
    animation: wiggle 0.5s ease-in-out;
    background: linear-gradient(135deg, #e53e3e 0%, #fc8181 100%);
    color: white;
}

.warning-animation {
    animation: swing 0.8s ease-in-out;
    background: linear-gradient(135deg, #ed8936 0%, #f6ad55 100%);
    color: white;
}

/* Card Stack Animation */
.card-stack {
    position: relative;
}

.card-stack .card {
    transition: transform 0.3s ease;
}

.card-stack:hover .card:nth-child(1) {
    transform: translateY(-8px) rotate(-2deg);
}

.card-stack:hover .card:nth-child(2) {
    transform: translateY(-4px) rotate(1deg);
}

.card-stack:hover .card:nth-child(3) {
    transform: translateY(0) rotate(-0.5deg);
}

/* Stagger Animation for Lists */
.stagger-animation .item {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 0.6s ease-out forwards;
}

.stagger-animation .item:nth-child(1) { animation-delay: 0.1s; }
.stagger-animation .item:nth-child(2) { animation-delay: 0.2s; }
.stagger-animation .item:nth-child(3) { animation-delay: 0.3s; }
.stagger-animation .item:nth-child(4) { animation-delay: 0.4s; }
.stagger-animation .item:nth-child(5) { animation-delay: 0.5s; }
.stagger-animation .item:nth-child(6) { animation-delay: 0.6s; }
.stagger-animation .item:nth-child(7) { animation-delay: 0.7s; }
.stagger-animation .item:nth-child(8) { animation-delay: 0.8s; }
.stagger-animation .item:nth-child(9) { animation-delay: 0.9s; }
.stagger-animation .item:nth-child(10) { animation-delay: 1.0s; }

/* Typewriter Animation */
.typewriter {
    overflow: hidden;
    border-right: 0.15em solid currentColor;
    white-space: nowrap;
    margin: 0 auto;
    animation: typing 3.5s steps(40, end), blink-caret 0.75s step-end infinite;
}

@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blink-caret {
    from, to { border-color: transparent; }
    50% { border-color: currentColor; }
}

/* Progress Bar Animation */
.progress-bar {
    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
    position: relative;
}

.progress-bar::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent);
    animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Morphing Button Animation */
.morphing-button {
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.morphing-button.loading {
    width: 40px;
    padding: 0;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

.morphing-button.success {
    background: #38a169;
    animation: scaleIn 0.3s ease-out;
}

.morphing-button.error {
    background: #e53e3e;
    animation: wiggle 0.5s ease-in-out;
}

/* Particle Effect */
.particle {
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
    animation: particle-float 3s linear infinite;
}

@keyframes particle-float {
    0% {
        opacity: 0;
        transform: translateY(0) scale(0);
    }
    10% {
        opacity: 1;
        transform: translateY(-10px) scale(1);
    }
    90% {
        opacity: 1;
        transform: translateY(-90px) scale(1);
    }
    100% {
        opacity: 0;
        transform: translateY(-100px) scale(0);
    }
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
    
    .animate-spin {
        animation: none;
    }
    
    .animate-pulse {
        animation: none;
    }
    
    .animate-bounce {
        animation: none;
    }
}

















// OSCAR Reconciliation Tool - Frontend JavaScript

class OSCARReconcile {
    constructor() {
        this.currentData = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeAnimations();
        this.checkSystemHealth();
        this.updateInputInfo();
    }

    bindEvents() {
        // Form submission
        const form = document.getElementById('reconcile-form');
        if (form) {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        }

        // Input value changes
        const inputValue = document.getElementById('input-value');
        if (inputValue) {
            inputValue.addEventListener('input', this.updateInputInfo.bind(this));
            inputValue.addEventListener('keyup', this.handleKeyUp.bind(this));
        }

        // Health check button
        const healthBtn = document.getElementById('health-check-btn');
        if (healthBtn) {
            healthBtn.addEventListener('click', this.checkSystemHealth.bind(this));
        }

        // Export button
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', this.exportResults.bind(this));
        }

        // Clear results button
        const clearBtn = document.getElementById('clear-results-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', this.clearResults.bind(this));
        }

        // Modal events
        this.bindModalEvents();

        // Window events
        window.addEventListener('resize', this.handleResize.bind(this));
    }

    bindModalEvents() {
        const modal = document.getElementById('action-modal');
        const modalClose = document.getElementById('modal-close');
        const modalCancel = document.getElementById('modal-cancel');
        const modalConfirm = document.getElementById('modal-confirm');

        if (modalClose) {
            modalClose.addEventListener('click', this.hideModal.bind(this));
        }

        if (modalCancel) {
            modalCancel.addEventListener('click', this.hideModal.bind(this));
        }

        if (modalConfirm) {
            modalConfirm.addEventListener('click', this.confirmAction.bind(this));
        }

        // Close modal when clicking outside
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideModal();
                }
            });
        }

        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideModal();
            }
        });
    }

    initializeAnimations() {
        // Add entrance animations to elements
        const elements = document.querySelectorAll('.fade-in, .slide-up');
        elements.forEach((el, index) => {
            el.style.animationDelay = `${index * 0.1}s`;
        });

        // Intersection Observer for scroll animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in-up');
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });
    }

    updateInputInfo() {
        const input = document.getElementById('input-value');
        const typeDisplay = document.getElementById('input-type-display');
        const lengthDisplay = document.getElementById('input-length-display');

        if (!input || !typeDisplay || !lengthDisplay) return;

        const value = input.value.trim();
        const length = value.length;

        lengthDisplay.textContent = length;

        let type = '-';
        let isValid = false;

        if (length === 12) {
            type = 'GUID';
            isValid = /^[A-Z0-9]{12}$/i.test(value);
        } else if (length === 5) {
            type = 'GUS';
            isValid = /^[A-Z0-9]{5}$/i.test(value);
        } else if (length === 4) {
            type = 'GFID';
            isValid = /^[A-Z0-9]{4}$/i.test(value);
        }

        typeDisplay.textContent = type;

        // Update input styling based on validity
        input.classList.toggle('valid', isValid && length > 0);
        input.classList.toggle('invalid', !isValid && length > 0);
    }

    handleKeyUp(e) {
        // Enter key to submit
        if (e.key === 'Enter') {
            const form = document.getElementById('reconcile-form');
            if (form) {
                form.dispatchEvent(new Event('submit'));
            }
        }
    }

    async handleFormSubmit(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const inputValue = formData.get('input_value').trim();
        const scenarioType = formData.get('scenario_type');

        if (!inputValue) {
            this.showToast('Please enter a GUID, GUS, or GFID', 'error');
            return;
        }

        // Validate input format
        if (!this.validateInput(inputValue)) {
            this.showToast('Invalid input format. Please check the length and format requirements.', 'error');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/reconcile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input_value: inputValue,
                    scenario_type: scenarioType
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.currentData = data;
                this.displayResults(data);
                this.showToast('Reconciliation completed successfully', 'success');
            } else {
                throw new Error(data.error || 'Reconciliation failed');
            }

        } catch (error) {
            console.error('Reconciliation error:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    validateInput(value) {
        const length = value.length;
        
        if (length === 12) {
            return /^[A-Z0-9]{12}$/i.test(value);
        } else if (length === 5) {
            return /^[A-Z0-9]{5}$/i.test(value);
        } else if (length === 4) {
            return /^[A-Z0-9]{4}$/i.test(value);
        }
        
        return false;
    }

    displayResults(data) {
        // Show results section
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }

        // Update summary cards
        this.updateSummaryCards(data);

        // Update scenario information
        this.updateScenarioInfo(data.scenario);

        // Update data tables
        this.updateDataTables(data);

        // Update differences
        this.updateDifferences(data.comparison);

        // Add stagger animation to results
        this.animateResults();
    }

    updateSummaryCards(data) {
        // OSCAR card
        const oscarStatus = document.getElementById('oscar-status');
        const oscarCount = document.getElementById('oscar-count');
        
        if (oscarStatus && oscarCount) {
            oscarStatus.textContent = data.oscar_data.status;
            oscarStatus.className = `status-badge ${data.oscar_data.found ? 'found' : 'not-found'}`;
            oscarCount.textContent = data.oscar_data.count;
        }

        // CoPPER card
        const copperStatus = document.getElementById('copper-status');
        const copperCount = document.getElementById('copper-count');
        
        if (copperStatus && copperCount) {
            copperStatus.textContent = data.copper_data.status;
            copperStatus.className = `status-badge ${data.copper_data.found ? 'found' : 'not-found'}`;
            copperCount.textContent = data.copper_data.count;
        }

        // Comparison card
        const comparisonStatus = document.getElementById('comparison-status');
        const differencesCount = document.getElementById('differences-count');
        
        if (comparisonStatus && differencesCount) {
            comparisonStatus.textContent = data.comparison.match_status;
            comparisonStatus.className = `status-badge ${this.getComparisonStatusClass(data.comparison.match_status)}`;
            differencesCount.textContent = data.comparison.differences.length;
        }
    }

    getComparisonStatusClass(status) {
        switch (status) {
            case 'MATCH':
            case 'SYNC':
                return 'found';
            case 'STATUS_MISMATCH':
            case 'COPPER_MISSING':
            case 'OSCAR_MISSING':
                return 'error';
            default:
                return 'not-found';
        }
    }

    updateScenarioInfo(scenario) {
        const scenarioInfo = document.getElementById('scenario-info');
        const scenarioTitle = document.getElementById('scenario-title');
        const scenarioSeverity = document.getElementById('scenario-severity');
        const scenarioDescription = document.getElementById('scenario-description');
        const recommendedActions = document.getElementById('recommended-actions');

        if (!scenarioInfo || !scenario) return;

        scenarioInfo.style.display = 'block';
        
        if (scenarioTitle) {
            scenarioTitle.textContent = scenario.type || 'Unknown Scenario';
        }

        if (scenarioSeverity) {
            scenarioSeverity.textContent = scenario.severity || 'LOW';
            scenarioSeverity.className = `severity-badge ${(scenario.severity || 'low').toLowerCase()}`;
        }

        if (scenarioDescription) {
            scenarioDescription.textContent = scenario.description || 'No description available';
        }

        if (recommendedActions) {
            recommendedActions.innerHTML = '';
            
            if (scenario.recommended_actions && scenario.recommended_actions.length > 0) {
                scenario.recommended_actions.forEach(action => {
                    const button = document.createElement('button');
                    button.className = `action-btn ${this.getActionButtonClass(action)}`;
                    button.innerHTML = `<i class="fas fa-play"></i> ${action}`;
                    button.addEventListener('click', () => this.showActionModal(action));
                    recommendedActions.appendChild(button);
                });
            }
        }
    }

    getActionButtonClass(action) {
        if (action.toLowerCase().includes('mass sync')) {
            return 'warning';
        } else if (action.toLowerCase().includes('investigate')) {
            return 'danger';
        }
        return '';
    }

    updateDataTables(data) {
        // Update OSCAR table
        this.updateTable('oscar', data.oscar_data.data);
        
        // Update CoPPER table
        this.updateTable('copper', data.copper_data.data);
    }

    updateTable(tableType, data) {
        const tableHead = document.getElementById(`${tableType}-table-head`);
        const tableBody = document.getElementById(`${tableType}-table-body`);
        const tableInfo = document.getElementById(`${tableType}-table-info`);

        if (!tableHead || !tableBody) return;

        if (!data || data.length === 0) {
            tableHead.innerHTML = '';
            tableBody.innerHTML = '<tr><td colspan="100%" class="no-data">No data available</td></tr>';
            if (tableInfo) {
                tableInfo.textContent = 'No data';
            }
            return;
        }

        // Create table headers
        const headers = Object.keys(data[0]);
        tableHead.innerHTML = headers.map(header => 
            `<th>${this.formatHeaderName(header)}</th>`
        ).join('');

        // Create table rows
        tableBody.innerHTML = data.map(row => 
            `<tr>${headers.map(header => 
                `<td>${this.formatCellValue(row[header])}</td>`
            ).join('')}</tr>`
        ).join('');

        // Update table info
        if (tableInfo) {
            tableInfo.textContent = `${data.length} record${data.length !== 1 ? 's' : ''}`;
        }
    }

    formatHeaderName(header) {
        return header.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    formatCellValue(value) {
        if (value === null || value === undefined) {
            return '<em>null</em>';
        }
        if (typeof value === 'object') {
            return JSON.stringify(value);
        }
        return String(value);
    }

    updateDifferences(comparison) {
        const differencesContainer = document.getElementById('differences-container');
        const differencesList = document.getElementById('differences-list');

        if (!differencesContainer || !differencesList) return;

        if (!comparison.differences || comparison.differences.length === 0) {
            differencesContainer.style.display = 'none';
            return;
        }

        differencesContainer.style.display = 'block';
        
        differencesList.innerHTML = comparison.differences.map(diff => 
            `<div class="difference-item">
                <i class="fas fa-exclamation-triangle"></i>
                ${diff}
            </div>`
        ).join('');
    }

    animateResults() {
        const cards = document.querySelectorAll('.summary-card');
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                card.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    showActionModal(action) {
        const modal = document.getElementById('action-modal');
        const modalTitle = document.getElementById('modal-title');
        const modalDescription = document.getElementById('modal-description');
        const modalDetails = document.getElementById('modal-details');

        if (!modal) return;

        if (modalTitle) {
            modalTitle.textContent = `Execute: ${action}`;
        }

        if (modalDescription) {
            modalDescription.textContent = `Are you sure you want to execute "${action}"?`;
        }

        if (modalDetails) {
            modalDetails.innerHTML = `
                <div class="modal-detail-item">
                    <strong>Input Value:</strong> ${this.currentData?.input_value || 'N/A'}
                </div>
                <div class="modal-detail-item">
                    <strong>Input Type:</strong> ${this.currentData?.input_type || 'N/A'}
                </div>
                <div class="modal-detail-item">
                    <strong>Estimated Time:</strong> ${this.getActionEstimatedTime(action)}
                </div>
            `;
        }

        // Store action for confirmation
        modal.dataset.action = action;

        modal.style.display = 'flex';
        modal.classList.add('animate-fade-in');
    }

    getActionEstimatedTime(action) {
        const timings = {
            'Run MASS SYNC job': '5-10 minutes',
            'Run Sync FLAG to N Job': '2-5 minutes',
            'Manual sync required': '10-30 minutes',
            'Investigate missing CoPPER data': 'Variable'
        };
        
        return timings[action] || '5-15 minutes';
    }

    hideModal() {
        const modal = document.getElementById('action-modal');
        if (modal) {
            modal.style.display = 'none';
            modal.classList.remove('animate-fade-in');
        }
    }

    async confirmAction() {
        const modal = document.getElementById('action-modal');
        const action = modal?.dataset.action;

        if (!action || !this.currentData) {
            this.showToast('Invalid action or no data available', 'error');
            return;
        }

        this.hideModal();
        this.showLoading(true, `Executing ${action}...`);

        try {
            const response = await fetch('/api/execute_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: action,
                    input_value: this.currentData.input_value
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.showToast(`Successfully executed: ${action}`, 'success');
                
                // Refresh data after action
                setTimeout(() => {
                    this.refreshData();
                }, 2000);
            } else {
                throw new Error(data.error || 'Action execution failed');
            }

        } catch (error) {
            console.error('Action execution error:', error);
            this.showToast(`Error executing action: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async refreshData() {
        if (!this.currentData) return;

        try {
            const response = await fetch('/api/reconcile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input_value: this.currentData.input_value,
                    scenario_type: 'guid_lookup'
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.currentData = data;
                this.displayResults(data);
                this.showToast('Data refreshed successfully', 'info');
            }

        } catch (error) {
            console.error('Data refresh error:', error);
            this.showToast('Failed to refresh data', 'warning');
        }
    }

    async checkSystemHealth() {
        const statusElement = document.querySelector('.connection-status span');
        const statusDot = document.querySelector('.status-dot');

        if (statusElement) {
            statusElement.textContent = 'Checking...';
        }

        if (statusDot) {
            statusDot.className = 'status-dot';
        }

        try {
            const response = await fetch('/api/health');
            const data = await response.json();

            if (response.ok && data.status === 'healthy') {
                if (statusElement) {
                    statusElement.textContent = 'Connected';
                }
                if (statusDot) {
                    statusDot.classList.add('connected');
                }
                
                // Check individual database connections
                if (data.database_status) {
                    const dbStatus = data.database_status;
                    const oscarStatus = dbStatus.oscar ? 'Connected' : 'Disconnected';
                    const copperStatus = dbStatus.copper ? 'Connected' : 'Disconnected';
                    
                    this.showToast(`OSCAR: ${oscarStatus}, CoPPER: ${copperStatus}`, 'info');
                }
            } else {
                throw new Error('System unhealthy');
            }

        } catch (error) {
            console.error('Health check error:', error);
            
            if (statusElement) {
                statusElement.textContent = 'Disconnected';
            }
            if (statusDot) {
                statusDot.classList.add('disconnected');
            }
            
            this.showToast('System health check failed', 'error');
        }
    }

    exportResults() {
        if (!this.currentData) {
            this.showToast('No data to export', 'warning');
            return;
        }

        try {
            const exportData = {
                input_value: this.currentData.input_value,
                input_type: this.currentData.input_type,
                timestamp: this.currentData.timestamp,
                oscar_data: this.currentData.oscar_data,
                copper_data: this.currentData.copper_data,
                comparison: this.currentData.comparison,
                scenario: this.currentData.scenario
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], {
                type: 'application/json'
            });

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `oscar_reconciliation_${this.currentData.input_value}_${new Date().toISOString().slice(0, 19)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showToast('Results exported successfully', 'success');

        } catch (error) {
            console.error('Export error:', error);
            this.showToast('Export failed', 'error');
        }
    }

    clearResults() {
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }

        this.currentData = null;
        this.showToast('Results cleared', 'info');
    }

    showLoading(show, message = 'Processing reconciliation...') {
        const overlay = document.getElementById('loading-overlay');
        const text = overlay?.querySelector('p');

        if (!overlay) return;

        if (show) {
            if (text) {
                text.textContent = message;
            }
            overlay.classList.remove('hidden');
            overlay.classList.add('animate-fade-in');
        } else {
            overlay.classList.add('hidden');
            overlay.classList.remove('animate-fade-in');
        }
    }

    showToast(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type} animate-slide-in-right`;
        
        const icon = this.getToastIcon(type);
        toast.innerHTML = `
            <i class="${icon}"></i>
            <span>${message}</span>
        `;

        container.appendChild(toast);

        // Auto remove toast
        setTimeout(() => {
            toast.classList.remove('animate-slide-in-right');
            toast.classList.add('animate-slide-out-right');
            
            setTimeout(() => {
                container.removeChild(toast);
            }, 300);
        }, duration);

        // Click to dismiss
        toast.addEventListener('click', () => {
            container.removeChild(toast);
        });
    }

    getToastIcon(type) {
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-times-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        return icons[type] || icons.info;
    }

    handleResize() {
        // Handle responsive adjustments
        const tables = document.querySelectorAll('.table-wrapper');
        tables.forEach(table => {
            if (window.innerWidth < 768) {
                table.style.overflowX = 'auto';
            }
        });
    }

    // Utility method to debounce function calls
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Method to create particle effects
    createParticleEffect(element, count = 10) {
        const rect = element.getBoundingClientRect();
        
        for (let i = 0; i < count; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.cssText = `
                left: ${rect.left + Math.random() * rect.width}px;
                top: ${rect.top + Math.random() * rect.height}px;
                width: ${Math.random() * 6 + 2}px;
                height: ${Math.random() * 6 + 2}px;
                background: hsl(${Math.random() * 360}, 70%, 60%);
            `;
            
            document.body.appendChild(particle);
            
            setTimeout(() => {
                document.body.removeChild(particle);
            }, 3000);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add page load animation
    document.body.classList.add('page-enter');
    
    setTimeout(() => {
        document.body.classList.add('page-enter-active');
    }, 10);

    // Initialize the main application
    window.oscarApp = new OSCARReconcile();

    // Add some Easter eggs
    const konamiCode = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65];
    let konamiIndex = 0;

    document.addEventListener('keydown', (e) => {
        if (e.keyCode === konamiCode[konamiIndex]) {
            konamiIndex++;
            if (konamiIndex === konamiCode.length) {
                // Easter egg: Party mode
                document.body.style.animation = 'rubberBand 1s infinite';
                setTimeout(() => {
                    document.body.style.animation = '';
                }, 5000);
                konamiIndex = 0;
            }
        } else {
            konamiIndex = 0;
        }
    });
});

// Service worker registration for PWA capabilities
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}























# OSCAR Reconciliation Tool - Python Dependencies

# Core Flask Framework
Flask==2.3.3
Flask-CORS==4.0.0

# Database Connections
psycopg2-binary==2.9.7
SQLAlchemy==2.0.21

# Environment Variables
python-dotenv==1.0.0

# HTTP Client for external APIs
requests==2.31.0

# JSON Web Tokens (if needed for authentication)
PyJWT==2.8.0

# Date and Time handling
python-dateutil==2.8.2

# Logging and Monitoring
structlog==23.1.0

# Configuration Management
pydantic==2.4.2
pydantic-settings==2.0.3

# Data Validation
cerberus==1.3.4

# Excel/CSV Processing (for exports)
pandas==2.1.1
openpyxl==3.1.2

# Testing Dependencies
pytest==7.4.2
pytest-flask==1.2.0
pytest-mock==3.11.1

# Code Quality
flake8==6.1.0
black==23.9.1
isort==5.12.0

# Security
cryptography==41.0.4

# Development Tools
watchdog==3.0.0

# Google Cloud SQL Proxy (if using Google Cloud)
cloud-sql-python-connector==1.4.3

# Performance Monitoring
psutil==5.9.5

# Background Tasks (if needed)
celery==5.3.2
redis==5.0.0

# API Documentation (optional)
flask-restx==1.2.0

# WSGI Server for Production
gunicorn==21.2.0
















# OSCAR Reconciliation Tool - Environment Variables Template
# Copy this file to .env and fill in your actual values

# Application Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
DEBUG=True
HOST=0.0.0.0
PORT=5000

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=oscar_tool.log

# OSCAR Database Configuration
OSCAR_DB_HOST=localhost
OSCAR_DB_PORT=5432
OSCAR_DB_NAME=oscar_db
OSCAR_DB_USER=oscar_user
OSCAR_DB_PASSWORD=oscar_password
OSCAR_DB_SCHEMA=PR01COSRO

# CoPPER Database Configuration
COPPER_DB_HOST=localhost
COPPER_DB_PORT=5432
COPPER_DB_NAME=copper_db
COPPER_DB_USER=copper_user
COPPER_DB_PASSWORD=copper_password
COPPER_DB_SCHEMA=pr01ccrdo

# Google Cloud SQL Proxy Configuration (if using Google Cloud)
CLOUD_SQL_CONNECTION_NAME=your-project:region:instance-name
ENABLE_IAM_AUTH=True
IP_TYPE=private

# Reconciliation Configuration
MAX_RECORDS_PER_QUERY=1000
QUERY_TIMEOUT=30
BATCH_SIZE=100

# Redis Configuration (for caching/background tasks)
REDIS_URL=redis://localhost:6379/0

# Performance Monitoring
ENABLE_MONITORING=True
MONITORING_ENDPOINT=

# Security Configuration
CORS_ORIGINS=http://localhost:5000,http://127.0.0.1:5000
SESSION_TIMEOUT=3600

# Email Configuration (for notifications)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@company.com
SMTP_PASSWORD=your-email-password
EMAIL_FROM=oscar-tool@company.com

# Cloud Storage (for exports/backups)
CLOUD_STORAGE_BUCKET=your-storage-bucket
CLOUD_STORAGE_PATH=oscar-exports/

# API Keys (if integrating with external services)
EXTERNAL_API_KEY=your-external-api-key
EXTERNAL_API_URL=https://api.external-service.com

# Development vs Production
ENVIRONMENT=development















# OSCAR Reconciliation Tool

A comprehensive web application for reconciling financial trading data between OSCAR and CoPPER systems. This tool provides real-time data comparison, discrepancy identification, and automated synchronization capabilities.

## 🎯 Features

- **Real-time Data Reconciliation**: Compare OSCAR and CoPPER databases instantly
- **Multiple Input Types**: Support for GUID (12 chars), GUS (5 chars), and GFID (4 chars)
- **Scenario-based Analysis**: Handle multiple reconciliation scenarios (2.1, 2.2, 2.3, 2.4)
- **Automated Actions**: Execute mass sync jobs and flag synchronization
- **Beautiful UI**: Modern, responsive interface with smooth animations
- **Export Capabilities**: Export reconciliation results to JSON
- **Health Monitoring**: Real-time system health checks
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS) → Flask API → PostgreSQL Connections → Results Display
```

### Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Database**: PostgreSQL (OSCAR & CoPPER)
- **Styling**: Custom CSS with animations
- **Icons**: Font Awesome
- **Fonts**: Inter (Google Fonts)

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL databases (OSCAR and CoPPER)
- Google Cloud SQL Proxy (if using Cloud SQL)
- Modern web browser

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd oscar-reconciliation-tool
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the template
cp .env.template .env

# Edit .env with your actual database credentials
nano .env  # or use your preferred editor
```

### 5. Database Setup

Ensure your PostgreSQL databases are accessible and contain the required tables:

**OSCAR Database Tables:**
- `PR01COSRO.ACTIVE_XML_DATA_STORE`
- `PR01COSRO.XML_DATA_INTERIM`

**CoPPER Database Tables:**
- `pr01ccrdo.trd_guid`
- `NR01CCRDO.TRD_SESSION_PRODUCT_PERMISSION`
- `NR01CEBDO.GB_SESSION`

### 6. Start the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database Connections
OSCAR_DB_HOST=your-oscar-host
OSCAR_DB_USER=your-oscar-user
OSCAR_DB_PASSWORD=your-oscar-password

COPPER_DB_HOST=your-copper-host
COPPER_DB_USER=your-copper-user
COPPER_DB_PASSWORD=your-copper-password

# Application Settings
SECRET_KEY=your-secret-key
DEBUG=True
PORT=5000
```

### Cloud SQL Proxy Setup

If using Google Cloud SQL:

1. Install Cloud SQL Proxy:
```bash
# Download and install Cloud SQL Proxy
curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64
chmod +x cloud_sql_proxy
```

2. Start the proxy:
```bash
./cloud_sql_proxy -instances=PROJECT:REGION:INSTANCE=tcp:5432
```

3. Enable IAM authentication in your `.env`:
```bash
ENABLE_IAM_AUTH=True
CLOUD_SQL_CONNECTION_NAME=your-project:region:instance
```

## 📊 Usage

### Basic Reconciliation

1. **Enter Input**: Type a GUID (12 chars), GUS (5 chars), or GFID (4 chars)
2. **Select Scenario**: Choose the appropriate reconciliation scenario
3. **Submit**: Click "Compare Data" to start reconciliation
4. **Review Results**: Analyze the comparison results and differences
5. **Execute Actions**: Run recommended sync jobs if needed

### Input Types

| Type | Length | Example | Description |
|------|--------|---------|-------------|
| GUID | 12 | ABCDEFGH1234 | Globex Unique Identifier |
| GUS | 5 | ABCDE | Globex User Signature |
| GFID | 4 | ABCD | Globex Firm ID |

### Reconciliation Scenarios

#### Scenario 2.1: Both Expired
- **Condition**: Expired GUID in both OSCAR and CoPPER
- **Action**: Run MASS SYNC job and Sync FLAG to N Job

#### Scenario 2.2: OSCAR Expired, CoPPER Active
- **Condition**: OSCAR has expired GUID, CoPPER has active with different GUID
- **Action**: Run Sync FLAG to N Job

#### Scenario 2.3: OSCAR Expired, CoPPER Missing
- **Condition**: Expired GUID in OSCAR, missing in CoPPER
- **Action**: Run Sync FLAG to N Job

#### Scenario 2.4: OSCAR Active, CoPPER Missing
- **Condition**: Active GUID in OSCAR, missing in CoPPER
- **Action**: Investigate and manual sync

## 🎨 UI Features

### Modern Design
- Clean, professional interface
- Gradient backgrounds and glassmorphism effects
- Smooth animations and transitions
- Responsive grid layouts

### Interactive Elements
- Real-time input validation
- Loading states with spinners
- Toast notifications for feedback
- Modal dialogs for confirmations

### Data Visualization
- Summary cards with metrics
- Side-by-side comparison tables
- Difference highlighting
- Status badges and indicators

## 🔌 API Endpoints

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard page |
| `/api/reconcile` | POST | Execute reconciliation |
| `/api/execute_action` | POST | Execute sync actions |
| `/api/health` | GET | System health check |

### API Request Examples

**Reconciliation Request:**
```json
{
  "input_value": "ABCDEFGH1234",
  "scenario_type": "scenario_2_1"
}
```

**Action Execution:**
```json
{
  "action": "Run MASS SYNC job",
  "input_value": "ABCDEFGH1234"
}
```

## 🔒 Security

### Database Security
- Use connection pooling
- Implement query parameterization to prevent SQL injection
- Use IAM authentication for Cloud SQL
- Rotate database credentials regularly

### Application Security
- CSRF protection enabled
- Secure session management
- Input validation and sanitization
- HTTPS enforcement in production

## 📈 Monitoring

### Health Checks
- Database connectivity monitoring
- Real-time connection status display
- Automatic retry mechanisms
- Performance metrics tracking

### Logging
- Structured logging with timestamps
- Error tracking and reporting
- Query performance monitoring
- User action auditing

## 🔧 Development

### Project Structure

```
oscar-reconciliation-tool/
├── app.py                 # Flask application
├── database.py           # Database manager
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── .env.template         # Environment variables template
├── README.md            # Documentation
├── static/
│   ├── style.css        # Main styles
│   ├── animations.css   # Animation styles
│   └── script.js        # Frontend JavaScript
└── templates/
    └── index.html       # Main template
```

### Adding New Scenarios

1. **Update Configuration** (`config.py`):
```python
SCENARIOS = {
    'new_scenario': {
        'name': 'New Scenario Name',
        'oscar_query': 'SELECT ...',
        'copper_query': 'SELECT ...',
        'recommended_actions': ['Action 1', 'Action 2']
    }
}
```

2. **Update Frontend** (`templates/index.html`):
```html
<option value="new_scenario">New Scenario Description</option>
```

3. **Update Logic** (`app.py`):
```python
def determine_scenario(oscar_data, copper_data, comparison):
    # Add new scenario logic
    if condition:
        scenario.update({
            'type': 'NEW_SCENARIO',
            'description': 'Description',
            'recommended_actions': ['Actions'],
            'severity': 'MEDIUM'
        })
```

### Testing

Run tests with pytest:
```bash
# Install test dependencies
pip install pytest pytest-flask pytest-mock

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Code Quality

Format code with black and isort:
```bash
# Format Python code
black .
isort .

# Check for style issues
flake8 .
```

## 🚀 Deployment

### Production Configuration

1. **Update Environment Variables**:
```bash
DEBUG=False
SECRET_KEY=your-production-secret-key
ENVIRONMENT=production
```

2. **Use Production WSGI Server**:
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Set Up Reverse Proxy** (nginx example):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t oscar-reconcile .
docker run -p 5000:5000 --env-file .env oscar-reconcile
```

## 🐛 Troubleshooting

### Common Issues

**Database Connection Failed:**
- Check database credentials in `.env`
- Verify network connectivity
- Ensure Cloud SQL Proxy is running
- Check firewall settings

**Import Errors:**
- Verify virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version compatibility

**Frontend Not Loading:**
- Clear browser cache
- Check browser console for errors
- Verify static files are served correctly

**Performance Issues:**
- Check database query performance
- Monitor connection pool usage
- Optimize query indices
- Consider caching frequently accessed data

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG
DEBUG=True
```

### Support

For technical support:
1. Check the logs: `tail -f oscar_tool.log`
2. Verify configuration: Review `.env` settings
3. Test database connections: Use health check endpoint
4. Review error messages: Check browser console and server logs

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📞 Contact

For questions or support, please contact the development team.

---

**Built with ❤️ for financial data reconciliation**














#!/usr/bin/env python3
"""
OSCAR Reconciliation Tool - Application Runner
Simple script to start the Flask application with proper configuration
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from app import app
    from config import APP_CONFIG, LOGGING_CONFIG
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def setup_logging():
    """Configure application logging"""
    log_level = getattr(logging, LOGGING_CONFIG['level'].upper(), logging.INFO)
    log_format = LOGGING_CONFIG['format']
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGGING_CONFIG['file'])
        ]
    )
    
    # Suppress some verbose loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def check_environment():
    """Check if required environment variables are set"""
    required_vars = [
        'OSCAR_DB_HOST',
        'OSCAR_DB_USER', 
        'OSCAR_DB_PASSWORD',
        'COPPER_DB_HOST',
        'COPPER_DB_USER',
        'COPPER_DB_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease copy .env.template to .env and fill in the required values:")
        print("cp .env.template .env")
        return False
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'flask',
        'psycopg2',
        'sqlalchemy',
        'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required Python packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install dependencies:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def display_startup_info():
    """Display startup information"""
    print("🚀 OSCAR Reconciliation Tool")
    print("=" * 50)
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Debug Mode: {APP_CONFIG['debug']}")
    print(f"Host: {APP_CONFIG['host']}")
    print(f"Port: {APP_CONFIG['port']}")
    print(f"Log Level: {LOGGING_CONFIG['level']}")
    print("=" * 50)
    print(f"🌐 Application will be available at: http://{APP_CONFIG['host']}:{APP_CONFIG['port']}")
    print("=" * 50)

def main():
    """Main application entry point"""
    print("Starting OSCAR Reconciliation Tool...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    # Display startup information
    display_startup_info()
    
    try:
        # Test database connections on startup
        from database import DatabaseManager
        db_manager = DatabaseManager()
        
        print("🔍 Testing database connections...")
        health_status = db_manager.check_connection()
        
        if health_status['oscar']:
            print("✅ OSCAR database connection: OK")
        else:
            print("❌ OSCAR database connection: FAILED")
        
        if health_status['copper']:
            print("✅ CoPPER database connection: OK") 
        else:
            print("❌ CoPPER database connection: FAILED")
        
        if not (health_status['oscar'] and health_status['copper']):
            print("\n⚠️  Warning: Some database connections failed.")
            print("The application will start but may not function correctly.")
            print("Please check your database configuration in the .env file.")
        
        print("\n🎉 Starting Flask application...")
        
        # Start the Flask application
        app.run(
            host=APP_CONFIG['host'],
            port=APP_CONFIG['port'],
            debug=APP_CONFIG['debug'],
            use_reloader=APP_CONFIG['debug'],
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()



















# OSCAR Reconciliation Tool - Docker Configuration
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --shell /bin/bash oscar_user \
    && chown -R oscar_user:oscar_user /app
USER oscar_user

# Copy requirements first for better caching
COPY --chown=oscar_user:oscar_user requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=oscar_user:oscar_user . .

# Create necessary directories
RUN mkdir -p logs static/uploads

# Expose the port the app runs on
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Default command
CMD ["python", "run.py"]










# OSCAR Reconciliation Tool - Docker Compose Configuration
version: '3.8'

services:
  # Main application
  oscar-app:
    build: .
    container_name: oscar_reconcile_app
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
      - DEBUG=True
      - OSCAR_DB_HOST=oscar-db
      - COPPER_DB_HOST=copper-db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./logs:/app/logs
      - ./.env:/app/.env
    depends_on:
      - oscar-db
      - copper-db
      - redis
    networks:
      - oscar-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # OSCAR Database (PostgreSQL)
  oscar-db:
    image: postgres:15-alpine
    container_name: oscar_db
    environment:
      POSTGRES_DB: oscar_db
      POSTGRES_USER: oscar_user
      POSTGRES_PASSWORD: oscar_password
      POSTGRES_INITDB_ARGS: "--encoding=UTF8"
    volumes:
      - oscar_db_data:/var/lib/postgresql/data
      - ./database/oscar_init.sql:/docker-entrypoint-initdb.d/01_init.sql
    ports:
      - "5432:5432"
    networks:
      - oscar-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U oscar_user -d oscar_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # CoPPER Database (PostgreSQL)
  copper-db:
    image: postgres:15-alpine
    container_name: copper_db
    environment:
      POSTGRES_DB: copper_db
      POSTGRES_USER: copper_user
      POSTGRES_PASSWORD: copper_password
      POSTGRES_INITDB_ARGS: "--encoding=UTF8"
    volumes:
      - copper_db_data:/var/lib/postgresql/data
      - ./database/copper_init.sql:/docker-entrypoint-initdb.d/01_init.sql
    ports:
      - "5433:5432"
    networks:
      - oscar-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U copper_user -d copper_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for caching and background tasks
  redis:
    image: redis:7-alpine
    container_name: oscar_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - oscar-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Database Administration UI (optional)
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: oscar_pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@oscar.com
      PGADMIN_DEFAULT_PASSWORD: admin
      PGADMIN_CONFIG_SERVER_MODE: 'False'
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    ports:
      - "8080:80"
    networks:
      - oscar-network
    restart: unless-stopped
    profiles:
      - tools

  # Nginx reverse proxy (for production)
  nginx:
    image: nginx:alpine
    container_name: oscar_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - oscar-app
    networks:
      - oscar-network
    restart: unless-stopped
    profiles:
      - production

  # Monitoring with Prometheus (optional)
  prometheus:
    image: prom/prometheus:latest
    container_name: oscar_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - oscar-network
    restart: unless-stopped
    profiles:
      - monitoring

  # Grafana for dashboards (optional)
  grafana:
    image: grafana/grafana:latest
    container_name: oscar_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - oscar-network
    restart: unless-stopped
    profiles:
      - monitoring

# Volumes for persistent data
volumes:
  oscar_db_data:
    driver: local
  copper_db_data:
    driver: local
  redis_data:
    driver: local
  pgadmin_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

# Networks
networks:
  oscar-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

















#!/bin/bash
# OSCAR Reconciliation Tool - Setup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.8"
        
        if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
            print_success "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python $PYTHON_VERSION found, but Python $REQUIRED_VERSION or higher is required"
            return 1
        fi
    else
        print_error "Python 3 is not installed"
        return 1
    fi
}

# Function to check pip
check_pip() {
    if command_exists pip3; then
        print_success "pip3 found"
        return 0
    elif command_exists pip; then
        print_success "pip found"
        return 0
    else
        print_error "pip is not installed"
        return 1
    fi
}

# Function to create virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_success "pip upgraded"
}

# Function to install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        return 1
    fi
}

# Function to setup environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            print_success "Environment file created from template"
            print_warning "Please edit .env file with your database credentials"
        else
            print_error ".env.template not found"
            return 1
        fi
    else
        print_warning ".env file already exists"
    fi
}

# Function to check database connectivity
check_databases() {
    print_status "Checking database connectivity..."
    
    # Source environment variables
    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
    fi
    
    # Check if required variables are set
    if [ -z "$OSCAR_DB_HOST" ] || [ -z "$COPPER_DB_HOST" ]; then
        print_warning "Database configuration not complete in .env file"
        return 1
    fi
    
    # Test database connections using Python
    python3 -c "
import os
import sys
sys.path.insert(0, '.')
try:
    from database import DatabaseManager
    db = DatabaseManager()
    status = db.check_connection()
    if status['oscar']:
        print('✅ OSCAR database: Connected')
    else:
        print('❌ OSCAR database: Failed')
    if status['copper']:
        print('✅ CoPPER database: Connected')
    else:
        print('❌ CoPPER database: Failed')
    
    if status['oscar'] and status['copper']:
        exit(0)
    else:
        exit(1)
except Exception as e:
    print(f'❌ Database check failed: {e}')
    exit(1)
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Database connectivity check passed"
    else
        print_warning "Database connectivity check failed - app may not work correctly"
    fi
}

# Function to setup Docker environment
setup_docker() {
    if command_exists docker && command_exists docker-compose; then
        print_status "Setting up Docker environment..."
        
        # Create necessary directories
        mkdir -p logs database monitoring/grafana/{dashboards,datasources} nginx/ssl
        
        print_success "Docker directories created"
        print_status "You can now use 'docker-compose up -d' to start the application"
    else
        print_warning "Docker not found - skipping Docker setup"
    fi
}

# Function to create sample database init files
create_sample_db_files() {
    print_status "Creating sample database initialization files..."
    
    mkdir -p database
    
    # OSCAR database init
    cat > database/oscar_init.sql << 'EOF'
-- OSCAR Database Initialization
-- Sample schema and tables for development

CREATE SCHEMA IF NOT EXISTS PR01COSRO;

-- Sample table structure (adjust according to your actual schema)
CREATE TABLE IF NOT EXISTS PR01COSRO.ACTIVE_XML_DATA_STORE (
    GUID VARCHAR(12) PRIMARY KEY,
    XML TEXT,
    STATUS VARCHAR(20),
    CREATED_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS PR01COSRO.XML_DATA_INTERIM (
    ID SERIAL PRIMARY KEY,
    GUID VARCHAR(12),
    VERSION VARCHAR(10),
    DATA TEXT,
    CREATED_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data for testing
INSERT INTO PR01COSRO.ACTIVE_XML_DATA_STORE (GUID, XML, STATUS) VALUES 
('TESTGUID0001', '<data><expDate>2024-12-31</expDate><idNumber>12345</idNumber><globexFirmID>TEST</globexFirmID><status>ACTIVE</status></data>', 'ACTIVE'),
('TESTGUID0002', '<data><expDate>2023-12-31</expDate><idNumber>12346</idNumber><globexFirmID>TEST</globexFirmID><status>EXPIRED</status></data>', 'EXPIRED');
EOF

    # CoPPER database init
    cat > database/copper_init.sql << 'EOF'
-- CoPPER Database Initialization
-- Sample schema and tables for development

CREATE SCHEMA IF NOT EXISTS pr01ccrdo;
CREATE SCHEMA IF NOT EXISTS NR01CCRDO;
CREATE SCHEMA IF NOT EXISTS NR01CEBDO;

-- Sample table structures (adjust according to your actual schema)
CREATE TABLE IF NOT EXISTS pr01ccrdo.trd_guid (
    ID SERIAL PRIMARY KEY,
    GFID_ID VARCHAR(12),
    GPID_ID VARCHAR(10),
    GFID_GUID VARCHAR(10),
    STATUS VARCHAR(20),
    CREATED_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS NR01CCRDO.TRD_SESSION_PRODUCT_PERMISSION (
    SESSION_ID VARCHAR(10),
    PRODUCT_ID VARCHAR(20),
    PERMISSION VARCHAR(20),
    CREATED_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS NR01CEBDO.GB_SESSION (
    SESSION_ID VARCHAR(10) PRIMARY KEY,
    SESSION_TYPE VARCHAR(20),
    STATUS VARCHAR(20),
    CREATED_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data for testing
INSERT INTO pr01ccrdo.trd_guid (GFID_ID, GPID_ID, GFID_GUID, STATUS) VALUES 
('TESTGUID0001', 'CAIBE', 'VGARC', 'ACTIVE'),
('TESTGUID0003', 'CAIBE', 'JRCOS', 'ACTIVE');

INSERT INTO NR01CEBDO.GB_SESSION (SESSION_ID, SESSION_TYPE, STATUS) VALUES 
('MDBLZ', 'MD', 'ACTIVE'),
('FIF', 'OE', 'ACTIVE');
EOF

    print_success "Sample database files created"
}

# Main setup function
main() {
    echo "🚀 OSCAR Reconciliation Tool - Setup Script"
    echo "============================================"
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! check_python; then
        print_error "Please install Python 3.8 or higher"
        exit 1
    fi
    
    if ! check_pip; then
        print_error "Please install pip"
        exit 1
    fi
    
    # Setup virtual environment
    setup_venv
    
    # Install dependencies
    install_dependencies
    
    # Setup environment
    setup_env
    
    # Create sample database files
    create_sample_db_files
    
    # Setup Docker if available
    setup_docker
    
    # Check database connectivity
    check_databases
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your database credentials"
    echo "2. Start the application:"
    echo "   python run.py"
    echo ""
    echo "Or using Docker:"
    echo "   docker-compose up -d"
    echo ""
    echo "The application will be available at: http://localhost:5000"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main "$@"
