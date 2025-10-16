from google.oauth2 import service_account
import os

SERVICE_ACCOUNT_KEY_PATH = "./service-account-key.json"

def test_service_account():
    try:
        if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
            print(f"❌ Service account key not found at: {SERVICE_ACCOUNT_KEY_PATH}")
            return False
        
        print(f"✅ Found service account key file")
        
        # Load credentials
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_KEY_PATH,
            scopes=["https://www.googleapis.com/auth/sqlservice.admin"]
        )
        
        print(f"✅ Service Account Email: {credentials.service_account_email}")
        print(f"✅ Project ID: {credentials.project_id}")
        print(f"✅ Credentials loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_service_account()
