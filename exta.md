if __name__ == '__main__':
    print("=" * 60)
    print("OSCAR RECONCILIATION TOOL - STARTING")
    print("=" * 60)
    
    # Check if certificate file exists
    import os
    cert_path = CA_CERT_PATH
    if not os.path.exists(cert_path):
        print(f"❌ ERROR: Certificate file not found at: {cert_path}")
        print(f"   Please update CA_CERT_PATH in app.py to point to your new.pem file")
        print(f"   Current working directory: {os.getcwd()}")
        exit(1)
    else:
        print(f"✅ Certificate file found: {cert_path}")
    
    print("\n" + "=" * 60)
    print("Initializing database connections...")
    print("=" * 60)
    
    try:
        # Try to initialize connections on startup
        print("\n🔄 Testing database connections...")
        engine = get_recon_engine()
        print("✅ All database connections initialized successfully!\n")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize connections on startup")
        print(f"   Error: {str(e)}")
        print(f"   The app will try to connect when first request is made.\n")
    
    print("=" * 60)
    print("🚀 Starting Flask application...")
    print("=" * 60)
    print("\n📍 Access the application at: http://localhost:5000")
    print("📍 Health check endpoint: http://localhost:5000/api/health")
    print("\n⏹️  Press CTRL+C to stop the server\n")
    print("=" * 60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to start Flask app")
        print(f"   {str(e)}")
        exit(1)
