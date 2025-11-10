#!/usr/bin/env python3
"""
Simple HTTP server to serve the annotation comparison tool
Avoids CORS issues when loading local JSON files
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

def start_server(port=8000):
    """Start HTTP server in the current directory"""
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Create handler with CORS headers
    class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    # Start server
    with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
        print(f"Serving annotation comparison tool at http://localhost:{port}")
        print(f"Open http://localhost:{port}/annotation_comparison.html in your browser")
        print("Press Ctrl+C to stop the server")
        
        # Try to open browser automatically
        try:
            webbrowser.open(f'http://localhost:{port}/annotation_comparison.html')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port)