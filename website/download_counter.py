#!/usr/bin/env python3
"""
Simple download counter API using Flask
Stores count in a JSON file on the server
"""
import json
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# File to store the counter
COUNTER_FILE = '/var/www/sortmoments/data/counter.json'
DATA_DIR = Path(COUNTER_FILE).parent

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

class CounterHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Enable CORS
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        if path == '/api/counter':
            # Return current count
            count = self.read_counter()
            response = json.dumps({'count': count})
            self.wfile.write(response.encode())
        elif path == '/api/counter/increment':
            # Increment and return new count
            count = self.increment_counter()
            response = json.dumps({'count': count})
            self.wfile.write(response.encode())
        else:
            response = json.dumps({'error': 'Not found'})
            self.wfile.write(response.encode())

    def do_HEAD(self):
        """Handle HEAD requests"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

    @staticmethod
    def read_counter():
        """Read counter from file"""
        try:
            if os.path.exists(COUNTER_FILE):
                with open(COUNTER_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('count', 0)
        except Exception as e:
            print(f"Error reading counter: {e}")
        return 0

    @staticmethod
    def increment_counter():
        """Increment counter and save to file"""
        try:
            count = CounterHandler.read_counter()
            count += 1
            with open(COUNTER_FILE, 'w') as f:
                json.dump({'count': count}, f)
            return count
        except Exception as e:
            print(f"Error incrementing counter: {e}")
            return 0

if __name__ == '__main__':
    # Run on localhost:8888
    server = HTTPServer(('127.0.0.1', 8888), CounterHandler)
    print('Counter API running on http://127.0.0.1:8888')
    server.serve_forever()
