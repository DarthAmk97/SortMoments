#!/usr/bin/env python3
import argparse
from http import cookies
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# Cache file for the public landing-page count. The source of truth is GitHub
# Releases download_count; this file keeps the site fast and avoids GitHub API
# calls from every visitor's browser.
COUNTER_FILE = '/var/www/sortmoments/data/counter.json'
VISITOR_DB_FILE = '/var/www/sortmoments/data/analytics.sqlite3'
VISITOR_SECRET_FILE = '/var/www/sortmoments/data/visitor_salt.secret'
VISITOR_COOKIE_NAME = 'sortmoments_visitor_id'
VISITOR_COOKIE_MAX_AGE = 365 * 24 * 60 * 60
VISITOR_BASELINE_OFFSET_KEY = 'visitor_baseline_offset'
DOWNLOAD_DISPLAY_FLOOR_KEY = 'download_display_floor'
DATA_DIR = Path(COUNTER_FILE).parent
GITHUB_RELEASE_API = 'https://api.github.com/repos/DarthAmk97/SortMoments/releases/tags/v1.0.0-beta'
TRACKED_ASSET_EXTENSIONS = ('.exe', '.zip', '.dmg')
TRACKED_ASSET_PREFIX = 'SortMoments'
CACHE_TTL_SECONDS = 15 * 60

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

class CounterHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = parsed_path.query

        if path == '/api/counter':
            # Return current GitHub release download count, using the local
            # cache if GitHub is unavailable or the cache is still fresh.
            count = self.read_counter()
            self._send_json({'count': count})
        elif path == '/api/counter/increment':
            # Record a click-driven public floor so the on-page count updates
            # immediately. GitHub Releases remains the external source; reads
            # return max(GitHub count, this local floor) so GitHub can catch up.
            count = self.increment_counter()
            self._send_json({'count': count})
        elif path in ('/api/analytics', '/api/visitor'):
            # track=0 is useful for health checks and previews that should not
            # create a unique visitor record.
            should_track = 'track=0' not in query and 'track=false' not in query
            payload, cookie_header = self.read_analytics(
                should_track=should_track,
                cookie_header=self.headers.get('Cookie', ''),
            )
            extra_headers = {}
            if cookie_header:
                extra_headers['Set-Cookie'] = cookie_header
            self._send_json(payload, extra_headers=extra_headers)
        else:
            self._send_json({'error': 'Not found'}, status=404)

    def do_HEAD(self):
        """Handle HEAD requests"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, max-age=0')
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

    def _send_json(self, payload, status=200, extra_headers=None):
        """Send a JSON response with common privacy/cache headers."""
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, max-age=0')
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _read_cache():
        """Read cached counter data from disk."""
        try:
            if os.path.exists(COUNTER_FILE):
                with open(COUNTER_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading counter: {e}")
        return {}

    @staticmethod
    def _write_cache(data):
        """Atomically write cached counter data."""
        tmp_file = f"{COUNTER_FILE}.tmp"
        with open(tmp_file, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_file, COUNTER_FILE)

    @staticmethod
    def _fetch_github_release_count():
        """Fetch and sum tracked GitHub release asset download counts."""
        request = Request(
            GITHUB_RELEASE_API,
            headers={
                'Accept': 'application/vnd.github+json',
                'User-Agent': 'sortmoments-download-counter',
            },
        )

        with urlopen(request, timeout=8) as response:
            release = json.loads(response.read().decode('utf-8'))

        assets = {}
        total = 0
        for asset in release.get('assets', []):
            name = asset.get('name')
            if CounterHandler._is_tracked_release_asset(name):
                count = int(asset.get('download_count') or 0)
                assets[name] = count
                total += count

        if not assets:
            raise RuntimeError('No tracked release assets found in GitHub release response')

        return {
            'count': total,
            'updated_at': int(time.time()),
            'source': 'github_releases',
            'tag': release.get('tag_name', 'v1.0.0-beta'),
            'assets': assets,
        }

    @staticmethod
    def _is_tracked_release_asset(name):
        """Return True for product installer/download assets we want to count."""
        if not isinstance(name, str):
            return False
        return (
            name.startswith(TRACKED_ASSET_PREFIX)
            and name.endswith(TRACKED_ASSET_EXTENSIONS)
        )

    @staticmethod
    def refresh_counter_cache():
        """Force-refresh the counter cache from GitHub and return the full data."""
        fresh = CounterHandler._fetch_github_release_count()
        CounterHandler._write_cache(fresh)
        return fresh

    @staticmethod
    def read_counter():
        """Return cached GitHub release total with the local display floor."""
        cached = CounterHandler._read_cache()
        now = int(time.time())

        # Backwards-compatible support for the original {"count": N} file.
        cached_count = int(cached.get('count') or 0) if isinstance(cached, dict) else 0
        updated_at = int(cached.get('updated_at') or 0) if isinstance(cached, dict) else 0

        if cached_count and updated_at and now - updated_at < CACHE_TTL_SECONDS:
            return CounterHandler._apply_download_display_floor(cached_count)

        try:
            fresh = CounterHandler.refresh_counter_cache()
            return CounterHandler._apply_download_display_floor(fresh['count'])
        except Exception as e:
            print(f"Error fetching GitHub release counter: {e}")
            return CounterHandler._apply_download_display_floor(cached_count)

    @staticmethod
    def _read_meta_int(key, default=0):
        conn = CounterHandler._ensure_visitor_db()
        try:
            row = conn.execute(
                'SELECT value FROM analytics_meta WHERE key = ?',
                (key,),
            ).fetchone()
            if not row:
                return default
            try:
                return int(row[0])
            except (TypeError, ValueError):
                return default
        finally:
            conn.close()

    @staticmethod
    def _write_meta_int(key, value):
        conn = CounterHandler._ensure_visitor_db()
        try:
            conn.execute(
                '''
                INSERT INTO analytics_meta (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                ''',
                (key, str(int(value)), int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _apply_download_display_floor(count):
        floor = CounterHandler._read_meta_int(DOWNLOAD_DISPLAY_FLOOR_KEY, 0)
        return max(int(count or 0), floor)

    @staticmethod
    def increment_counter():
        """Increment the public download floor and return the displayed total."""
        next_count = CounterHandler.read_counter() + 1
        CounterHandler._write_meta_int(DOWNLOAD_DISPLAY_FLOOR_KEY, next_count)
        return next_count

    @staticmethod
    def _ensure_visitor_secret():
        """Create/read the local secret used to hash visitor cookie IDs."""
        secret_path = Path(VISITOR_SECRET_FILE)
        if secret_path.exists():
            return secret_path.read_bytes()

        secret = secrets.token_bytes(32)
        fd = os.open(str(secret_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(secret)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        return secret

    @staticmethod
    def _visitor_hash(visitor_id):
        secret = CounterHandler._ensure_visitor_secret()
        return hmac.new(secret, visitor_id.encode('utf-8'), hashlib.sha256).hexdigest()

    @staticmethod
    def _ensure_visitor_db():
        conn = sqlite3.connect(VISITOR_DB_FILE, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS visitors (
                visitor_hash TEXT PRIMARY KEY,
                first_seen INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                visits INTEGER NOT NULL DEFAULT 1
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS analytics_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )
            '''
        )
        return conn

    @staticmethod
    def _parse_visitor_cookie(cookie_header):
        if not cookie_header:
            return None
        jar = cookies.SimpleCookie()
        try:
            jar.load(cookie_header)
        except cookies.CookieError:
            return None
        morsel = jar.get(VISITOR_COOKIE_NAME)
        if not morsel:
            return None
        value = morsel.value.strip()
        # token_urlsafe(32) is normally 43 chars. Keep this permissive enough
        # for future format changes while rejecting obviously bad values.
        if len(value) < 24 or len(value) > 128:
            return None
        return value

    @staticmethod
    def _build_visitor_cookie(visitor_id):
        return (
            f'{VISITOR_COOKIE_NAME}={visitor_id}; '
            f'Max-Age={VISITOR_COOKIE_MAX_AGE}; '
            'Path=/; Secure; HttpOnly; SameSite=Lax'
        )

    @staticmethod
    def _record_visitor(visitor_id):
        visitor_hash = CounterHandler._visitor_hash(visitor_id)
        now = int(time.time())

        conn = CounterHandler._ensure_visitor_db()
        try:
            conn.execute(
                '''
                INSERT INTO visitors (visitor_hash, first_seen, last_seen, visits)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(visitor_hash) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    visits = visits + 1
                ''',
                (visitor_hash, now, now),
            )
            conn.commit()
            row = conn.execute('SELECT COUNT(*) FROM visitors').fetchone()
            raw_unique_visitors = int(row[0] or 0)
            return CounterHandler._apply_visitor_baseline(raw_unique_visitors, conn)
        finally:
            conn.close()

    @staticmethod
    def _read_unique_visitors():
        conn = CounterHandler._ensure_visitor_db()
        try:
            row = conn.execute('SELECT COUNT(*) FROM visitors').fetchone()
            raw_unique_visitors = int(row[0] or 0)
            return CounterHandler._apply_visitor_baseline(raw_unique_visitors, conn)
        finally:
            conn.close()

    @staticmethod
    def _read_raw_unique_visitors(conn):
        row = conn.execute('SELECT COUNT(*) FROM visitors').fetchone()
        return int(row[0] or 0)

    @staticmethod
    def _read_visitor_baseline_offset(conn):
        row = conn.execute(
            'SELECT value FROM analytics_meta WHERE key = ?',
            (VISITOR_BASELINE_OFFSET_KEY,),
        ).fetchone()
        if not row:
            return 0
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _apply_visitor_baseline(raw_unique_visitors, conn):
        return raw_unique_visitors + CounterHandler._read_visitor_baseline_offset(conn)

    @staticmethod
    def set_visitor_total(display_total):
        """Set the public visitor total without creating fake visitor rows."""
        if display_total < 0:
            raise ValueError('Visitor total must be non-negative')

        conn = CounterHandler._ensure_visitor_db()
        try:
            raw_unique_visitors = CounterHandler._read_raw_unique_visitors(conn)
            baseline_offset = int(display_total) - raw_unique_visitors
            conn.execute(
                '''
                INSERT INTO analytics_meta (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                ''',
                (VISITOR_BASELINE_OFFSET_KEY, str(baseline_offset), int(time.time())),
            )
            conn.commit()
            return {
                'unique_visitors': display_total,
                'raw_unique_visitors': raw_unique_visitors,
                'visitor_baseline_offset': baseline_offset,
            }
        finally:
            conn.close()

    @staticmethod
    def read_analytics(should_track=True, cookie_header=''):
        """Return downloads + unique visitors, optionally recording this visit."""
        downloads = CounterHandler.read_counter()
        set_cookie_header = None

        if should_track:
            visitor_id = CounterHandler._parse_visitor_cookie(cookie_header)
            if not visitor_id:
                visitor_id = secrets.token_urlsafe(32)
                set_cookie_header = CounterHandler._build_visitor_cookie(visitor_id)
            unique_visitors = CounterHandler._record_visitor(visitor_id)
        else:
            unique_visitors = CounterHandler._read_unique_visitors()

        return {
            'downloads': downloads,
            'unique_visitors': unique_visitors,
        }, set_cookie_header

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--refresh-once',
        action='store_true',
        help='Refresh the GitHub release download-count cache once, then exit.',
    )
    parser.add_argument(
        '--set-visitor-total',
        type=int,
        help='Set the public unique visitor total by storing a baseline offset.',
    )
    args = parser.parse_args()

    if args.set_visitor_total is not None:
        try:
            result = CounterHandler.set_visitor_total(args.set_visitor_total)
            print(json.dumps(result))
            sys.exit(0)
        except Exception as e:
            print(f"Error setting visitor total: {e}", file=sys.stderr)
            sys.exit(1)

    if args.refresh_once:
        try:
            refreshed = CounterHandler.refresh_counter_cache()
            print(json.dumps(refreshed))
            sys.exit(0)
        except Exception as e:
            print(f"Error refreshing GitHub release counter: {e}", file=sys.stderr)
            sys.exit(1)

    # Run on localhost:8888
    server = HTTPServer(('127.0.0.1', 8888), CounterHandler)
    print('Counter API running on http://127.0.0.1:8888')
    server.serve_forever()
