# Deployment Guide

This guide explains how to deploy the Sort Moments landing page to Hetzner and configure the sortmoments.com domain via GoDaddy.

## Overview

- **Hosting**: Hetzner Cloud (CX11 - cheapest option ~€3.29/month)
- **Domain**: sortmoments.com via GoDaddy
- **Web Server**: Nginx
- **SSL**: Let's Encrypt (free)

---

## Part 1: Hetzner Server Setup

### Step 1: Create a Hetzner Cloud Account

1. Go to [Hetzner Cloud](https://console.hetzner.cloud/)
2. Create an account and add a payment method

### Step 2: Create a Server

1. Click "Add Server"
2. **Location**: Choose closest to your users (e.g., Ashburn for US, Falkenstein for EU)
3. **Image**: Ubuntu 22.04
4. **Type**: CX11 (2 vCPU, 2GB RAM) - cheapest option
5. **SSH Key**: Add your SSH public key (recommended) or use password
6. **Name**: `sortmoments`
7. Click "Create & Buy Now"

### Step 3: Note Your Server IP

After creation, note the IPv4 address (e.g., `123.45.67.89`). You'll need this for DNS.

### Step 4: Connect to Your Server

```bash
ssh root@YOUR_SERVER_IP
```

### Step 5: Initial Server Setup

```bash
# Update system
apt update && apt upgrade -y

# Install Nginx
apt install nginx -y

# Install Certbot for SSL
apt install certbot python3-certbot-nginx -y

# Enable firewall
ufw allow 'Nginx Full'
ufw allow OpenSSH
ufw enable
```

### Step 6: Create Website Directory

```bash
# Create web directory
mkdir -p /var/www/sortmoments

# Set permissions
chown -R www-data:www-data /var/www/sortmoments
chmod -R 755 /var/www/sortmoments

# Create downloads directory for the .exe
mkdir -p /var/www/sortmoments/downloads
```

### Step 7: Upload Website Files

From your local machine:

```bash
# Upload the website folder contents
scp -r website/* root@YOUR_SERVER_IP:/var/www/sortmoments/

# Upload the executable (when ready)
scp dist/SortMoments.exe root@YOUR_SERVER_IP:/var/www/sortmoments/downloads/
```

### Step 8: Configure Nginx

```bash
# Create Nginx config
nano /etc/nginx/sites-available/sortmoments
```

Paste this configuration:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name sortmoments.com www.sortmoments.com;
    root /var/www/sortmoments;
    index index.html;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript;

    # Main location
    location / {
        try_files $uri $uri/ =404;
    }

    # Downloads - allow large files
    location /downloads {
        add_header Content-Disposition 'attachment';
        # Increase timeout for large downloads
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }

    # Cache static assets
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|woff|woff2)$ {
        expires 7d;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable the site:

```bash
# Enable site
ln -s /etc/nginx/sites-available/sortmoments /etc/nginx/sites-enabled/

# Remove default site
rm /etc/nginx/sites-enabled/default

# Test config
nginx -t

# Reload Nginx
systemctl reload nginx
```

---

## Part 2: GoDaddy DNS Configuration

### Step 1: Log into GoDaddy

1. Go to [GoDaddy](https://godaddy.com) and log in
2. Go to "My Products" → find sortmoments.com → "DNS"

### Step 2: Configure DNS Records

Delete existing A records and add:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | @ | YOUR_SERVER_IP | 600 |
| A | www | YOUR_SERVER_IP | 600 |

Replace `YOUR_SERVER_IP` with your Hetzner server's IP address.

### Step 3: Wait for DNS Propagation

DNS changes can take 5-30 minutes to propagate. You can check status at [whatsmydns.net](https://www.whatsmydns.net/)

---

## Part 3: SSL Certificate Setup

Once DNS is pointing to your server:

```bash
# Get SSL certificate
certbot --nginx -d sortmoments.com -d www.sortmoments.com

# Follow the prompts:
# - Enter email address
# - Agree to terms
# - Choose whether to redirect HTTP to HTTPS (recommended: Yes)
```

Certbot will:
- Obtain certificates from Let's Encrypt
- Configure Nginx to use HTTPS
- Set up automatic renewal

### Verify SSL Auto-Renewal

```bash
# Test renewal process
certbot renew --dry-run
```

---

## Part 4: Upload the Executable

### Initial Upload

```bash
# From your local machine
scp dist/SortMoments.exe root@YOUR_SERVER_IP:/var/www/sortmoments/downloads/
```

### Set Permissions

```bash
# On the server
chmod 644 /var/www/sortmoments/downloads/SortMoments.exe
```

### Updating the Executable

When you have a new version:

```bash
# From local machine
scp dist/SortMoments.exe root@YOUR_SERVER_IP:/var/www/sortmoments/downloads/
```

---

## Part 5: Verification Checklist

After deployment, verify:

- [ ] http://sortmoments.com redirects to https://sortmoments.com
- [ ] https://sortmoments.com loads the landing page
- [ ] https://www.sortmoments.com works (redirects to non-www)
- [ ] Download button works: https://sortmoments.com/downloads/SortMoments.exe
- [ ] Download counter displays (may show 0 initially)
- [ ] GitHub links work
- [ ] SSL certificate is valid (check for padlock in browser)

---

## Part 6: Adding the Demo Video

When you have the demo video ready:

1. Upload to the server:
```bash
scp your-demo-video.mp4 root@YOUR_SERVER_IP:/var/www/sortmoments/assets/demo.mp4
```

2. Update `index.html` on the server - uncomment the video section and comment out the placeholder:
```bash
nano /var/www/sortmoments/index.html
```

Find and update the video section:
```html
<!-- Remove or comment out the placeholder -->
<!-- <div class="video-placeholder" id="videoPlaceholder">...</div> -->

<!-- Uncomment the video element -->
<video id="demoVideo" controls poster="assets/video-poster.jpg">
    <source src="assets/demo.mp4" type="video/mp4">
</video>
```

---

## Maintenance Commands

### Check Nginx Status
```bash
systemctl status nginx
```

### View Nginx Logs
```bash
# Access logs
tail -f /var/log/nginx/access.log

# Error logs
tail -f /var/log/nginx/error.log
```

### Restart Nginx
```bash
systemctl restart nginx
```

### Check Disk Space
```bash
df -h
```

### Renew SSL Certificate (manual)
```bash
certbot renew
```

---

## Cost Summary

| Service | Cost |
|---------|------|
| Hetzner CX11 | ~€3.29/month |
| Domain (GoDaddy) | ~$12-20/year |
| SSL (Let's Encrypt) | Free |
| **Total** | **~$5-6/month** |

---

## Troubleshooting

### Site not loading after DNS change
- Wait 5-30 minutes for DNS propagation
- Check DNS at whatsmydns.net
- Verify Nginx is running: `systemctl status nginx`

### SSL certificate errors
- Ensure DNS is pointing to server before running certbot
- Check certificate status: `certbot certificates`

### 403 Forbidden error
- Check file permissions: `ls -la /var/www/sortmoments/`
- Fix: `chown -R www-data:www-data /var/www/sortmoments`

### Download not working
- Check file exists: `ls -la /var/www/sortmoments/downloads/`
- Check Nginx error log: `tail /var/log/nginx/error.log`

### Large file download timing out
- The Nginx config includes increased timeouts
- For very slow connections, users can use download managers
