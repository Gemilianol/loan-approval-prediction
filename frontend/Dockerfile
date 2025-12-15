# ==========================================
# STAGE 1: Compile your Frontend
# ==========================================
FROM node:18-alpine AS builder

WORKDIR /app

# 1. Install Dependencies
# We copy package.json first to leverage Docker caching
COPY package.json package-lock.json ./
# ci means Clean Install. Only see package-lock.json
RUN npm ci 

COPY . .
# Build the distr instead of create physical files on the disk. (npm run dev)
RUN npm run build

# This process takes your React code and compiles it down to a small 
# set of static files (HTML, CSS, and JavaScript) and places them 
# in the /app/dist folder (your static files).

# ==========================================
# STAGE 2: The Web Server
# ==========================================
# When Docker sees a second FROM command, it resets.
FROM nginx:alpine

# 3. The Transfer
# We reach back into the "builder" stage and grab ONLY the output folder.
# We place it in the standard Nginx hosting directory. /app/dist (origin) → /usr/share/nginx/html (destination)
COPY --from=builder /app/dist /usr/share/nginx/html

# In this way, we replaced the default nginx files for our files.
# Then, the default configuration of nginx will fail so we need to adapt accordingly. 

# Copy our custom Nginx config (The Routing Rules)
# We replace the 'default.conf' provided by the Nginx image.
# HINT: Structural logic (like "how to handle a React route") is usually baked
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 (Standard web traffic)
EXPOSE 80

# Start nginx (default command)
# This is the most critical part for Docker. 
# By default, Nginx runs as a background "daemon." 
# If it did that, Docker would think the main process finished 
# and would immediately shut down the container. 
# This flag tells Nginx to run in the foreground, which keeps the container alive and serving your app.
CMD ["nginx", "-g", "daemon off;"]