# filepath: /mnt/c/Users/danik/phd/main/docker/osrm/Dockerfile
FROM osrm/osrm-backend:latest

# Expose port
EXPOSE 5000

# Copy the OSM data
COPY docker/data/osrm/data/ /data/

# Copy the startup script
COPY docker/data/osrm/scripts/start-osrm.sh /usr/local/bin/start-osrm.sh

# Make the script executable
RUN chmod +x /usr/local/bin/start-osrm.sh

# Set the script as the default command
CMD [ "/usr/local/bin/start-osrm.sh" ]