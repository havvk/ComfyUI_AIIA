#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# Check if required variables are set
if [ -z "$DEPLOY_USER" ] || [ -z "$DEPLOY_HOST" ] || [ -z "$DEPLOY_PATH" ]; then
    echo "Error: Missing required environment variables (DEPLOY_USER, DEPLOY_HOST, DEPLOY_PATH)."
    exit 1
fi

echo "Deploying to ${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH}..."

# Run rsync
# Excluding git, pycache, DS_Store, and python bytecode
rsync -avz --exclude '__pycache__' \
           --exclude '.git' \
           --exclude '.DS_Store' \
           --exclude '*.pyc' \
           --exclude '.env' \
           --exclude '.vscode' \
           --exclude '.gemini' \
           . "${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH}"

echo "Deployment complete."
