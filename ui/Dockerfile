FROM node:20-alpine

# Set working dir
WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy app source
COPY . .

# Expose the dev server port
EXPOSE 3000

# Start the dev server
CMD ["npm", "start"]
