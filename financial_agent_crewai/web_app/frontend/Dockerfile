FROM node:20-alpine

WORKDIR /app

# Copy configuration files
COPY package.json yarn.lock vite.config.ts ./

# Install dependencies
RUN yarn install

# Copy source code
COPY ./src ./src
COPY index.html  ./

EXPOSE 5173

CMD ["yarn", "dev", "--host"]