FROM node:20-alpine

WORKDIR /app

COPY package.json yarn.lock vite.config.ts index.html  ./

RUN yarn install

COPY ./src ./src

EXPOSE 5173

CMD ["yarn", "dev", "--host"]