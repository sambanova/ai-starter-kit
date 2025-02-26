FROM node:20-alpine

WORKDIR /app

COPY package.json yarn.lock  ./

RUN yarn install

COPY ./src ./src

EXPOSE 5173

CMD ["yarn", "dev", "--host"]