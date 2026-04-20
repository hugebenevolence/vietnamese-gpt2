FROM node:20-alpine

WORKDIR /app

COPY ui/package.json ./package.json
RUN npm install

COPY ui .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0", "-p", "3000"]
