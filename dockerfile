FROM node:latest

WORKDIR /app

COPY . .

RUN npm install
RUN npm install -g nodemon mongoose
RUN npm install bcryptjs
RUN npm i /app/frontend
RUN npm install -D tailwindcss postcss autoprefixer /app/frontend
RUN npx tailwindcss init -p /app/frontend
RUN	npm i react-router-dom /app/frontend
RUN npm i react-icons /app/frontend


# Command to keep the container running indefinitely
CMD ["tail", "-f", "/dev/null"]