FROM --platform=linux/amd64 node:20

WORKDIR /app

ADD package.json package.json
RUN npm install

ADD src src
ADD start.sh start.sh
ADD tsconfig.json tsconfig.json

RUN chmod +x start.sh

RUN npm run build

RUN mkdir -p /temp
RUN chmod 755 /temp

#CMD /app/start.sh