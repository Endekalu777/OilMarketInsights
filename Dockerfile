# Stage 1: Build the React frontend
FROM node:16 AS frontend-builder
WORKDIR /app/frontend
COPY oil_price_dashboard/frontend/package*.json ./
RUN npm install
COPY oil_price_dashboard/frontend ./
RUN npm run build

# Stage 2: Set up the Flask backend
FROM python:3.9-slim AS backend
WORKDIR /app/backend
COPY oil_price_dashboard/backend/requirements.txt ./
RUN pip install -r requirements.txt
COPY oil_price_dashboard/backend ./

# Stage 3: Final container (NGINX for serving frontend + Flask for backend)
FROM nginx:alpine
COPY --from=frontend-builder /app/frontend/build /usr/share/nginx/html
COPY --from=backend /app/backend /app/backend
WORKDIR /app/backend
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
